#  Copyright 2015 Rouslan Korneychuk
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import copy
import ast
import symtable
import os
import io
import sys
from functools import partial
from collections import namedtuple, OrderedDict

from . import pyinternals
from . import astloader
from . import debug
from . import abi
from .code_gen import *


DUMP_OBJ_FILE = False
NODE_COMMENTS = True


BINOP_FUNCS = {
    ast.Add : 'PyNumber_Add',
    ast.Sub : 'PyNumber_Subtract',
    ast.Mult : 'PyNumber_Multiply',
    ast.Div : 'PyNumber_TrueDivide',
    ast.LShift : 'PyNumber_Lshift',
    ast.RShift : 'PyNumber_Rshift',
    ast.BitOr : 'PyNumber_Or',
    ast.BitXor : 'PyNumber_Xor',
    ast.BitAnd : 'PyNumber_And',
    ast.FloorDiv : 'PyNumber_FloorDivide'}


class UtilityFunctions:
    def __init__(self):
        self.local_name = JumpTarget()
        self.global_name = JumpTarget()
        self.prepare_exc_handler = JumpTarget()
        self.reraise_exc_handler = JumpTarget()
        self.unwind_exc = JumpTarget()
        self.unwind_finally = JumpTarget()
        self.swap_exc_state = JumpTarget()


def ast_and_symtable_map(code,filename,compile_type):
    ast,tables = astloader.symtable(code,filename,compile_type)

    return ast,{id : symtable._newSymbolTable(tab,filename) for id,tab in tables.items()}


def create_uninitialized(t):
    return t.__new__(t)

def check_context_load(node):
    if not isinstance(node.ctx,ast.Load):
        raise NCompileError('{} node has assign/delete context type but is not in assign/delete statement'.format(node.__class__.__name__))

class _ParsedValue:
    """A fake AST node to allow inserting arbitrary values into AST trees"""
    
    def __init__(self,value):
        self.value = value

class ExprCompiler(ast.NodeVisitor):
    def __init__(self,code,s_table=None,s_map=None,util_funcs=None,global_scope=False,entry_points=None):
        self.code = code
        self.stable = s_table
        self.sym_map = s_map
        self.util_funcs = util_funcs
        self.global_scope = global_scope
        self.entry_points = entry_points
        self.consts = {}
        self.names = OrderedDict()
        self.local_addrs = None
        self.local_addr_start = None
        self.visit_depth = 0
    
    @property
    def abi(self):
        return self.code.abi
    
    def is_local(self,x):
        return (not self.global_scope) and x.is_local()

    def swap(self,a,b):
        tmp = self.unused_reg()
        (self.code
            .mov(a,tmp1)
            .mov(b,a)
            .mov(tmp1,b))
        return b,a
    
    def allocate_locals(self,args):
        assert (self.local_addrs is None
            and self.stable is not None
            and (isinstance(self.stable,symtable.Function) != (args is None))
            and (isinstance(self.stable,symtable.Function) == self.stable.is_optimized()))

        self.local_addrs = {}
        if isinstance(self.stable,symtable.Function):
            # FUNCTION_BODY_NAME_ORDER
            # the order of the arguments must match the order in
            # pyinternals.FunctionBody.names which is described in
            # pyinternals.c
            
            locals = [a.arg for a in args.args]
            if args.vararg: locals.append(args.vararg.arg)
            locals.extend(a.arg for a in args.kwonlyargs)
            if args.kwarg: locals.append(args.kwarg.arg)
            
            the_rest = set(self.stable.get_locals())
            the_rest.difference_update(locals)
            locals.extend(the_rest)
            
            for n in locals:
                self.names[n] = address_of(n)
            
            tmp = self.code.unused_reg()
            self.code.mov(0,tmp)
 
            # locals is reversed because each stack value has a lower address
            # than the previous value
            for loc in reversed(locals):
                addr = self.code.new_stack_value(True)(self.code)
                self.code.mov(tmp,addr)
                # TODO: create annotation for local
                self.local_addrs[loc] = addr
            
            self.local_addr_start = addr
    
    def deallocate_locals(self,s=None):
        assert self.local_addrs is not None
        
        if s is None: s = self.code
        
        for loc in self.local_addrs.values():
            (s
                .mov(loc,R_RET)
                .if_(R_RET)
                    .decref(R_RET)
                .endif())
        
        self.local_addrs = None
        self.local_addr_start = None
    
    def do_kw_args(self,args,func_self,func_kwds,with_dict,without_dict):
        assert args.kwonlyargs or with_dict
        
        miss_flag = self.code.new_stack_value(True)
        
        def mark_miss(s):
            # miss_flag needs to be set to any non-zero value and using a
            # register instead of an immediate value results in shorter
            # machine code
            s.mov(R_PRES1,miss_flag)
        
        hit_count = self.code.new_stack_value(True)
        
        def mark_hit(name):
            def inner(s):
                if args.kwarg:
                    (s
                        .invoke('PyDict_DelItem',R_PRES1,name)
                        .check_err(True))
                else:
                    s.add(1,hit_count)
            
            return inner
        
        s = (self.code
            .mov(0,miss_flag)
            .mov(0,hit_count)
            .mov(func_kwds,R_RET)
            .if_(R_RET)
                .mov(len(args.args),R_PRES3)
                .sub(CType('PyVarObject',R_PRES2).ob_size,R_PRES3))
        
        if args.kwarg:
            (s
                .invoke('PyDict_Copy',R_RET)
                .check_err()
                .mov(R_RET,R_PRES1)
                .mov(R_RET,self.get_name(args.kwarg.arg)))
        else:
            s.mov(R_RET,R_PRES1)
        
        if with_dict: with_dict(s,mark_miss,mark_hit)
        
        if args.kwonlyargs:
            (s
                .mov(func_self,R_PRES2)
                .mov(CType('Function',R_PRES2).kwdefaults,R_PRES2))
            
            for i,a in enumerate(args.kwonlyargs):
                name = self.get_name(a.arg)
                no_def = JumpTarget()
                
                s = (s
                    .invoke('PyDict_GetItem',R_PRES1,name)
                    .if_(R_RET)
                        .mov(R_RET,self.local_addrs[a.arg])
                        .incref(R_RET)
                        [mark_hit(name)]
                    .elif_(R_PRES2)
                        .mov(0,R_SCRATCH1)
                        .mov(addr(i*self.abi.ptr_size,R_PRES2),R_RET)
                        .test(R_RET,R_RET)
                        .jz(no_def)
                        .mov(R_RET,self.local_addrs[a.arg])
                        .incref(R_RET)
                    .else_()(no_def)
                        [mark_miss]
                    .endif())
        
        if not args.kwarg:
            (s
                .invoke('PyDict_Size',R_PRES1)
                .if_(signed(R_RET) > hit_count)
                   .invoke('excess_keyword',func_self,R_PRES1)
                   .exc_cleanup()
                .endif())
        
        s.else_()
        
        if args.kwarg:
            (s
                .invoke('PyDict_New')
                .check_err()
                .mov(R_RET,self.get_name(args.kwarg.arg)))
        
        if without_dict: without_dict(s,mark_miss)
        
        if args.kwonlyargs:
            s = (s
                .mov(func_self,R_PRES2)
                .mov(CType('Function',R_PRES2).kwdefaults,R_PRES2)
                .if_(R_PRES2))
            
            no_def = JumpTarget()
            
            for i,a in enumerate(args.kwonlyargs):
                name = self.get_name(a.arg)
                
                (s
                    .mov(addr(i*self.abi.ptr_size,R_PRES2),R_RET)
                    .test(R_RET,R_RET)
                    .jz(no_def)
                    .mov(R_RET,self.local_addrs[a.arg])
                    .incref(R_RET))
            
            s = (s
                .else_()(no_def)
                    [mark_miss]
                .endif())
        
        r_arg = s.arg_reg(R_RET,1)
        (s
            .endif()
            .lea(self.local_addr_start,r_arg)
            .if_(miss_flag)
                .invoke('missing_arguments',func_self,r_arg,tmpreg=R_SCRATCH1)
                .exc_cleanup()
            .endif())
        
        hit_count.discard(s)
        miss_flag.discard(s)
    
    def handle_args(self,args,func_self,func_args,func_kwds):
        assert self.local_addrs is not None
        
        if args.args:
            # a short-cut to the part where positional arguments are moved
            arg_target1 = JumpTarget()
            
            s = (self.code
                .mov(func_self,R_PRES2)
                .mov(func_args,R_PRES1)
                .mov(len(args.args),R_PRES3)
                .sub(CType('PyVarObject',R_PRES1).ob_size,R_PRES3)
                .mov(CType('Function',R_PRES2).defaults,R_PRES2)
                .if_cond(TEST_L)
                    .neg(R_PRES3))
             
            # if there are more arguments than parameters
            if args.vararg:
                
                dest_item = tuple_item(R_RET,0)(c)
                dest_item.index = R_PRES3
                dest_item.scale = self.abi.ptr_size
                
                src_item = tuple_item(R_PRES1,len(args.args))(c)
                src_item.index = R_PRES3
                src_item.scale = self.abi.ptr_size
                
                (s
                        .invoke('PyTuple_New',len(node.args))
                        .check_err()
                        .do()
                            .sub(1,R_PRES3)
                            .mov(src_item,R_RET)
                            .mov(R_RET,dest_item)
                            .incref(R_RET)
                        .while_(R_PRES3)
                        .mov(R_RET,self.local_addrs[args.vararg.arg])
                        .jmp(arg_target1)
                    .else_()
                        .mov(self.get_const(()),self.local_addrs[args.vararg.arg]))
            else:
                (s
                    .invoke('too_many_positional',func_self,R_PRES3,func_kwds)
                    .exc_cleanup())
            
            s.endif()
            
            # set positional arguments
            
            targets = []
            
            (self.code
                .jump_table(R_PRES3,targets,R_SCRATCH2,R_RET))
            
            for i,a in reversed(list(enumerate(args.args))):
                target = JumpTarget() if targets else arg_target1
                targets.append(target)
                
                (self.code
                    (target)
                    .mov(tuple_item(R_PRES1,i),R_RET)
                    .mov(R_RET,self.local_addrs[a.arg])
                    .incref(R_RET))
            
            target = JumpTarget()
            targets.append(target)
            self.code(target)
            
            # set keyword arguments
            
            def with_dict(s,mark_miss,mark_hit):
                default_item = tuple_item(R_PRES2,0)
                default_item.index = R_RET
                default_item.scale = self.abi.ptr_size
                
                c_func_self = CType('Function',R_PRES1)
                
                for i,a in enumerate(args.args):
                    name = self.get_name(a.arg)
                    s = (s
                        .invoke('PyDict_GetItem',R_PRES1,name)
                        .if_(R_RET)
                            .if_(self.local_addrs[a.arg])
                                .mov(func_self,R_PRES1)
                                .invoke('PyErr_Format',
                                    'PyExc_TypeError',
                                    'DUPLICATE_VAL_MSG',
                                    c_func_self.name,
                                    name)
                                .exc_cleanup()
                            .endif()
                            .mov(R_RET,self.local_addrs[a.arg])
                            .incref(R_RET)
                            [mark_hit(name)]
                        .else_()
                            .mov(i,R_RET)
                            .sub(R_PRES3,R_RET)
                            .if_cond(TEST_GE)
                                .mov(default_item,R_RET)
                                .mov(R_RET,self.local_addrs[a.arg])
                                .incref(R_RET)
                            .else_()
                                [mark_miss]
                            .endif()
                        .endif())
            
            def without_dict(s,mark_miss):
                # just set the defaults.
                
                targets = []
                
                (s
                    .mov(len(args.args),R_RET)
                    .sub(R_PRES3,R_RET)
                    .sub(CType('PyVarObject',R_PRES2).ob_size,R_SCRATCH1)
                    .if_(signed(R_RET) < R_SCRATCH1)
                        [mark_miss]
                        .mov(R_SCRATCH1,R_RET)
                    .endif()
                    .jump_table(R_RET,targets,R_SCRATCH2,R_RET))
                
                for i,a in enumerate(args.args):
                    target = JumpTarget()
                    targets.append(target)
                    
                    (s
                        (target)
                        .mov(tuple_item(R_PRES2,i),R_RET)
                        .mov(R_RET,self.local_addrs[a.arg])
                        .incref(R_RET))
                
                target = JumpTarget()
                targets.append(target)
                s(target)
            
            self.do_kw_args(args,func_self,func_kwds,with_dict,without_dict)
        else:
            if args.vararg:
                (self.code
                    .mov(func_args,R_RET)
                    .mov(R_RET,self.local_addrs[args.vararg.arg])
                    .incref(R_RET))
            else:
                (self.code
                    .mov(func_args,R_RET)
                    .if_(CType('PyVarObject',R_RET).ob_size)
                        .invoke('too_many_positional',func_self,R_RET,func_kwds)
                        .exc_cleanup()
                    .endif())

            if args.kwonlyargs:
                self.do_kw_args(args,func_self,func_kwds,None,None)
            elif args.kwarg:
                (self.code
                    .mov(func_kwds,R_RET)
                    .if_(R_RET)
                        .invoke('PyDict_Copy',R_RET)
                    .else_()
                        .invoke('PyDict_New')
                    .check_err()
                    .mov(R_RET,self.local_addrs[args.kwarg.arg])
                    .endif())
            else:
                (self.code
                    .mov(func_kwds,R_PRES1)
                    .if_(R_PRES1)
                        .invoke('PyDict_Size',R_PRES1)
                        .if_(R_RET)
                            .invoke('excess_keyword',func_self,R_PRES1)
                        .endif()
                    .endif())
    
    def basic_binop(self,func,arg1,arg2):
        (self.code
            .invoke(func,arg1,arg2)
            .check_err())
        
        r = ScratchValue(self.code,R_RET)
        
        arg1.discard(self.code)
        arg2.discard(self.code)

        return r

    def get_const(self,val):
        c = self.consts.get(val)
        if c is None:
            c = address_of(val)
            self.consts[val] = c
        return ConstValue(c)
    
    def get_name(self,n):
        c = self.names.get(n)
        if c is None:
            c = address_of(n)
            self.names[n] = c
        return ConstValue(c)
    
    
    #### VISITOR METHODS ####
    
    if NODE_COMMENTS:
        @staticmethod
        def node_descr(node):
            r = node.__class__.__name__
            if isinstance(node,ast.BinOp):
                r += ' ({})'.format({
                    ast.Add : '+',
                    ast.Sub : '-',
                    ast.Mult : '*',
                    ast.Div : '/',
                    ast.Mod : '%',
                    ast.Pow : '**',
                    ast.LShift : '<<', 
                    ast.RShift : '>>',
                    ast.BitOr : '|',
                    ast.BitXor : '^',
                    ast.BitAnd : '&',
                    ast.FloorDiv : '//'}[node.op.__class__])
            return r
        
        def visit(self,node):
            self.code.comment('{}node {} {{'.format('  '*self.visit_depth,self.node_descr(node)))
            self.visit_depth += 1
            r = super().visit(node)
            self.visit_depth -= 1
            self.code.comment('  '*self.visit_depth + '}')
            return r
    
    def _visit_slice(self,slice):
        if isinstance(slice,ast.Index):
            return self.visit(slice.value)
        
        if isinstance(slice,ast.Slice):
            start = None
            if slice.lower:
                start = self.visit(slice.lower)
            
            end = None
            if slice.upper:
                end = self.visit(slice.upper)
            
            step = None
            if slice.step:
                step = self.visit(slice.step)
            
            (self.code
                .invoke('PySlice_New',
                    start or 0,
                    end or 0,
                    step or 0)
                .check_err())
            
            r = ScratchValue(self.code,R_RET)
            
            if start: start.discard(self.code)
            if end: end.discard(self.code)
            if step: step.discard(self.code)
            
            return r
        
        if isinstance(slice,ast.ExtSlice):
            raise NotImplementedError()
        
        raise NCompileError('invalid index type in subscript')
    
    def visit__ParsedValue(self,node):
        return node.value
    
    def generic_visit(self,node):
        # all types need their own handler
        raise NotImplementedError()
    
    def visit_Module(self,node):
        for stmt in node.body: self.visit(stmt)
        (self.code
            .mov(self.get_const(None),R_RET)
            .incref(R_RET))
    
    def visit_FunctionDef(self,node):
        func = create_uninitialized(pyinternals.FunctionBody)
        
        compile_eval(node,self.sym_map,self.abi,self.util_funcs,self.entry_points,False,node.name,func,node.args)
        
        fobj = self.get_const(func)
        
        # the decorator application and name assignment can be broken down into
        # other AST nodes
        
        alt_node = _ParsedValue(fobj)
        for d in reversed(node.decorator_list):
            alt_node = ast.Call(d,[alt_node],[],None,None,lineno=d.lineno,col_offset=d.col_offset)
        
        self.visit(ast.Assign([ast.Name(node.name,ast.Store())],alt_node,lineno=node.lineno,col_offset=node.col_offset))
    
    def visit_Return(self,node):
        if node.value:
            val = self.visit(node.value)
        else:
            val = self.get_const(None)
        
        for ld in self.code.state.stack:
            if ld is not None and ld[1]:
                ld[0].discard(self.code)
        
        loc = val(self.code)
        val.steal(self.code)
        self.code.mov(loc,R_RET)
    
    def visit_Assign(self,node):
        expr = self.visit(node.value)
        
        for t in node.targets:
            assert isinstance(t.ctx,ast.Store)
            
            if isinstance(t,ast.Name):
                s = self.stable.lookup(t.id)
        
                if self.is_local(s):
                    if self.stable.is_optimized():
                        # TODO: this code checks to see if the local has a
                        # value and free it if it does. This check could be
                        # eliminated in most cases if we were to track
                        # assignments and deletions (it would still be required
                        # if whether the local is assigned to, depended on a
                        # branch, and inside exception handlers).
                        
                        tmp1,r_tmp2 = self.code.unused_regs(2)
                        
                        item = self.local_addrs[t.id]
                        self.code.mov(item,tmp1)
                        
                        tmp2 = expr(self.code)
                        if not isinstance(tmp2,self.abi.Register):
                            tmp2 = r_tmp2
                            self.code.mov(expr,tmp2)
                        
                        if expr.owned:
                            expr.owned = False
                        else:
                            self.code.incref(tmp2)
                        
                        (self.code
                            .mov(tmp2,item)
                            .if_(tmp1)
                                .decref(tmp1)
                            .endif())
                    else:
                        dict_addr = self.code.fit_addr('PyDict_Type',R_SCRATCH2)
                        
                        (self.code
                            .push_arg(expr,n=2)
                            .push_arg(self.get_name(t.id),n=1)
                            .mov(self.code.special_addrs['locals'],R_RET)
                            .if_(not_(R_RET))
                                .clear_args()
                                .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_STORE_MSG')
                                .exc_cleanup()
                            .endif()
                            .mov(dict_addr,R_SCRATCH2)
                            .mov('PyObject_SetItem',R_SCRATCH1)
                            .push_arg(R_RET,n=0)
                            .if_(R_SCRATCH2 == type_of(R_RET))
                                .mov('PyDict_SetItem',R_SCRATCH1)
                            .endif()
                            .call(R_SCRATCH1)
                            .check_err(True))
                else:
                    (self.code
                        .invoke('PyDict_SetItem',self.code.special_addrs['globals'],self.get_name(t.id),expr)
                        .check_err(True))
            
            elif isinstance(t,ast.Attribute):
                obj = self.visit(t.value)
                
                (self.code
                    .invoke('PyObject_SetAttr',obj,self.get_name(t.attr),expr)
                    .check_err(True))
                
                obj.discard(self.code)
            
            elif isinstance(t,ast.Subscript):
                obj = self.visit(t.value)
                slice = self._visit_slice(t.slice)
                
                (self.code
                    .invoke('PyObject_SetItem',obj,slice,expr)
                    .check_err(True))
                
                slice.discard(self.code)
                obj.discard(self.code)
            
            elif isinstance(t,ast.Starred):
                raise NotImplementedError()
            elif isinstance(t,ast.List):
                raise NotImplementedError()
            elif isinstance(t,ast.Tuple):
                raise NotImplementedError()
            else:
                raise NCompileError('invalid expression in assignment')
        
        expr.discard(self.code)
    
    def visit_If(self,node):
        test = self.visit(node.test)
        self.code.invoke('PyObject_IsTrue',test)
        test.discard(self.code,preserve_reg=R_RET)
        
        self.code = (self.code
            .test(R_RET,R_RET)
            .if_cond(TEST_L)
                .exc_cleanup()
            .endif()
            .if_cond(TEST_NZ))
        
        for stmt in node.body:
            self.visit(stmt)
        
        if node.orelse:
            self.code.else_()
            for stmt in node.orelse:
                self.visit(stmt)
        
        self.code = self.code.endif()
    
    def visit_Expr(self,node):
        self.visit(node.value).discard(self.code)
    
    def visit_BoolOp(self,node):
        raise NotImplementedError()
    def visit_BinOp(self,node):
        arg1 = self.visit(node.left)
        arg2 = self.visit(node.right)
        
        op = type(node.op)
        
        if op is ast.Add:
            raise NotImplementedError()
        
        if op is ast.Mod:
            with self.code.reg_temp_for(arg1) as r_arg1:
                func,r_uaddr = self.code.unused_regs(2)
                r_arg1 = r_arg1.reg
            
            uaddr = self.code.fit_addr('PyUnicode_Type',r_uaddr)
            
            (self.code
                .mov('PyUnicode_Format',func)
                .if_(uaddr != type_of(r_arg1))
                    .mov('PyNumber_Remainder',func)
                .endif()
                .invoke(func,r_arg1,arg2)
                .check_err())
        elif op is ast.Pow:
            (self.code
                .invoke(func,arg2,arg1,'Py_None')
                .check_err())
        else:
            return self.basic_binop(BINOP_FUNCS[op],arg1,arg2)
    
        r = ScratchValue(self.code,R_RET)
        
        arg1.discard(self.code)
        arg2.discard(self.code)
        
        return r
    
    def visit_UnaryOp(self,node):
        arg = self.visit(node.operand)
        
        op = type(node.op)
        if op is ast.Not:
            (self.code
                .invoke('PyObject_IsTrue',arg)
                .test(R_RET,R_RET)
                .mov('Py_True',R_RET)
                .if_cond(TEST_G)
                    .mov('Py_False',R_RET)
                .elif_cond(TEST_NZ)
                    .exc_cleanup()
                .endif()
                .incref())
        else:
            if op is ast.Invert:
                func = 'PyNumber_Invert'
            elif op is ast.UAdd:
                func = 'PyNumber_Positive'
            elif op is ast.USub:
                func = 'PyNumber_Negative'
            else:
                raise NCompileError('unrecognized unary operation type encountered')
            
            self.code.invoke(func,arg).check_err()
            
        r = ScratchValue(self.code,R_RET)
        arg.discard(self.code)
        return r
        
    
    def visit_Lambda(self,node):
        raise NotImplementedError()
    def visit_IfExp(self,node):
        raise NotImplementedError()
    def visit_Dict(self,node):
        (self.code
            .invoke('_PyDict_NewPresized',arg)
            .check_err())
        
        r = ScratchValue(self.code,R_RET)
        
        for k,v in zip(node.keys,node.values):
            kobj = self.visit(k)
            vobj = self.visit(v)
            
            (self.code
                .invoke('PyDict_SetItem',r,kobj,vobj)
                .check_err(True))
            
            kobj.discard(self.code)
            vobj.discard(self.code)
        
        return r
    
    def visit_Set(self,node):
        raise NotImplementedError()
    def visit_ListComp(self,node):
        raise NotImplementedError()
    def visit_SetComp(self,node):
        raise NotImplementedError()
    def visit_DictComp(self,node):
        raise NotImplementedError()
    def visit_GeneratorExp(self,node):
        raise NotImplementedError()
    def visit_Yield(self,node):
        raise NotImplementedError()
    def visit_YieldFrom(self,node):
        raise NotImplementedError()
    def visit_Compare(self,node):
        raise NotImplementedError()
    def visit_Call(self,node):
        fobj = self.visit(node.func)
        
        args_obj = None
        if node.args:
            (self.code
                .invoke('PyTuple_New',len(node.args))
                .check_err())
            
            args_obj = ScratchValue(self.code,R_RET)
            
            for i,a in enumerate(node.args):
                aobj = self.visit(a).to_addr_movable_val(self.code)
                
                args_obj.reload_reg(self.code)
                
                self.code.mov(aobj,tuple_item(args_obj,i))
                aobj.steal(self.code)
        
        if node.starargs:
            s_args_obj = self.visit(node.starargs)
            
            args_p = 0
            if args_obj:
                args_p = args_obj(self.code)
                args_obj.steal(self.code)
            
            (self.code
                .invoke('append_tuple_for_call',fobj,args_p,s_args_obj)
                .check_err())
            
            args_obj = ScratchValue(self.code,R_RET)
            
            s_args_obj.discard(self.code)
        
        args_kwds = None
        if node.keywords:
            (self.code
                .invoke('_PyDict_NewPresized',len(node.keywords))
                .check_err())
            
            args_kwds = ScratchValue(self.code,R_RET)
            
            for kwds in node.keywords:
                obj = self.visit(kwds.value)
                
                (self.code
                    .invoke('PyDict_SetItem',args_kwds,self.get_name(kwds.arg),obj)
                    .check_err(True))
                
                obj.discard(self.code)
        
        if node.kwargs:
            s_kwds = self.visit(node.kwargs)
            
            args_p = 0
            if args_kwds:
                args_p = args_kwds(self.code)
                args_kwds.steal(self.code)
            
            (self.code
                .invoke('append_dict_for_call',fobj,args_p,s_kwds)
                .check_err())
            
            args_kwds = ScratchValue(self.code,R_RET)
            
            s_kwds.discard(self.code)
        
        (self.code
            .invoke('PyObject_Call',
                fobj,
                args_obj or self.get_const(()),
                args_kwds or 0)
            .check_err())
        
        r = ScratchValue(self.code,R_RET)
        
        if args_kwds: args_kwds.discard(self.code)
        if args_obj: args_obj.discard(self.code)
        fobj.discard(self.code)
        
        return r
    
    def visit_Num(self,node):
        return self.get_const(node.n)
    
    def visit_Str(self,node):
        return self.get_const(node.s)
    
    def visit_Bytes(self,node):
        return self.get_const(node.s)
    
    def visit_NameConstant(node):
        raise NotImplementedError()
    def visit_Ellipsis(node):
        return self.get_const(...)
    
    def visit_Attribute(self,node):
        check_context_load(node)
        
        obj = self.visit(node.value)
        
        (self.code
            .invoke('PyObject_GetAttr',obj,self.get_name(node.attr))
            .check_err())
        
        r = ScratchValue(self.code,R_RET)
        
        obj.discard(self.code)
        
        return r
    
    def visit_Subscript(self,node):
        check_context_load(node)
        
        value = self.visit(node.value)
        slice = self._visit_slice(node.slice)
        
        (self.code
            .invoke('PyObject_GetItem',value,slice)
            .check_err())
        
        r = ScratchValue(self.code,R_RET)
        
        value.discard(self.code)
        slice.discard(self.code)
        
        return r
    
    def visit_Starred(self,node):
        raise NotImplementedError()
    def visit_Name(self,node):
        check_context_load(node)
        
        s = self.stable.lookup(node.id)
        
        if self.is_local(s) and self.stable.is_optimized():
            # TODO: this code checks to see if the local has a value. This
            # check could be eliminated in most cases if we were to track
            # assignments and deletions (it would still be required if whether
            # the local is assigned to, depended on a branch, and inside
            # exception handlers).
            
            r = self.code.unused_reg()
            
            (self.code
                .mov(self.local_addrs[node.id],r)
                .if_(not_(r))
                    .invoke('format_exc_check_arg',
                        'PyExc_UnboundLocalError',
                        'UNBOUNDLOCAL_ERROR_MSG',
                        self.get_name(node.id))
                    .exc_cleanup()
                .endif())
            
            return reg_or_scratch_value(self.code,r)

        local = self.is_local(s)
        (self.code
            .mov(self.get_name(node.id),R_PRES1)
            .mov(self.code.special_addrs['frame'],R_PRES3)
            .inner_call(self.util_funcs.local_name if local else self.util_funcs.global_name)
            .check_err())
        return ScratchValue(self.code,R_RET)
    
    def visit_List(self,node):
        check_context_load(node)
        
        item_offset = pyinternals.member_offsets['PyListObject']['ob_item']
        
        (self.code
            .invoke('PyList_New',len(node.elts))
            .check_err())
        
        r = ScratchValue(self.code,R_RET)
        
        with RegCache(self.code) as tmp:
            for i,item in enumerate(node.elts):
                obj = self.visit(item).to_addr_movable_val(self.code)
                
                if not tmp.valid:
                    src = r(self.code)
                    tmp.validate()
                    if not isinstance(src,self.abi.Register):
                        self.code.mov(src,tmp.reg)
                        src = tmp.reg
                    self.code.mov(addr(item_offset,src),tmp.reg)
                

                self.code.mov(obj,addr(self.abi.ptr_size * i,tmp.reg))
                obj.steal(self.code)
        
        return r
    
    def visit_Tuple(self,node):
        check_context_load(node)
        
        (self.code
            .invoke('PyTuple_New',len(node.elts))
            .check_err())
        
        r = ScratchValue(self.code,R_RET)
        
        for i,item in enumerate(node.elts):
            obj = self.visit(item).to_addr_movable_val(self.code)
            
            r.reload_reg(self.code)

            self.code.mov(obj,tuple_item(r,i))
            obj.steal(self.code)
        
        return r


def reserve_debug_temps(s):
    if pyinternals.REF_DEBUG:
        # a place to store r_ret, r_scratch[0] and r_scratch[1] when increasing
        # reference counts (which calls a function when ref_debug is True)
        s.new_stack_value(True,name='temp_ax')
        s.new_stack_value(True,name='temp_cx')
        s.new_stack_value(True,name='temp_dx')
    elif pyinternals.COUNT_ALLOCS:
        # when COUNT_ALLOCS is True and REF_DEBUG is not, an extra space is
        # needed to save a temporary value
        s.new_stack_value(True,name='count_allocs_temp')


def simple_frame(func_name):
    def decorator(func_name,f):
        def inner(abi,end_targets,*extra):
            s = Stitch(abi)
            
            # the return address added by CALL
            s.new_stack_value(True)
            s.annotation(debug.RETURN_ADDRESS)
            
            s.reserve_stack()
            reserve_debug_temps(s)
            
            s = f(s,*extra).release_stack().ret()
            
            r = resolve_jumps(abi.op,destitch(s),end_targets)
            r.name = func_name
            return r
        
        return inner
    
    return partial(decorator,func_name) if isinstance(func_name,str) else decorator(None,func_name)

@simple_frame('local_name')
def local_name_func(s):
    # TODO: have just one copy shared by all compiled modules
    
    # R_PRES1 is expected to have the address of the name to load
    # R_PRES3 is expected to have the address of the frame object

    ret = JumpTarget()
    inc_ret = JumpTarget()
    d_addr = s.fit_addr('PyDict_Type',R_PRES2)
    
    frame = CType('PyFrameObject',R_PRES3)

    return (s
        .mov(frame.f_locals,R_RET)
        .if_(not_(R_RET))
            .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_LOAD_MSG',R_PRES1)
            .jmp(ret)
        .endif()

        .if_(d_addr != type_of(R_RET))
            .invoke('PyObject_GetItem',R_RET,R_PRES1)
            .test(R_RET,R_RET)
            .jnz(ret)

            .invoke('PyErr_ExceptionMatches','PyExc_KeyError')
            .test(R_RET,R_RET)
            .jz(ret)
            .call('PyErr_Clear')
        .else_()
            .invoke('PyDict_GetItem',R_RET,R_PRES1)
            .test(R_RET,R_RET)
            .jnz(inc_ret)
        .endif()

        .invoke('PyDict_GetItem',frame.f_globals,R_PRES1)
        .if_(not_(R_RET))
            .mov(frame.f_builtins,R_RET)
            .if_(d_addr != type_of(R_RET))
                .invoke('PyObject_GetItem',R_RET,R_PRES1)
                .test(R_RET,R_RET)
                .jnz(ret)

                .invoke('PyErr_ExceptionMatches','PyExc_KeyError')
                .test(R_RET,R_RET)
                .jz(ret)
                .call('PyErr_Clear')
                .jmp(ret)
            .else_()
                .invoke('PyDict_GetItem',R_RET,R_PRES1)
                .test(R_RET,R_RET)
                .jnz(inc_ret)
                .invoke('format_exc_check_arg',
                    'PyExc_NameError',
                    'NAME_ERROR_MSG',
                    R_PRES1)
                .jmp(ret)
            .endif()
        .endif()

        (inc_ret)
        .incref(R_RET)
        (ret))

@simple_frame('global_name')
def global_name_func(s):
    # TODO: have just one copy shared by all compiled modules

    # R_PRES1 is expected to have the address of the name to load
    # R_PRES3 is expected to have the address of the frame object

    ret = JumpTarget()
    name_err = JumpTarget()
    
    d_addr = s.fit_addr('PyDict_Type',R_SCRATCH1)
    
    frame = CType('PyFrameObject',R_PRES3)

    return (s
        .mov(frame.f_globals,R_SCRATCH2)
        .mov(frame.f_builtins,R_PRES2)
        .if_(and_(d_addr == type_of(R_SCRATCH2),d_addr == type_of(R_PRES2)))
            .invoke('_PyDict_LoadGlobal',R_SCRATCH2,R_PRES2,R_PRES1)
            .if_(not_(R_RET))
                .call('PyErr_Occurred')
                .test(R_RET,R_RET)
                .jz(name_err)
                .mov(0,R_RET)
                .jmp(ret)
            .endif()
            .incref()
        .else_()
            .invoke('PyObject_GetItem',R_SCRATCH2,R_PRES1)
            .if_(not_(R_RET))
                .invoke('PyObject_GetItem',R_PRES2,R_PRES1)
                .if_(not_(R_RET))
                    .invoke('PyErr_ExceptionMatches','PyExc_KeyError')
                    .if_(not_(R_RET))
                        (name_err)
                        .invoke('format_exc_check_arg',
                            'PyExc_NameError',
                            'GLOBAL_NAME_ERROR_MSG',
                            R_PRES1)
                        .mov(0,R_RET)
                    .endif()
                .endif()
            .endif()
        .endif()
        (ret))


ProtoFunction = namedtuple('ProtoFunction',['name','code'])

class PyFunction:
    def __init__(self,fb_obj,func,names,args,consts):
        self.fb_obj = fb_obj or create_uninitialized(pyinternals.FunctionBody)
        self.func = func
        
        self.names = names
        self.args = args
        self.consts = consts
    
    @property
    def name(self):
        return self.func.name
    
    def build(self,compiled_code):
        args = 0
        vararg = False
        kwargs = 0
        varkw = False
        if self.args:
            args = len(self.args.args)
            vararg = self.args.vararg
            kwargs = len(self.args.kwonlyargs)
            varkw = self.args.kwarg
        
        self.fb_obj.__init__(
            compiled_code,
            self.func.offset,
            self.name,
            self.names,
            args,
            vararg,
            kwargs,
            varkw,
            self.consts)


def func_arg_addr(ec,i):
    r = ec.code.func_arg(i)
    
    if isinstance(r,ec.abi.Register):
        psv = PendingStackValue()
        ec.code.mov(r,psv)
        return psv
    
    return r

def func_arg_bind(ec,arg):
    if isinstance(arg,PendingStackValue):
        sv = ec.code.new_stack_value()
        sv.owned = False
        arg.bind(sv)
        return sv
    
    return BorrowedValue(arg)


def compile_eval(scope_ast,sym_map,abi,util_funcs,entry_points,global_scope,name='<module>',fb_obj=None,args=None):
    #move_throw_flag = (code.co_flags & CO_GENERATOR and len(abi.r_arg) >= 2)
    mov_throw_flag = False
    
    ec = ExprCompiler(Stitch(abi),sym_map[scope_ast._raw_id],sym_map,util_funcs,global_scope,entry_points)
    
    # the stack will have following items:
    #     - return address
    #     - old value of R_PRES1
    #     - old value of R_PRES2
    #     - old value of R_PRES3
    #     - Frame object
    #     - globals
    #     - miscellaneous temp value
    #     - temp_ax (if pyinternals.REF_DEBUG)
    #     - temp_cx (if pyinternals.REF_DEBUG)
    #     - temp_dx (if pyinternals.REF_DEBUG)
    #     - count_allocs_temp (if pyinternals.COUNT_ALLOCS and not
    #       pyinternals.REF_DEBUG)
    
    #state.throw_flag_store = ec.func_arg(1)

    #if move_throw_flag:
        # we use temp_store to use the throw flag and temporarily use another
        # address (the address where the first Python stack value will go) as
        # our temporary store
        #state.throw_flag_store = state.temp_store
        #state.temp_store = state.pstack_addr(-stack_first-1)

    fast_end = JumpTarget()
    
    ec.code.new_stack_value(True) # the return address added by CALL

    (ec.code
        .comment('prologue')
        .annotation(debug.RETURN_ADDRESS)
        .save_reg(R_PRES1)
        .save_reg(R_PRES2)
        .save_reg(R_PRES3)
        .reserve_stack())

    f_obj = CType('PyFrameObject',R_PRES1)
    tstate = CType('PyThreadState',R_SCRATCH1)

    #if move_throw_flag:
    #    ec.code.mov(dword(ec.func_arg(1)),STATE.throw_flag_store)
    
    if args:
        func_self = func_arg_addr(ec,1)
        func_args = func_arg_addr(ec,2)
        func_kwds = func_arg_addr(ec,3)

    argreg = arg_reg(0,R_SCRATCH1)
    (ec.code
        .mov(ec.code.func_arg(0),R_PRES1)
        .call('_EnterRecursiveCall')
        .check_err(True)

        .get_threadstate(R_SCRATCH1)

        .mov(f_obj.f_globals,R_SCRATCH2)

        .push_stack_prolog(R_PRES1,'frame',debug.PushVariable('f'))
        .push_stack_prolog(R_SCRATCH2,'globals')

        .mov(R_PRES1,tstate.frame))
    
    ec.code.new_stack_value(True,name='temp')
    
    reserve_debug_temps(ec.code)

    naddr = ec.code.fit_addr('Py_None',R_RET)
    
    # at the epilogue, GLOBALS is no longer used and we use its space as a
    # temporary store for the return value
    ret_temp_store = ec.code.special_addrs['globals']

    if False: #STATE.code.co_flags & CO_GENERATOR:
        (ec.code
            .mov(f_obj.f_exc_type,R_SCRATCH2)
            .if_(and_(R_SCRATCH2,R_SCRATCH2 != naddr))
                .inner_call(STATE.util_funcs.swap_exc_state)
            .else_()
                .mov(tstate.exc_type,R_RET)
                .mov(tstate.exc_value,R_PRES2)
                .mov(tstate.exc_traceback,R_SCRATCH2)

                .if_(R_RET).incref(R_RET).endif()
                .if_(R_PRES2).incref(R_PRES2).endif()
                .if_(R_SCRATCH2).incref(R_SCRATCH2).endif()

                .mov(f_obj.f_exc_type,R_SCRATCH1)
                .mov(R_RET,f_obj.f_exc_type)
                .mov(f_obj.f_exc_value,R_RET)
                .mov(R_PRES2,f_obj.f_exc_value)
                .mov(f_obj.f_exc_traceback,R_PRES2)
                .mov(R_SCRATCH2,f_obj.f_exc_traceback)

                .if_(R_SCRATCH1).decref(R_SCRATCH1,preserve_reg=R_RET).endif()
                .if_(R_RET).decref(R_RET).endif()
                .if_(R_PRES2).decref(R_PRES2).endif()
            .endif()
            .mov(f_obj.f_lasti,dword(R_SCRATCH1))
            .if_(dword(R_SCRATCH1) != -1))
        
        rip = getattr(abi,'rip',None)
        if rip is not None:
            (ec.code
                .lea(addr(0,rip),R_RET)
                (STATE.yield_start))
        else:
            (ec.code
                .call(abi.Displacement(0))
                (STATE.yield_start)
                .pop(R_RET))
        
        (ec.code
                .add(R_RET,R_SCRATCH1)
            .jmp(R_SCRATCH1)
            .endif()
            .cmpl(0,STATE.throw_flag_store)
            .if_cond(TEST_NE)
                .mov(0,ret_temp_store)
                .jmp(fast_end)
            .endif())

    #if move_throw_flag:
        # use the original address for temp_store again
        #state.temp_store = state.throw_flag_store
    
    counted = []
    
    ec.allocate_locals(args)
    
    if args:
        func_self = func_arg_bind(ec,func_self)
        func_args = func_arg_bind(ec,func_args)
        func_kwds = func_arg_bind(ec,func_kwds)
        
        ec.handle_args(args,func_self,func_args,func_kwds)
        
        func_self.discard(ec.code)
        func_args.discard(ec.code)
        func_kwds.discard(ec.code)
    
    for n in scope_ast.body:
        ec.visit(n)

    tstate = CType('PyThreadState',R_SCRATCH1)
    f_obj = CType('PyFrameObject',R_SCRATCH2)

    # return R_RET
    (ec.code
        .mov(ec.get_const(None),R_RET)
        .incref(R_RET)
        (ec.code.cleanup.dest[0])
        .comment('epilogue')
        .mov(R_RET,ret_temp_store)
        [ec.deallocate_locals]
        (fast_end)
        .call('_LeaveRecursiveCall')
        .mov(ret_temp_store,R_RET)
        .get_threadstate(R_SCRATCH1)
        .mov(ec.code.special_addrs['frame'],R_SCRATCH2)
        .release_stack()
        .restore_reg(R_PRES3)
        .restore_reg(R_PRES2)
        .mov(f_obj.f_back,R_SCRATCH2)
        .restore_reg(R_PRES1)
        .mov(R_SCRATCH2,tstate.frame)
        .ret())

    entry_points.append(PyFunction(fb_obj,ProtoFunction(name,ec.code.code),tuple(ec.names.keys()),args,tuple(ec.consts.keys())))


def compile_raw(code,abi):
    assert len(abi.r_scratch) >= 2 and len(abi.r_pres) >= 2

    ufuncs = UtilityFunctions()
    prepare_exc_handler_tail = JumpTarget()
    unwind_exc_tail = JumpTarget()

    entry_points = []
    mod_ast,sym_map = ast_and_symtable_map(code,'<string>','exec')
    compile_eval(mod_ast,sym_map,abi,ufuncs,entry_points,True)

    functions = []
    end_targets = []

    def add_util_func(target,body,*extra_args):
        target.displacement = 0
        functions.insert(0,body(abi,end_targets,*extra_args))
        end_targets.append(target)

    if ufuncs.swap_exc_state.used:
        add_util_func(ufuncs.swap_exc_state,swap_exc_state_func)

    if ufuncs.unwind_exc.used or ufuncs.unwind_finally.used:
        add_util_func(unwind_exc_tail,unwind_exc_func_tail)

    if ufuncs.unwind_exc.used:
        add_util_func(ufuncs.unwind_exc,unwind_exc_func_head)

    if ufuncs.unwind_finally.used:
        add_util_func(ufuncs.unwind_finally,unwind_finally_func,unwind_exc_tail)

    if ufuncs.prepare_exc_handler.used or ufuncs.reraise_exc_handler.used:
        add_util_func(prepare_exc_handler_tail,prepare_exc_handler_func_tail)

    if ufuncs.prepare_exc_handler.used:
        add_util_func(ufuncs.prepare_exc_handler,prepare_exc_handler_func_head)

    if ufuncs.reraise_exc_handler.used:
        add_util_func(
            ufuncs.reraise_exc_handler,
            reraise_exc_handler_func,
            prepare_exc_handler_tail)

    if ufuncs.global_name.used:
        add_util_func(ufuncs.global_name,global_name_func)

    if ufuncs.local_name.used:
        add_util_func(ufuncs.local_name,local_name_func)


    for pyfunc in reversed(entry_points):
        name,fcode = pyfunc.func
        pyfunc.func = resolve_jumps(abi.op,fcode,end_targets)
        pyfunc.func.name = name
        pyfunc.func.pyfunc = True
        functions.insert(0,pyfunc.func)

    offset = 0
    for func in functions:
        func.offset = offset
        offset += len(func)

    return CompilationUnit(functions),entry_points


def native_abi(*,assembly=False):
    if pyinternals.ARCHITECTURE == "X86":
        return abi.CdeclAbi(assembly=assembly)
    
    if pyinternals.ARCHITECTURE == "X86_64":
        if sys.platform in ('win32','cygwin'):
            return abi.MicrosoftX64Abi(assembly=assembly)

        return abi.SystemVAbi(assembly=assembly)

    raise Exception("native compilation is not supported on this CPU")


def compile(code,globals=None):
    global DUMP_OBJ_FILE
    
    abi = native_abi()
    cu,entry_points = compile_raw(code,abi)

    if debug.GDB_JIT_SUPPORT:
        out = debug.generate(abi,cu,entry_points)
        if DUMP_OBJ_FILE:
            with open('OBJ_DUMP_{}_{}'.format(os.getpid(),int(DUMP_OBJ_FILE)),'wb') as f:
                f.write(out.buff.getbuffer())
            DUMP_OBJ_FILE += 1
    else:
        out = [f.code for f in cu.functions]
    
    compiled = pyinternals.CompiledCode(out)
    
    for e in entry_points: e.build(compiled)
    
    if globals is None:
        globals = {'__builtins__':__builtins__}
    elif '__builtins__' not in globals:
        # it would be slightly more efficient to use the setdefault method of
        # dict objects, but since globals doesn't have to be a dict, we can't
        # assume that method is defined (__contains__ is a safer bet)
        globals['__builtins__'] = __builtins__
    
    return pyinternals.Function(entry_points[0].fb_obj,entry_points[0].name,globals)


def compile_asm(code):
    """Compile code and return the assembly representation.
    
    This is for debugging purposes and is not necessarily usable for
    assembling. In particular, certain instructions in 64-bit mode have
    different names from their 32-bit equivalents, despite having the same
    binary representation, but the assembly returned by this function always
    uses the 32-bit names (since using the 32-bit versions of those
    instructions in 64-bit mode requires using a special prefix, which this
    library doesn't even have support for, there is no ambiguity)."""
    
    abi = native_abi(assembly=True)
    
    parts = []
    for f in compile_raw(code,abi)[0].functions:
        if f.name: parts.append(f.name+':')
        parts.append(f.code.dump())
        
    return '\n'.join(parts)
