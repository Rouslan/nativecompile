
import copy
import ast
import symtable
import os
import io
import sys
from functools import partial
from collections import namedtuple

from . import pyinternals
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


class ExprCompiler(ast.NodeVisitor):
    def __init__(self,code,s_table=None,util_funcs=None,global_scope=False):
        self.code = code
        self.stable = s_table
        self.util_funcs = util_funcs
        self.global_scope = global_scope
        self.consts = {}
        self.names = {}
        self.local_addrs = None
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
    
    def allocate_locals(self):
        assert self.local_addrs is None and self.stable is not None
        
        self.local_addrs = {}
        if self.stable.is_optimized() and isinstance(self.stable,symtable.Function) and self.stable.get_locals():
            tmp = self.unused_reg(self)
            self.code.mov(0,tmp)
            for loc in self.stable.get_locals():
                addr = self.code.new_stack_value(True)(self.code)
                self.code.mov(tmp,addr)
                # TODO: create annotation for local
                self.local_addrs[loc] = addr
    
    def deallocate_locals(self,s=None):
        assert self.local_addrs is not None
        
        if s is None: s = self.code
        
        for loc in self.local_addrs.values():
            (s
                .mov(loc,R_RET)
                .if_(R_RET)
                    .decref(R_RET)
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
    
    def visit_Module(self,node):
        for stmt in node.body: self.visit(stmt)
        (self.code
            .mov(self.get_const(None),R_RET)
            .incref(R_RET))
    
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
                        
                        with self.unused_reg_temp() as tmp:
                            item = self.local_addrs[t.id]
                            self.code.mov(item,tmp.reg)
                            
                            tmp2 = expr(self.code)
                            if not isinstance(tmp2,self.abi.Register):
                                tmp2 = self.unused_reg()
                                self.code.mov(expr,tmp2)
                            
                            if expr.owned:
                                expr.owned = False
                            else:
                                self.code.incref(tmp2)
                            
                            (self.code
                                .mov(tmp2,item)
                                .if_(tmp.reg)
                                    .decref(tmp.reg)
                                .endif())
                    else:
                        dict_addr = self.code.fit_addr('PyDict_Type',R_SCRATCH2)
                        
                        (self.code
                            .push_arg(expr,n=2)
                            .push_arg(self.get_name(t.id),n=1)
                            .mov(LOCALS,R_RET)
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
                        .invoke('PyDict_SetItem',GLOBALS,self.get_name(t.id),expr)
                        .check_err(True))
            
            elif isinstance(t,ast.Attribute):
                obj = self.visit(t.value)
                
                (self.code
                    .invoke('PyObject_SetAttr',obj,self.get_name(t.attr),expr)
                    .check_err(True))
                
                obj.discard(self.code)
            
            elif isinstance(t,ast.Subscript):
                obj = self.visit(t.value)
                
                if isinstance(t.slice,ast.Index):
                    slice = self.visit(t.slice.value)
                elif isinstance(t.slice,ast.Slice):
                    raise NotImplementedError()
                elif isinstance(t.slice,ast.ExtSlice):
                    raise NotImplementedError()
                else:
                    raise NCompileError('invalid index type in subscript')
                
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
        ok = JumpTarget()
        endif = JumpTarget()
        else_ = endif
        if node.orelse: else_ = JumpTarget()
        
        test = self.visit(node.test)
        self.code.invoke('PyObject_IsTrue',test)
        test.discard(self,preserve_reg=R_RET)
        
        if __debug__ or node.orelse:
            old_state = self.state.copy()
        
        (self.code
            .test(R_RET,R_RET)
            .jge(ok)
            .exc_cleanup()
            (ok)
            .jz(else_))
        
        for stmt in node.body:
            self.visit(stmt)
        
        if node.orelse:
            if __debug__:
                new_state = self.state
            self.state = old_state
            
            self.code(else_)
            for stmt in node.orelse:
                self.visit(stmt)
        
        assert (self.state == new_state) if node.orelse else (self.state == old_state)
        
        self.code(endif)
    
    def visit_Expr(self,node):
        self.visit(node.value).discard(self.code)
    
    def visit_BoolOp(self,node):
        raise NotImplementedError()
    def visit_BinOp(self,node):
        arg1 = self.visit(node.left)
        arg2 = self.visit(node.right)
        
        op = type(node.op)
        
        if op is ast.Add:
            assert False
        
        if op is ast.Mod:
            uaddr = r.fit_addr('PyUnicode_Type',R_SCRATCH1)
            func = self.unused_reg()

            (self.code
                .mov(arg1.location,R_RET)
                .mov('PyUnicode_Format',func)
                .if_(uaddr != type_of(R_RET))
                    .mov('PyNumber_Remainder',func)
                .endif()
                .invoke(func,arg2.location,R_RET)
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
        raise NotImplementedError()
    def visit_Subscript(self,node):
        raise NotImplementedError()
    def visit_Starred(self,node):
        raise NotImplementedError()
    def visit_Name(self,node):
        if not isinstance(node.ctx,ast.Load):
            raise NCompileError('name node has assign/delete context type but is not in assign/delete statement')
        
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

        (self.code
            .mov(self.get_name(node.id),R_PRES1)
            .inner_call(self.util_funcs.local_name if self.is_local(s) else self.util_funcs.global_name)
            .check_err())
        return ScratchValue(self.code,R_RET)
    
    def visit_List(self,node):
        if not isinstance(node.ctx,ast.Load):
            raise NCompileError('list node has assign/delete context type but is not in assign/delete statement')
        
        item_offset = pyinternals.member_offsets['PyListObject']['ob_item']
        
        (self.code
            .invoke('PyList_New',len(node.elts))
            [self.check_err()])
        
        r = ScratchValue(self.code,R_RET)
        
        with RegCache(self.code) as tmp:
            for i,item in enumerate(node.elts):
                obj = self.visit(item).to_addr_movable_val(self.code)
                
                if not tmp.valid:
                    src = r(self.code)
                    tmp.validate(self.unused_reg())
                    if not isinstance(src,self.abi.Register):
                        self.mov(src,tmp.reg)
                        src = tmp.reg
                    self.mov(addr(item_offset,src),tmp.reg)
                

                self.mov(obj,addr(self.abi.ptr_size * i,tmp.reg))
                obj.steal(self.code)
        
        return r
    
    def visit_Tuple(self,node):
        if not isinstance(node.ctx,ast.Load):
            raise NCompileError('tuple node has assign/delete context type but is not in assign/delete statement')
        
        item_offset = pyinternals.member_offsets['PyTupleObject']['ob_item']
        
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


def simple_frame(func_name):
    def decorator(func_name,f):
        def inner(abi,end_targets,*extra):
            s = Stitch(abi)
            s.def_stack_offset.base = aligned_size((MAX_ARGS + 1) * abi.ptr_size) // abi.ptr_size
            
            s.new_stack_value(True) # the return address added by CALL

            s.annotation(debug.RETURN_ADDRESS)
            
            r = resolve_jumps(abi.op,destitch(f(s,*extra)),end_targets)
            r.name = func_name
            return r
        
        return inner
    
    return partial(decorator,func_name) if isinstance(func_name,str) else decorator(None,func_name)

@simple_frame('local_name')
def local_name_func(s):
    # TODO: have just one copy shared by all compiled modules

    ret = JumpTarget()
    inc_ret = JumpTarget()
    d_addr = s.fit_addr('PyDict_Type',R_PRES2)

    return (s
        .reserve_stack()
        .mov(LOCALS,R_RET)
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

        .invoke('PyDict_GetItem',GLOBALS,R_PRES1)
        .if_(not_(R_RET))
            .mov(BUILTINS,R_RET)
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
        (ret)
        .release_stack()
        .ret())

@simple_frame('global_name')
def global_name_func(s):
    # TODO: have just one copy shared by all compiled modules

    # R_PRES1 is expected to have the address of the name to load

    ret = JumpTarget()
    name_err = JumpTarget()
    
    d_addr = s.fit_addr('PyDict_Type',R_SCRATCH1)

    return (s
        .reserve_stack()
        .mov(GLOBALS,R_SCRATCH2)
        .mov(BUILTINS,R_PRES2)
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
        (ret)
        .release_stack()
        .ret())


ProtoFunction = namedtuple('ProtoFunction',['name','code'])

class PyFunction:
    def __init__(self,func,names,consts):
        self.func = func
        
        self.names = names
        self.consts = consts
    
    @property
    def name(self):
        return self.func.name

def compile_eval(code,abi,util_funcs,entry_points,global_scope):
    """Generate a function equivalent to PyEval_EvalFrameEx called with f.code"""
    
    # TODO: create and use extension function that creates the SymbolTable
    # object from the AST object (symtable.symtable generates an AST and then
    # discards it)
    mod_ast = ast.parse(code,'<string>')
    mod_sym = symtable.symtable(code,'<string>','exec')
    
    #move_throw_flag = (code.co_flags & CO_GENERATOR and len(abi.r_arg) >= 2)
    mov_throw_flag = False
    
    ec = ExprCompiler(Stitch(abi),mod_sym,util_funcs,global_scope)
    
    # the stack will have following items:
    #     - return address
    #     - old value of %ebp/%rbp
    #     - old value of %ebx/%rbx
    #     - old value of %esi/%rsi
    #     - Frame object
    #     - GLOBALS
    #     - BUILTINS
    #     - LOCALS
    #     - miscellaneous temp value
    #     - TEMP_AX (if pyinternals.REF_DEBUG)
    #     - TEMP_CX (if pyinternals.REF_DEBUG)
    #     - TEMP_DX (if pyinternals.REF_DEBUG)
    #     - COUNT_ALLOCS_TEMP (if pyinternals.COUNT_ALLOCS and not
    #       pyinternals.REF_DEBUG)
    
    #state.throw_flag_store = ec.func_arg(1)

    #if move_throw_flag:
        # we use temp_store to use the throw flag and temporarily use another
        # address (the address where the first Python stack value will go) as
        # our temporary store
        #state.throw_flag_store = state.temp_store
        #state.temp_store = state.pstack_addr(-stack_first-1)

    fast_end = JumpTarget()
    
    # at the epilogue, GLOBALS is no longer used and we use its space as a
    # temporary store for the return value
    ret_temp_store = GLOBALS
    
    ec.code.new_stack_value(True) # the return address added by CALL

    (ec.code
        .comment('prologue')
        .annotation(debug.RETURN_ADDRESS)
        .save_reg(R_BP)
        .mov(R_SP,R_BP)
        .save_reg(R_PRES1)
        .save_reg(R_PRES2)
        .reserve_stack())

    f_obj = CType('PyFrameObject',R_PRES1)
    tstate = CType('PyThreadState',R_SCRATCH1)

    #if move_throw_flag:
    #    ec.code.mov(dword(ec.func_arg(1)),STATE.throw_flag_store)

    argreg = arg_reg(0,R_SCRATCH1)
    (ec.code
        .mov(ec.code.func_arg(0),R_PRES1)
        .call('_EnterRecursiveCall')
        .check_err(True)

        .get_threadstate(R_SCRATCH1)

        .mov(f_obj.f_globals,R_SCRATCH2)
        .mov(f_obj.f_builtins,R_RET)

        .push_stack_prolog(R_PRES1,debug.PushVariable('f'))
        .push_stack_prolog(R_SCRATCH2)
        .push_stack_prolog(R_RET)

        .mov(f_obj.f_locals,R_SCRATCH2)
        .push_stack_prolog(R_SCRATCH2)

        .mov(R_PRES1,tstate.frame))
    
    stack_extra = 1 # 1 for the miscellaneous temp value
    if pyinternals.REF_DEBUG:
        # a place to store r_ret, r_scratch[0] and r_scratch[1] when increasing
        # reference counts (which calls a function when ref_debug is True)
        stack_extra += DEBUG_TEMPS
    elif pyinternals.COUNT_ALLOCS:
        # when COUNT_ALLOCS is True and REF_DEBUG is not, an extra space is
        # needed to save a temporary value
        stack_extra += 1
    
    for _ in range(stack_extra):
        ec.code.new_stack_value(True)

    naddr = ec.code.fit_addr('Py_None',R_RET)

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
    
    ec.allocate_locals()

    #if move_throw_flag:
        # use the original address for temp_store again
        #state.temp_store = state.throw_flag_store
    
    ec.visit(mod_ast)
    
    stack_len = len(ec.code.state.stack)

    tstate = CType('PyThreadState',R_SCRATCH1)
    f_obj = CType('PyFrameObject',R_SCRATCH2)
    
    # at the epilogue, GLOBALS is no longer used and we use its space as a
    # temporary store for the return value
    ret_temp_store = GLOBALS

    # return R_RET
    (ec.code
        (ec.code.cleanup.dest[0])
        .comment('epilogue')
        .mov(R_RET,ret_temp_store)
        [ec.deallocate_locals]
        (fast_end)
        .call('_LeaveRecursiveCall')
        .mov(ret_temp_store,R_RET)
        .get_threadstate(R_SCRATCH1)
        .mov(FRAME,R_SCRATCH2)
        .release_stack()
        .restore_reg(R_PRES2)
        .restore_reg(R_PRES1)
        .mov(f_obj.f_back,R_SCRATCH2)
        .restore_reg(R_BP)
        .mov(R_SCRATCH2,tstate.frame)
        .ret())

    ec.code.def_stack_offset.base = aligned_size(
        (stack_len + max(MAX_ARGS-len(abi.r_arg),0)) * abi.ptr_size + abi.shadow) // abi.ptr_size

    return PyFunction(ProtoFunction('stuff',ec.code.code),tuple(ec.names.keys()),tuple(ec.consts.keys()))


def compile_raw(code,abi):
    assert len(abi.r_scratch) >= 2 and len(abi.r_pres) >= 2

    ufuncs = UtilityFunctions()
    prepare_exc_handler_tail = JumpTarget()
    unwind_exc_tail = JumpTarget()

    entry_points = []

    # if this ever gets ported to C or C++, this will be a prime candidate for
    # parallelization
    #def compile_code_constants(code):
    #    for c in code:
    #        if isinstance(c,types.CodeType) and id(c) not in entry_points:
    #            # reserve the entry
    #            entry_points[id(c)] = None

    #            compile_code_constants(c.co_consts)
    #            entry_points[id(c)] = (
    #                pyinternals.create_compiled_entry_point(c),
    #                ceval(c))

    #compile_code_constants(_code)
    entry_points.append(compile_eval(code,abi,ufuncs,entry_points,True))

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


def compile(code):
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
    
    head = entry_points[0]
    return pyinternals.Function(compiled,head.func.offset,head.name,head.names,head.consts)


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
