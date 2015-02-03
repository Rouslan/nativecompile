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


import ast
import symtable
import os
from collections import OrderedDict

from . import astloader
from . import abi
from .code_gen import *


DUMP_OBJ_FILE = False
NODE_COMMENTS = True


BINOP_FUNCS = {
    ast.Add : 'PyNumber_Add',
    ast.Sub : 'PyNumber_Subtract',
    ast.Mult : 'PyNumber_Multiply',
    ast.Div : 'PyNumber_TrueDivide',
    ast.Mod : 'PyNumber_Remainder',
    ast.LShift : 'PyNumber_Lshift',
    ast.RShift : 'PyNumber_Rshift',
    ast.BitOr : 'PyNumber_Or',
    ast.BitXor : 'PyNumber_Xor',
    ast.BitAnd : 'PyNumber_And',
    ast.FloorDiv : 'PyNumber_FloorDivide'}

BINOP_IFUNCS = {
    ast.Add : 'PyNumber_InPlaceAdd',
    ast.Sub : 'PyNumber_InPlaceSubtract',
    ast.Mult : 'PyNumber_InPlaceMultiply',
    ast.Div : 'PyNumber_InPlaceTrueDivide',
    ast.Mod : 'PyNumber_InPlaceRemainder',
    ast.LShift : 'PyNumber_InPlaceLshift',
    ast.RShift : 'PyNumber_InPlaceRshift',
    ast.BitOr : 'PyNumber_InPlaceOr',
    ast.BitXor : 'PyNumber_InPlaceXor',
    ast.BitAnd : 'PyNumber_InPlaceAnd',
    ast.FloorDiv : 'PyNumber_InPlaceFloorDivide'}


class SyntaxTreeError(Exception):
    """The abstract syntax tree is malformed or has an incorrect value"""

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
        raise SyntaxTreeError('{} node has assign/delete context type but is not in assign/delete statement'.format(node.__class__.__name__))

def maybe_unicode(node):
    """Return False if node will definitely not produce an instance of str

    :type node: ast.AST
    :rtype: bool

    """

    return not isinstance(node,(ast.Num,ast.Bytes,ast.Ellipsis,ast.List,ast.Tuple))

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

        self.free_var_indexes = {}
        if isinstance(s_table,symtable.Function):
            for i,s in enumerate(s_table.get_frees()):
                self.free_var_indexes[s] = i

            self.cell_vars = frozenset(s_table._Function__idents_matching(
                lambda x:((x >> symtable.SCOPE_OFF) & symtable.SCOPE_MASK) == symtable.CELL))
        else:
            self.cell_vars = frozenset()

    @property
    def abi(self):
        return self.code.abi

    def is_local(self,x):
        if not isinstance(x,symtable.Symbol): x = self.stable.lookup(x)
        return (not self.global_scope) and x.is_local()

    def is_cell(self,x):
        if not isinstance(x,symtable.Symbol): x = self.stable.lookup(x)
        return x.get_name() in self.cell_vars or (x.is_free() and x.is_local())

    def swap(self,a,b):
        tmp = self.code.unused_reg()
        (self.code
            .mov(a,tmp)
            .mov(b,a)
            .mov(tmp,b))
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

            zero = 0
            if len(locals) > 1:
                zero = RegValue(self.code,R_PRES1)
                self.code.mov(0,zero)

            addr = None

            # locals is reversed because each stack value has a lower address
            # than the previous value
            for loc in reversed(locals):
                addr = self.code.new_stack_value(True)(self.code)
                if self.is_cell(loc):
                    self.code.invoke('PyCell_New',zero)
                    self.code.mov(R_RET,addr)
                else:
                    self.code.mov(zero,addr)

                self.code.annotation(debug.PushVariable(loc))
                self.local_addrs[loc] = addr

            if zero: zero.discard(self.code)
            self.local_addr_start = addr

    def deallocate_locals(self,s=None):
        assert self.local_addrs is not None

        if s is None: s = self.code

        for name,loc in self.local_addrs.items():
            s.mov(loc,R_RET)
            if self.is_cell(name):
                s.decref(R_RET)
            else:
                s.if_(R_RET).decref(R_RET).endif()

        self.local_addrs = None
        self.local_addr_start = None

    def do_kw_args(self,args,func_self,func_kwds,with_dict,without_dict):
        assert args.kwonlyargs or with_dict

        miss_flag = self.code.new_stack_value(True)

        def mark_miss(s):
            s.add(1,miss_flag)

        hit_count = self.code.new_stack_value(True)

        r_kwds = R_PRES1

        def mark_hit(name):
            def inner(s):
                if args.kwarg:
                    (s
                        .invoke('PyDict_DelItem',r_kwds,name)
                        .check_err(True))
                else:
                    s.add(1,hit_count)

            return inner

        s = (self.code
            .comment('set keyword arguments')
            .mov(0,miss_flag)
            .mov(0,hit_count)
            .mov(func_kwds,r_kwds)
            .if_(r_kwds)
                .comment('with keyword dict'))

        if args.kwarg:
            (s
                .invoke('PyDict_Copy',r_kwds)
                .check_err()
                .mov(R_RET,r_kwds)
                .mov(R_RET,self.get_name(args.kwarg.arg)))

        if with_dict: with_dict(s,mark_miss,mark_hit,r_kwds)

        if args.kwonlyargs:
            kwdefaults = R_PRES2
            (s
                .mov(func_self,kwdefaults)
                .mov(CType('Function',kwdefaults).kwdefaults,kwdefaults))

            for i,a in enumerate(args.kwonlyargs):
                name = self.get_name(a.arg)
                no_def = JumpTarget()

                s = (s
                    .invoke('PyDict_GetItem',r_kwds,name)
                    .if_(R_RET)
                        .mov(R_RET,self.local_addrs[a.arg])
                        .incref(R_RET)
                        [mark_hit(name)]
                    .elif_(kwdefaults)
                        .mov(0,R_SCRATCH1)
                        .mov(addr(i*self.abi.ptr_size,kwdefaults),R_RET)
                        .test(R_RET,R_RET)
                        .jz(no_def)
                        .mov(R_RET,self.local_addrs[a.arg])
                        .incref(R_RET)
                    .else_()(no_def)
                        [mark_miss]
                    .endif())

        if not args.kwarg:
            (s
                .invoke('PyDict_Size',r_kwds)
                .if_(signed(R_RET) > hit_count)
                   .invoke('excess_keyword',func_self,r_kwds)
                   .exc_cleanup()
                .endif())

        s.else_().comment('without keyword dict')

        if args.kwarg:
            (s
                .invoke('PyDict_New')
                .check_err()
                .mov(R_RET,self.get_name(args.kwarg.arg)))

        if without_dict: without_dict(s,mark_miss)

        if args.kwonlyargs:
            kwdefaults = R_PRES2
            s = (s
                .mov(func_self,kwdefaults)
                .mov(CType('Function',kwdefaults).kwdefaults,kwdefaults)
                .if_(kwdefaults))

            no_def = JumpTarget()

            for i,a in enumerate(args.kwonlyargs):
                (s
                    .mov(addr(i*self.abi.ptr_size,kwdefaults),R_RET)
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

            missing_args = R_PRES3
            defaults = R_PRES2
            r_args = R_PRES1

            s = (self.code
                .mov(func_self,defaults)
                .mov(func_args,r_args)
                .mov(len(args.args),missing_args)
                .sub(CType('PyVarObject',r_args).ob_size,missing_args)
                .mov(CType('Function',defaults).defaults,defaults)
                .if_cond(TEST_L)
                    .comment('if there are more arguments than parameters')
                    .neg(missing_args))

            if args.vararg:

                dest_item = tuple_item(R_RET,0)
                dest_item.index = missing_args
                dest_item.scale = self.abi.ptr_size

                src_item = tuple_item(r_args,len(args.args))
                src_item.index = missing_args
                src_item.scale = self.abi.ptr_size

                (s
                        .invoke('PyTuple_New',len(args.args))
                        .check_err()
                        .do()
                            .sub(1,missing_args)
                            .mov(src_item,R_RET)
                            .mov(R_RET,dest_item)
                            .incref(R_RET)
                        .while_(missing_args)
                        .mov(R_RET,self.local_addrs[args.vararg.arg])
                        .jmp(arg_target1)
                    .else_()
                        .mov(self.get_const(()),self.local_addrs[args.vararg.arg]))
            else:
                (s
                    .invoke('too_many_positional',func_self,missing_args,func_kwds)
                    .exc_cleanup())

            s.endif()

            targets = []

            (self.code
                .comment('set positional arguments')
                .jump_table(missing_args,targets,R_SCRATCH2,R_RET))

            for i,a in reversed(list(enumerate(args.args))):
                target = JumpTarget() if targets else arg_target1
                targets.append(target)

                (self.code
                    (target)
                    .mov(tuple_item(r_args,i),R_RET)
                    .mov(R_RET,self.local_addrs[a.arg])
                    .incref(R_RET))

            target = JumpTarget()
            targets.append(target)
            self.code(target)

            def with_dict(s,mark_miss,mark_hit,r_kwds):
                default_item = tuple_item(defaults,0)
                default_item.index = R_RET
                default_item.scale = self.abi.ptr_size

                c_func_self = CType('Function',r_kwds)

                def_len = R_PRES3
                s.mov(CType('PyVarObject',defaults).ob_size,def_len)

                for i,a in enumerate(args.args):
                    name = self.get_name(a.arg)
                    s = (s
                        .invoke('PyDict_GetItem',r_kwds,name)
                        .if_(R_RET)
                            .if_(self.local_addrs[a.arg])
                                .mov(func_self,r_kwds)
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
                        .elif_(not_(self.local_addrs[a.arg]))
                            .if_(signed(len(args.args)-1-i) < def_len)
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
                    .mov(CType('PyVarObject',defaults).ob_size,R_SCRATCH1)
                    .if_(signed(missing_args) > R_SCRATCH1)
                        [mark_miss]
                        .mov(R_SCRATCH1,missing_args)
                    .endif()
                    .mov(len(args.args),R_RET)
                    .sub(missing_args,R_RET)
                    .jump_table(R_RET,targets,R_SCRATCH2,R_RET))

                for i,a in enumerate(args.args):
                    target = JumpTarget()
                    targets.append(target)

                    (s
                        (target)
                        .mov(tuple_item(defaults,i),R_RET)
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
                    .endif()
                    .check_err()
                    .mov(R_RET,self.local_addrs[args.kwarg.arg]))
            else:
                (self.code
                    .mov(func_kwds,R_PRES1)
                    .if_(R_PRES1)
                        .invoke('PyDict_Size',R_PRES1)
                        .if_(R_RET)
                            .invoke('excess_keyword',func_self,R_PRES1)
                        .endif()
                    .endif())

    def assign_expr(self,target,expr,*,check_dest=True):
        if isinstance(target,ast.Name):
            s = self.stable.lookup(target.id)

            if self.is_local(s):
                if self.stable.is_optimized() or self.is_cell(s):
                    # TODO: this code checks to see if the local has a
                    # value and frees it if it does. This check could be
                    # eliminated in most cases if we were to track
                    # assignments and deletions (it would still be required
                    # if whether the local is assigned to depended on a
                    # branch, and inside exception handlers).

                    tmp1 = None
                    tmp3 = None

                    if s.is_free():
                        assert self.is_cell(s)
                        if tmp1 is None: tmp1 = RegValue(self.code)
                        self.code.mov(self.code.special_addrs['free'],tmp1)
                        item = addr(self.abi.ptr_size*self.free_var_indexes[target.id],tmp1)
                    else:
                        item = self.local_addrs[target.id]

                    if check_dest:
                        if tmp1 is None: tmp1 = RegValue(self.code)
                        self.code.mov(item,tmp1)

                    if self.is_cell(s):
                        if check_dest:
                            tmp3 = tmp1
                            tmp1 = RegValue(self.code)
                        else:
                            tmp3 = RegValue(self.code)
                        item = CType('PyCellObject',tmp3).ob_ref

                        if check_dest: self.code.mov(item,tmp1)

                    tmp2 = expr.to_addr_movable_val(self.code)

                    self.code.mov(steal(tmp2),item)

                    if tmp3: tmp3.discard(self.code)

                    if check_dest:
                        r_1 = tmp1(self.code)
                        tmp1.discard(self.code)
                        self.code.if_(r_1).decref(r_1).endif()
                    elif tmp1 is not None:
                        tmp1.discard(self.code)
                else:
                    dict_addr = self.code.fit_addr('PyDict_Type',R_SCRATCH2)
                    locals,setitem = self.code.unused_regs(2)

                    (self.code
                        .push_arg(steal(expr),n=2)
                        .push_arg(self.get_name(target.id),n=1)
                        .mov(self.code.special_addrs['locals'],locals)
                        .if_(not_(locals))
                            .clear_args()
                            .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_STORE_MSG')
                            .exc_cleanup()
                        .endif()
                        .mov('PyObject_SetItem',setitem)
                        .push_arg(locals,n=0)
                        .if_(dict_addr == type_of(R_RET))
                            .mov('PyDict_SetItem',setitem)
                        .endif()
                        .call(setitem)
                        .check_err(True))
            else:
                assert not (s.is_free() or self.is_cell(s))
                (self.code
                    .invoke('PyDict_SetItem',self.code.special_addrs['globals'],self.get_name(target.id),steal(expr))
                    .check_err(True))

        elif isinstance(target,ast.Attribute):
            obj = self.visit(target.value)

            (self.code
                .invoke('PyObject_SetAttr',obj,self.get_name(target.attr),steal(expr))
                .check_err(True))

            obj.discard(self.code)

        elif isinstance(target,ast.Subscript):
            obj = self.visit(target.value)
            slice = self._visit_slice(target.slice)

            (self.code
                .invoke('PyObject_SetItem',obj,slice,steal(expr))
                .check_err(True))

            slice.discard(self.code)
            obj.discard(self.code)

        elif isinstance(target,ast.Starred):
            raise NotImplementedError()
        elif isinstance(target,ast.List):
            raise NotImplementedError()
        elif isinstance(target,ast.Tuple):
            raise NotImplementedError()
        else:
            raise SyntaxTreeError('invalid expression in assignment')

    def add_for_assign(self,a,b,fallback,delete):
        # perform the same optimization that unicode_concatenate() from
        # Python/ceval.c does

        arg1 = self.visit(a)
        arg2 = self.visit(b)

        self.visit_Delete(ast.Delete(delete))

        str_type = self.code.fit_addr('PyUnicode_Type')

        assert isinstance(arg1,StackObj)
        arg1.reload_reg(self.code)
        ttest = type_of(arg1) == str_type
        if not isinstance(b,ast.Str):
            assert isinstance(arg2,StackObj)
            arg2.reload_reg(self.code)
            ttest = and_(ttest,type_of(arg2) == str_type)

        s = (self.code
            .if_(ttest))

        addr_arg1 = arg1.to_addr(s)
        tmp = s.arg_reg(R_RET,0)
        s.free_regs(tmp)

        (s
                .lea(addr_arg1,tmp)
                .invoke('PyUnicode_Append',tmp,arg2)
                .mov(steal(addr_arg1),R_RET)
            .else_()
                .invoke(fallback,arg1,arg2))

        tmp = StackObj(s,R_RET)
        arg1.discard(s)

        (s
                .mov(steal(tmp),R_RET)
            .endif()
            .check_err())

        r = StackObj(self.code,R_RET)

        arg2.discard(self.code)

        return r

    def get_const(self,val):
        c = self.consts.get(val)
        if c is None:
            c = address_of(val)
            self.consts[val] = c
        return ConstObj(c)

    def get_name(self,n):
        c = self.names.get(n)
        if c is None:
            c = address_of(n)
            self.names[n] = c
        return ConstObj(c)


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
            if isinstance(node,ast.Name):
                r += ' ({})'.format(node.id)
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

            r = StackObj(self.code,R_RET)

            if start: start.discard(self.code)
            if end: end.discard(self.code)
            if step: step.discard(self.code)

            return r

        if isinstance(slice,ast.ExtSlice):
            raise NotImplementedError()

        raise SyntaxTreeError('invalid index type in subscript')

    def visit__ParsedValue(self,node):
        return node.value

    def generic_visit(self,node):
        # all types need their own handler
        raise NotImplementedError()

    def visit_Module(self,node):
        assert False

    def visit_FunctionDef(self,node):
        body = create_uninitialized(pyinternals.FunctionBody)

        func = compile_eval(node,self.sym_map,self.abi,self.util_funcs,self.entry_points,False,node.name,body,node.args)

        doc = 0

        if sys.flags.optimize < 2:
            d_str = ast.get_docstring(node)
            if d_str is not None:
                doc = self.get_const(d_str)

        defaults = 0
        if node.args.defaults:
            defaults = self.visit_Tuple(ast.Tuple(node.args.defaults,ast.Load()))

        kwdefaults = 0
        if node.args.kw_defaults:
            (self.code
                .invoke('PyMem_Malloc',len(node.args.kwonlyargs)*self.abi.ptr_size)
                .check_err())

            kwdefaults = ObjArrayValue(self.code,R_RET)

            assert node.args.kwonlyargs and len(node.args.kwonlyargs) == len(node.args.kw_defaults)

            for i,a in enumerate(node.args.kw_defaults):
                if a is not None:
                    val = self.visit(a)
                    kwdefaults.reload_reg(self.code)
                    self.code.mov(steal(val),addr(i*self.abi.ptr_size,kwdefaults))
                else:
                    self.code.mov(0,addr(i*self.abi.ptr_size,kwdefaults))

        free_items = 0
        if func.free_names:
            (self.code
                .invoke('PyMem_Malloc',len(func.free_names)*self.abi.ptr_size)
                .check_err())

            free_items = ObjArrayValue(self.code,R_RET)

            # cells get moved directly and don't need any special handling
            with TempValue(self.code,RegCache) as free_tup:
                for i,n in enumerate(func.free_names):
                    s = self.stable.lookup(n)
                    if s.is_free():
                        if not free_tup.valid:
                            free_tup.validate()
                            self.code.mov(self.code.special_addrs['free'],free_tup)

                        src = tuple_item(free_tup,self.free_var_indexes[n])
                    else:
                        src = self.local_addrs[n]

                    tmp = self.code.unused_reg()
                    (self.code
                        .mov(src,tmp)
                        .mov(tmp,addr(i*self.abi.ptr_size,free_items)))
                    if self.is_cell(s):
                        self.code.incref(tmp)
                    else:
                        self.code.if_(tmp).incref(tmp).endif()

        annots = 0
        a_keys = []
        a_vals = []

        def add_annot(k,v):
            a_keys.append(ast.Str(k))
            a_vals.append(v)

        def add_annot_arg(a):
            if a.annotation:
                add_annot(a.arg,a.annotation)

        if node.returns: add_annot('return',node.returns)
        for a in node.args.args: add_annot_arg(a)
        if node.args.vararg: add_annot_arg(node.args.vararg)
        for a in node.args.kwonlyargs: add_annot_arg(a)
        if node.args.kwarg: add_annot_arg(node.args.kwarg)

        if a_keys:
            annots = self.visit_Dict(ast.Dict(a_keys,a_vals))

        (self.code
            .invoke('new_function',
                self.get_const(body),
                self.get_name(node.name),
                self.code.special_addrs['globals'],
                doc,
                defaults,
                kwdefaults and steal(kwdefaults),
                free_items and steal(free_items),
                annots)
            .check_err())

        fobj = StackObj(self.code,R_RET)

        if defaults: defaults.discard(self.code)
        if annots: annots.discard(self.code)

        # the decorator application and name assignment can be broken down into
        # other AST nodes

        alt_node = _ParsedValue(fobj)
        for d in reversed(node.decorator_list):
            alt_node = ast.Call(d,[alt_node],[],None,None,lineno=d.lineno,col_offset=d.col_offset)

        self.visit(ast.Assign([ast.Name(node.name,ast.Store())],alt_node,lineno=node.lineno,col_offset=node.col_offset))

        fobj.discard(self.code)

    def visit_Return(self,node):
        if node.value:
            val = self.visit(node.value)
        else:
            val = self.get_const(None)

        for ld in self.code.state.stack:
            if ld is not None and ld[1]:
                ld[0].discard(self.code)

        self.code.mov(steal(val),R_RET)
        self.code.jmp(self.code.cleanup.dest[0])

    def visit_Delete(self,node):
        for target in node.targets:
            if isinstance(target,ast.Name):
                s = self.stable.lookup(target.id)

                if self.is_local(s):
                    if self.stable.is_optimized() or self.is_cell(s):
                        # TODO: this code checks to see if the local has a
                        # value. This check could be eliminated in most cases
                        # if we were to track assignments and deletions (it
                        # would still be required if whether the local is
                        # assigned to depended on a branch, and inside
                        # exception handlers).

                        tmp1 = RegValue(self.code)
                        tmp2 = None

                        if s.is_free():
                            assert self.is_cell(s)
                            self.code.mov(self.code.special_addrs['free'],tmp1)
                            item = addr(self.abi.ptr_size*self.free_var_indexes[target.id],tmp1)
                        else:
                            item = self.local_addrs[target.id]

                        self.code.mov(item,tmp1)

                        if self.is_cell(s):
                            tmp2 = tmp1
                            tmp1 = RegValue(self.code)

                            item = CType('PyCellObject',tmp2).ob_ref
                            self.code.mov(item,tmp1)

                        self.code.mov(0,item)

                        if tmp2: tmp2.discard(self.code)

                        r_1 = tmp1(self.code)
                        tmp1.discard(self.code)
                        self.code.invalidate_scratch()
                        self.code.if_(r_1).decref(r_1).endif()
                    else:
                        dict_addr = self.code.fit_addr('PyDict_Type',R_SCRATCH2)
                        locals,delitem = self.code.unused_regs(2)

                        (self.code
                            .push_arg(self.get_name(target.id),n=1)
                            .mov(self.code.special_addrs['locals'],locals)
                            .if_(not_(locals))
                                .clear_args()
                                .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_DELETE_MSG')
                                .exc_cleanup()
                            .endif()
                            .mov('PyObject_DelItem',delitem)
                            .push_arg(locals,n=0)
                            .if_(dict_addr == type_of(R_RET))
                                .mov('PyDict_DelItem',delitem)
                            .endif()
                            .call(delitem)
                            .check_err(True))
                else:
                    assert not (s.is_free() or self.is_cell(s))
                    (self.code
                        .invoke('PyDict_DelItem',self.code.special_addrs['globals'],self.get_name(target.id))
                        .check_err(True))

            elif isinstance(target,ast.Attribute):
                obj = self.visit(target.value)

                (self.code
                    .invoke('PyObject_DelAttr',obj,self.get_name(target.attr))
                    .check_err(True))

                obj.discard(self.code)

            elif isinstance(target,ast.Subscript):
                obj = self.visit(target.value)
                slice = self._visit_slice(target.slice)

                (self.code
                    .invoke('PyObject_DelItem',obj,slice)
                    .check_err(True))

                slice.discard(self.code)
                obj.discard(self.code)

            elif isinstance(target,ast.Starred):
                raise NotImplementedError()
            elif isinstance(target,ast.List):
                raise NotImplementedError()
            elif isinstance(target,ast.Tuple):
                raise NotImplementedError()
            else:
                raise SyntaxTreeError('invalid expression in assignment')

    def visit_Assign(self,node):
        if len(node.targets) < 1:
            raise SyntaxTreeError('instance of ast.Assign must have at least one target')

        check_dest = True

        if (isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.Add)
                and isinstance(node.value.left,ast.Name)
                and any(isinstance(t,ast.Name) and node.value.left.id == t.id for t in node.targets)
                and maybe_unicode(node.value.right)):
            check_dest = False

            expr = self.add_for_assign(
                node.value.left,
                node.value.right,
                'PyNumber_Add',
                [t for t in node.targets if isinstance(t,ast.Name)])
        else:
            expr = self.visit(node.value)

        for t in node.targets[:-1]:
            self.assign_expr(t,borrow(expr))

        self.assign_expr(node.targets[-1],expr,check_dest=check_dest)

    def visit_AugAssign(self,node):
        check_dest = True

        op = type(node.op)

        if (op is ast.Add
                and isinstance(node.target,ast.Name)
                and maybe_unicode(node.value)):
            check_dest = False

            expr = self.add_for_assign(
                node.target,
                node.value,
                'PyNumber_InPlaceAdd',
                [node.target])
        else:
            arg1 = self.visit(node.target)
            arg2 = self.visit(node.value)

            if op is ast.Pow:
                (self.code
                    .invoke('PyNumber_InPlacePower',arg2,arg1,'Py_None')
                    .check_err())
            else:
                (self.code
                    .invoke(BINOP_IFUNCS[op],arg1,arg2)
                    .check_err())

            expr = StackObj(self.code,R_RET)

            arg1.discard(self.code)
            arg2.discard(self.code)

        self.assign_expr(node.target,expr,check_dest=check_dest)

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

    def visit_Nonlocal(self,node):
        # this is handled by the symbol table, so we don't have to do anything
        # here
        pass

    def visit_Expr(self,node):
        self.visit(node.value).discard(self.code)

    def visit_BoolOp(self,node):
        raise NotImplementedError()
    def visit_BinOp(self,node):
        # TODO: find out if constant folding is done automatically by Python.
        # If it is, replace this with an assertion. If it isn't, implement a
        # more comprehensive folding system.
        if isinstance(node.op,ast.Add):
            for t,attr in ((ast.Num,'n'),(ast.Str,'s'),(ast.Bytes,'s')):
                if isinstance(node.left,t) and isinstance(node.right,t):
                    return self.visit(t(getattr(node.left,attr) + getattr(node.right,attr)))

        arg1 = self.visit(node.left)
        arg2 = self.visit(node.right)

        op = type(node.op)

        if op is ast.Add and maybe_unicode(node.left) and maybe_unicode(node.right):
            str_type = self.code.fit_addr('PyUnicode_Type')

            if isinstance(node.left,ast.Str):
                assert isinstance(arg2,StackObj)
                arg2.reload_reg(self.code)
                ttest = type_of(arg2) == str_type
            else:
                assert isinstance(arg1,StackObj)
                arg1.reload_reg(self.code)
                ttest = type_of(arg1) == str_type
                if not isinstance(node.right,ast.Str):
                    assert isinstance(arg2,StackObj)
                    arg2.reload_reg(self.code)
                    ttest = and_(ttest,type_of(arg2) == str_type)

            f = self.code.unused_reg()
            (self.code
                .mov('PyNumber_Add',f)
                .if_(ttest)
                    .mov('PyUnicode_Concat',f)
                .endif()
                .invoke(f,arg1,arg2)
                .check_err())
        elif op is ast.Mod:
            if isinstance(arg1(self.code),self.abi.Register):
                r_arg1 = borrow(arg1)
            else:
                r_arg1 = RegCache(self.code,True)
                self.code.mov(arg1,r_arg1)

            func,r_uaddr = self.code.unused_regs(2)

            uaddr = self.code.fit_addr('PyUnicode_Type',r_uaddr)

            (self.code
                .mov('PyUnicode_Format',func)
                .if_(steal(uaddr) != type_of(r_arg1))
                    .mov('PyNumber_Remainder',func)
                .endif()
                .invoke(steal(func),steal(r_arg1),arg2)
                .check_err())
        elif op is ast.Pow:
            (self.code
                .invoke('PyNumber_Power',arg2,arg1,'Py_None')
                .check_err())
        else:
            (self.code
                .invoke(BINOP_FUNCS[op],arg1,arg2)
                .check_err())

        r = StackObj(self.code,R_RET)

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
                raise SyntaxTreeError('unrecognized unary operation type encountered')

            self.code.invoke(func,arg).check_err()

        r = StackObj(self.code,R_RET)
        arg.discard(self.code)
        return r


    def visit_Lambda(self,node):
        raise NotImplementedError()
    def visit_IfExp(self,node):
        raise NotImplementedError()
    def visit_Dict(self,node):
        (self.code
            .invoke('_PyDict_NewPresized',len(node.keys))
            .check_err())

        r = StackObj(self.code,R_RET)

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

        args_obj = 0
        if node.args:
            (self.code
                .invoke('PyTuple_New',len(node.args))
                .check_err())

            args_obj = StackObj(self.code,R_RET)

            for i,a in enumerate(node.args):
                aobj = self.visit(a).to_addr_movable_val(self.code)

                args_obj.reload_reg(self.code)

                self.code.mov(steal(aobj),tuple_item(args_obj,i))

        if node.starargs:
            s_args_obj = self.visit(node.starargs)

            if args_obj:
                args_obj = steal(args_obj)

            (self.code
                .invoke('append_tuple_for_call',fobj,args_obj,s_args_obj)
                .check_err())

            args_obj = StackObj(self.code,R_RET)

            s_args_obj.discard(self.code)

        args_kwds = 0
        if node.keywords:
            (self.code
                .invoke('_PyDict_NewPresized',len(node.keywords))
                .check_err())

            args_kwds = StackObj(self.code,R_RET)

            for kwds in node.keywords:
                obj = self.visit(kwds.value)

                (self.code
                    .invoke('PyDict_SetItem',args_kwds,self.get_name(kwds.arg),obj)
                    .check_err(True))

                obj.discard(self.code)

        if node.kwargs:
            s_kwds = self.visit(node.kwargs)

            if args_kwds:
                args_kwds = steal(args_kwds)

            (self.code
                .invoke('append_dict_for_call',fobj,args_kwds,s_kwds)
                .check_err())

            args_kwds = StackObj(self.code,R_RET)

            s_kwds.discard(self.code)

        (self.code
            .invoke('PyObject_Call',
                fobj,
                args_obj or self.get_const(()),
                args_kwds or 0)
            .check_err())

        r = StackObj(self.code,R_RET)

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

    def visit_NameConstant(self,node):
        raise NotImplementedError()
    def visit_Ellipsis(self,node):
        return self.get_const(...)

    def visit_Attribute(self,node):
        #check_context_load(node)

        obj = self.visit(node.value)

        (self.code
            .invoke('PyObject_GetAttr',obj,self.get_name(node.attr))
            .check_err())

        r = StackObj(self.code,R_RET)

        obj.discard(self.code)

        return r

    def visit_Subscript(self,node):
        #check_context_load(node)

        value = self.visit(node.value)
        slice = self._visit_slice(node.slice)

        (self.code
            .invoke('PyObject_GetItem',value,slice)
            .check_err())

        r = StackObj(self.code,R_RET)

        value.discard(self.code)
        slice.discard(self.code)

        return r

    def visit_Starred(self,node):
        raise NotImplementedError()
    def visit_Name(self,node):
        #check_context_load(node)

        s = self.stable.lookup(node.id)

        if s.is_free() or (self.is_local(s) and self.stable.is_optimized()) or self.is_cell(s):
            r = self.code.unused_reg()

            if s.is_free():
                (self.code
                    .mov(self.code.special_addrs['free'],r)
                    .mov(addr(self.free_var_indexes[node.id]*self.abi.ptr_size,r),r))
            else:
                self.code.mov(self.local_addrs[node.id],r)

            if self.is_cell(s):
                self.code.mov(CType('PyCellObject',r).ob_ref,r)

            (self.code
                .if_(not_(r))
                    .invoke('format_exc_check_arg',
                        'PyExc_UnboundLocalError',
                        'UNBOUNDLOCAL_ERROR_MSG',
                        self.get_name(node.id))
                    .exc_cleanup()
                .endif())

            return StackObj(self.code,r,owned=False)

        self.code.free_regs(R_PRES1,R_PRES3)
        (self.code
            .mov(self.get_name(node.id),R_PRES1)
            .mov(self.code.special_addrs['frame'],R_PRES3)
            .inner_call(self.util_funcs.local_name if self.is_local(s) else self.util_funcs.global_name)
            .check_err())
        return StackObj(self.code,R_RET)

    def visit_List(self,node):
        #check_context_load(node)

        item_offset = pyinternals.member_offsets['PyListObject']['ob_item']

        (self.code
            .invoke('PyList_New',len(node.elts))
            .check_err())

        r = StackObj(self.code,R_RET)

        with TempValue(self.code,RegCache) as tmp:
            for i,item in enumerate(node.elts):
                obj = self.visit(item).to_addr_movable_val(self.code)

                if not tmp.valid:
                    src = r(self.code)
                    tmp.validate()
                    if not isinstance(src,self.abi.Register):
                        self.code.mov(src,tmp.reg)
                        src = tmp.reg
                    self.code.mov(addr(item_offset,src),tmp.reg)


                self.code.mov(obj.steal(self.code),addr(self.abi.ptr_size * i,tmp.reg))

        return r

    def visit_Tuple(self,node):
        #check_context_load(node)

        (self.code
            .invoke('PyTuple_New',len(node.elts))
            .check_err())

        r = StackObj(self.code,R_RET)

        for i,item in enumerate(node.elts):
            obj = self.visit(item).to_addr_movable_val(self.code)

            r.reload_reg(self.code)

            self.code.mov(obj.steal(self.code),tuple_item(r,i))

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

    frame = CType('PyFrameObject',R_PRES3)

    with TempValue(s,s.fit_addr('PyDict_Type',R_PRES2)) as d_addr:
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

    frame = CType('PyFrameObject',R_PRES3)

    with TempValue(s,s.fit_addr('PyDict_Type',R_SCRATCH1)) as d_addr:
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
    def __init__(self,fb_obj,func,names,args,free_names,cells,consts):
        self.fb_obj = fb_obj or create_uninitialized(pyinternals.FunctionBody)
        self.func = func

        self.names = names
        self.args = args
        self.free_names = free_names
        self.cells = cells
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
            self.free_names,
            self.cells,
            self.consts)


def func_arg_addr(ec,i):
    r = ec.code.func_arg(i)

    if isinstance(r,ec.abi.Register):
        psv = PendingStackObj()
        ec.code.mov(r,psv)
        return psv

    return r

def func_arg_bind(ec,arg):
    if isinstance(arg,PendingStackObj):
        sv = ec.code.new_stack_value(True)
        arg.bind(sv)
        return sv

    return BorrowedValue(arg)


def compile_eval(scope_ast,sym_map,abi,util_funcs,entry_points,global_scope,name='<module>',fb_obj=None,args=None):
    """Compile scope_ast into an intermediate representation

    :type scope_ast: ast.Module | ast.FunctionDef | ast.ClassDef
    :type sym_map: dict[int,symtable.SymbolTable]
    :type abi: abi.Abi
    :type util_funcs: UtilityFunctions
    :type entry_points: list
    :type global_scope: bool
    :type name: str
    :type fb_obj: pyinternals.FunctionBody | None
    :type args: ast.arguments | None
    :rtype: PyFunction

    """
    #move_throw_flag = (code.co_flags & CO_GENERATOR and len(abi.r_arg) >= 2)
    mov_throw_flag = False

    ec = ExprCompiler(Stitch(abi),sym_map[scope_ast._raw_id],sym_map,util_funcs,global_scope,entry_points)

    # the stack will have following items:
    #     - return address
    #     - old value of R_PRES1
    #     - old value of R_PRES2
    #     - old value of R_PRES3
    #     - Frame object
    #     - globals (if ec.stable.get_globals())
    #     - free (if ec.stable.get_free())
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
    r_funcself = CType('Function',R_PRES2)

    #if move_throw_flag:
    #    ec.code.mov(dword(ec.func_arg(1)),STATE.throw_flag_store)

    if args:
        func_self = func_arg_addr(ec,1)
        func_args = func_arg_addr(ec,2)
        func_kwds = func_arg_addr(ec,3)

    ec.code.mov(ec.code.func_arg(0),R_PRES1)
    has_free = isinstance(ec.stable,symtable.Function) and ec.stable.get_frees()
    if has_free:
        ec.code.mov(ec.code.func_arg(1),R_PRES2)

    (ec.code
        .call('_EnterRecursiveCall')
        .check_err(True)

        .get_threadstate(R_SCRATCH1)

        .push_stack_prolog(R_PRES1,'frame',debug.PushVariable('__f'))
        .mov(R_PRES1,tstate.frame))

    #if ec.stable.get_globals():
    if True:
        (ec.code
            .mov(f_obj.f_globals,R_RET)
            .push_stack_prolog(R_RET,'globals'))

    if has_free:
        (ec.code
            .mov(r_funcself.closure,R_RET)
            .push_stack_prolog(R_RET,'free'))

    ec.code.new_stack_value(True,name='temp')

    reserve_debug_temps(ec.code)

    naddr = steal(ec.code.fit_addr('Py_None',R_RET))

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

    if not ec.code.last_op_is_uncond_jmp:
        (ec.code
            .mov(ec.get_const(None),R_RET)
            .incref(R_RET))

    tstate = CType('PyThreadState',R_SCRATCH1)
    f_obj = CType('PyFrameObject',R_SCRATCH2)

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
        .mov(ec.code.special_addrs['frame'],R_SCRATCH2)
        .release_stack()
        .restore_reg(R_PRES3)
        .restore_reg(R_PRES2)
        .mov(f_obj.f_back,R_SCRATCH2)
        .restore_reg(R_PRES1)
        .mov(R_SCRATCH2,tstate.frame)
        .ret())

    r = PyFunction(
        fb_obj,
        ProtoFunction(name,ec.code.code),
        tuple(ec.names.keys()),
        args,
        tuple(ec.free_var_indexes.keys()),
        0,
        tuple(ec.consts.keys()))

    entry_points.append(r)
    return r


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


    for pyfunc in entry_points:
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
    """Compile "code" and return a function taking no arguments.

    :type code: str
    :rtype: pyinternals.Function

    """
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

    return pyinternals.Function(entry_points[-1].fb_obj,entry_points[-1].name,globals)


def compile_asm(code):
    """Compile "code" and return the assembly representation.

    This is for debugging purposes and is not necessarily usable for
    assembling. In particular, certain instructions in 64-bit mode have
    different names from their 32-bit equivalents, despite having the same
    binary representation, but the assembly returned by this function always
    uses the 32-bit names (since using the 32-bit versions of those
    instructions in 64-bit mode requires using a special prefix, which this
    library doesn't even have support for, there is no ambiguity).

    :type code: str
    :rtype: str

    """

    abi = native_abi(assembly=True)

    parts = []
    for f in compile_raw(code,abi)[0].functions:
        if f.name: parts.append(f.name+':')
        parts.append(f.code.dump())

    return '\n'.join(parts)
