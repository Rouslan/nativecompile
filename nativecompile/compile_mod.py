#  Copyright 2017 Rouslan Korneychuk
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


import sys
import ast
import symtable
import os
import importlib
from collections import namedtuple
from itertools import chain
from typing import Callable,Dict,List,Optional,Tuple,Union

from .code_gen import *
from .intermediate import *
from . import astloader
from . import pyinternals
from . import debug
from . import c_types


DUMP_OBJ_FILE = 1 if int(os.environ.get('DUMP_OBJ_FILE','0')) else 0
NODE_COMMENTS = True
DISABLE_DEBUG = False


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

CONST_NODES = ast.Num,ast.Str,ast.Bytes,ast.NameConstant,ast.Ellipsis

# these MUST have the order that the Py_* macros in object.h prescribe
RICH_CMP_NODES = [ast.Lt,ast.LtE,ast.Eq,ast.NotEq,ast.Gt,ast.GtE]

utility_funcs = None


class SyntaxTreeError(Exception):
    """The abstract syntax tree is malformed or has an invalid value"""


def ast_and_symtable_map(code,filename,compile_type):
    ast,tables = astloader.symtable(code,filename,compile_type)

    return ast,{id_ : symtable._newSymbolTable(tab,filename) for id_,tab in tables.items()}


StitchCallback = Callable[[Stitch],None]

class ScopeContext:
    def __init__(self,depth : int) -> None:
        self.depth = depth
        self.extra_callbacks = [] # type: List[StitchCallback]

class LoopContext(ScopeContext):
    def __init__(self,depth : int) -> None:
        super().__init__(depth)
        self.begin = Target()
        self.break_ = Target()

class TryContext(ScopeContext):
    def __init__(self,depth : int) -> None:
        super().__init__(depth)
        self.depth = depth
        self.target = Target()

class ExceptContext(ScopeContext):
    def __init__(self,depth : int) -> None:
        super().__init__(depth)
        self.depth = depth
        self.target = Target()

class FinallyContext(ScopeContext):
    def __init__(self,depth : int) -> None:
        super().__init__(depth)
        self.depth = depth
        self.ftarget = SFinallyTarget()
        self.may_return = False


def maybe_unicode(node : ast.AST) -> bool:
    """Return False iff node will definitely not produce an instance of str"""
    return not isinstance(node,(ast.Num,ast.Bytes,ast.Ellipsis,ast.List,ast.Tuple))

class _ParsedValue(ast.AST):
    """A fake AST node to allow inserting arbitrary values into AST trees.

    Note that since AST nodes normally create new Python objects, this class
    effectively steals a reference. To pass an owned object without losing
    ownership, wrap it in borrow().

    """

    def __init__(self,value):
        self.value = value

class CallTargetAction:
    def __init__(self,ftarget):
        self.ftarget = ftarget
        self.ret_target = Target()

    def __call__(self,s):
        s.enter_finally(self.ftarget,self.ret_target)(self.ret_target)

    def __hash__(self):
        return hash(self.ftarget)

    def __eq__(self,b):
        return isinstance(b,CallTargetAction) and self.ftarget == b.target

if NODE_COMMENTS:
    def visit_extra(f):
        def inner(self,node,*args,**kwds):
            self.code.comment('{}node {} {{','  '*self.visit_depth,self.node_descr(node))
            self.visit_depth += 1
            r = f(self,node,*args,**kwds)
            self.visit_depth -= 1
            self.code.comment('  '*self.visit_depth + '}')
            self.node_var = None
            return r
        return inner
else:
    def visit_extra(f):
        return f

class FinallyBody:
    def __init__(self,ec: 'ExprCompiler',f_context: FinallyContext) -> None:
        self.ec = ec
        self.f_context = f_context
        self.no_return = False

    def __enter__(self):
        if self.f_context.may_return:
            self.ec.check_var_return += 1
        self.ec.code.begin_finally(self.f_context.ftarget)

        return self

    def __exit__(self,exc_type,exc_val,exc_tb):
        if exc_type is None:
            self.ec.code.end_finally(self.no_return)
        if self.f_context.may_return:
            self.ec.check_var_return -= 1

# noinspection PyPep8Naming
class ExprCompiler:
    def __init__(self,cgen,var_frame,s_table=None,s_map=None,u_funcs=None,global_scope=False,entry_points=None):
        self.code = Stitch(cgen)
        self.stable = s_table
        self.sym_map = s_map
        self.u_funcs = u_funcs
        self.global_scope = global_scope
        self.entry_points = entry_points
        self.consts = {}
        self.names = {}
        self.name_overrides = {}
        self.local_addrs = None
        self.local_addr_block = None
        self.visit_depth = 0
        self.target_contexts = []
        self.func_end = Target()
        self.var_frame = var_frame
        self.var_globals = Var('var_globals',data_type=c_types.PyObject_ptr)
        self.var_locals = Var('var_locals',data_type=c_types.PyObject_ptr)
        self.var_free = Var('var_free',data_type=c_types.PyObject_ptr)
        self.var_return = Var('var_return',dbg_symbol='var_return',data_type=c_types.PyObject_ptr)

        # if this is greater than 0, var_return may have a value
        self.check_var_return = 0

        self._del_name_cleanups = {}

        self.free_var_indexes = {}
        if isinstance(s_table,symtable.Function):
            for i,s in enumerate(s_table.get_frees()):
                self.free_var_indexes[s] = i

            # noinspection PyUnresolvedReferences
            self.cell_vars = frozenset(s_table._Function__idents_matching(
                lambda x: ((x >> symtable.SCOPE_OFF) & symtable.SCOPE_MASK) == symtable.CELL))
        else:
            self.cell_vars = frozenset()

    @property
    def abi(self):
        return self.code.cgen.abi

    def is_local(self,x):
        if not isinstance(x,symtable.Symbol): x = self.stable.lookup(x)

        # "not x.is_global()" should be used instead of "x.is_local()" because
        # both can return True for the same symbol
        return not (self.global_scope or x.is_global())

    def is_cell(self,x):
        if not isinstance(x,symtable.Symbol): x = self.stable.lookup(x)
        return x.get_name() in self.cell_vars or (x.is_free() and x.is_local())

    def swap(self,a,b):
        tmp = Var(data_type=a.data_type)
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

            locals_ = [a.arg for a in args.args]
            if args.vararg: locals_.append(args.vararg.arg)
            locals_.extend(a.arg for a in args.kwonlyargs)
            if args.kwarg: locals_.append(args.kwarg.arg)

            the_rest = set(self.stable.get_locals())
            the_rest.difference_update(locals_)
            locals_.extend(the_rest)

            if locals_:
                for n in locals_:
                    self.names[n] = (address_of(n),len(self.names))

                zero = 0
                if len(locals_) > 1:
                    zero = Var(data_type=c_types.PyObject_ptr)
                    self.code.mov(0,zero)

                self.local_addr_block = Block(len(locals_))

                for i,loc in enumerate(locals_):
                    addr = self.local_addr_block[i]
                    if self.is_cell(loc):
                        tmp = Var(data_type=c_types.PyObject_ptr)
                        self.code.call('PyCell_New',zero,store_ret=tmp)
                        self.code.mov(tmp,addr)
                    else:
                        self.code.mov(zero,addr)

                    self.local_addrs[loc] = addr

    def deallocate_locals(self,s=None):
        assert self.local_addrs is not None

        if s is None: s = self.code

        for name,loc in self.local_addrs.items():
            tmp = Var(data_type=c_types.PyObject_ptr)
            s.mov(loc,tmp)
            if self.is_cell(name):
                s.decref(tmp)
            else:
                s.if_(tmp).decref(tmp).endif()

        self.local_addrs = None
        self.local_addr_block = None

    def do_kw_args(self,args,func_self,func_kwds,with_dict,without_dict):
        assert args.kwonlyargs or with_dict

        miss_flag = Var(data_type=c_types.t_int)

        def mark_miss(s):
            s.add(miss_flag,1)

        hit_count = Var(data_type=c_types.t_int)

        r_kwds = Var(data_type=c_types.PyObject_ptr)

        def mark_hit(name):
            def inner(s):
                if args.kwarg:
                    r = Var(data_type=c_types.t_int)
                    (s
                        .call('PyDict_DelItem',r_kwds,name,store_ret=r)
                        (self.check_err(r,True)))
                else:
                    s.add(hit_count,1)

            return inner

        (self.code
            .comment('set keyword arguments')
            .mov(0,miss_flag)
            .mov(0,hit_count)
            .mov(func_kwds,r_kwds)
            .if_(r_kwds)
                .comment('with keyword dict'))

        if args.kwarg:
            tmp = Var(data_type=c_types.PyObject_ptr)
            (self.code
                .call('PyDict_Copy',r_kwds,store_ret=tmp)
                (self.check_err(tmp))
                .mov(tmp,r_kwds)
                .mov(tmp,self.get_name(args.kwarg.arg)))

        if with_dict: with_dict(self.code,mark_miss,mark_hit,r_kwds)

        if args.kwonlyargs:
            kwdefaults = Var(data_type=c_types.PyObject_ptr)
            (self.code
                .mov(func_self,kwdefaults)
                .mov(CType('Function',kwdefaults).kwdefaults,kwdefaults))

            for i,a in enumerate(args.kwonlyargs):
                name = self.get_name(a.arg)
                no_def = Target()
                item = Var(data_type=c_types.PyObject_ptr)
                kwdef = Var(data_type=c_types.PyObject_ptr)

                (self.code
                    .call('PyDict_GetItem',r_kwds,name,store_ret=item)
                    .if_(item)
                        .mov(item,self.local_addrs[a.arg])
                        .incref(item)
                        [mark_hit(name)]
                    .elif_(kwdefaults)
                        .mov(SIndirect(i*SIZE_PTR,kwdefaults),kwdef)
                        .jump_if(not_(kwdef),no_def)
                        .mov(kwdef,self.local_addrs[a.arg])
                        .incref(kwdef)
                    .else_()(no_def)
                        [mark_miss]
                    .endif())

        if not args.kwarg:
            size = Var(data_type=c_types.t_long)
            (self.code
                .call('PyDict_Size',r_kwds,store_ret=size)
                .if_(signed(size) > hit_count)
                   .call('excess_keyword',func_self,r_kwds)
                (self.exc_cleanup())
                .endif())

        self.code.else_().comment('without keyword dict')

        if args.kwarg:
            tmp = Var(data_type=c_types.PyObject_ptr)
            (self.code
                .call('PyDict_New',store_ret=tmp)
                (self.check_err(tmp))
                .mov(tmp,self.get_name(args.kwarg.arg)))

        if without_dict: without_dict(self.code,mark_miss)

        if args.kwonlyargs:
            kwdefaults = Var(data_type=c_types.PyObject_ptr)
            (self.code
                .mov(func_self,kwdefaults)
                .mov(CType('Function',kwdefaults).kwdefaults,kwdefaults)
                .if_(kwdefaults))

            no_def = Target()

            for i,a in enumerate(args.kwonlyargs):
                kwdef = Var(data_type=c_types.PyObject_ptr)
                (self.code
                    .mov(SIndirect(i*SIZE_PTR,kwdefaults),kwdef)
                    .jump_if(not_(kwdef),no_def)
                    .mov(kwdef,self.local_addrs[a.arg])
                    .incref(kwdef))

            (self.code
                .else_()(no_def)
                    (mark_miss)
                .endif())

        r_arg = Var(data_type=c_types.TPtr(c_types.PyObject_ptr))
        (self.code
            .endif()
            .lea(self.local_addr_block,r_arg)
            .if_(miss_flag)
                .call('missing_arguments',func_self,r_arg)
                (self.exc_cleanup())
            .endif())

    def handle_args(self,args,func_self,func_args,func_kwds):
        assert self.local_addrs is not None

        if args.args:
            # a short-cut to the part where positional arguments are moved
            arg_target1 = Target()

            missing_args = Var('missing_args',c_types.t_long)
            defaults = Var('defaults',c_types.PyObject_ptr)
            r_args = Var('r_args',c_types.PyObject_ptr)

            (self.code
                .mov(func_args,r_args)
                .mov(len(args.args),missing_args)
                .sub(missing_args,CType(c_types.PyVarObject,r_args).ob_size)
                .mov(CType('Function',func_self).defaults,defaults)
                .if_(signed(missing_args) < 0)
                    .comment('if there are more arguments than parameters')
                    .neg(missing_args))

            if args.vararg:
                va_tup = Var('va_tup',c_types.PyObject_ptr)
                tmp = Var('tmp',c_types.PyObject_ptr)

                dest_item = tuple_item(va_tup,0)
                dest_item.index = missing_args
                dest_item.scale = self.abi.ptr_size

                src_item = tuple_item(r_args,len(args.args))
                src_item.index = missing_args
                src_item.scale = self.abi.ptr_size

                (self.code
                        .call('PyTuple_New',len(args.args),store_ret=va_tup)
                        (self.check_err(va_tup))
                        .do()
                            .sub(missing_args,1)
                            .mov(src_item,tmp)
                            .mov(tmp,dest_item)
                            .incref(tmp)
                        .while_(missing_args)
                        .mov(va_tup,self.local_addrs[args.vararg.arg])
                        .jmp(arg_target1)
                    .else_()
                        .mov(self.get_const(()),self.local_addrs[args.vararg.arg]))
            else:
                (self.code
                    .call('too_many_positional',func_self,missing_args,func_kwds)
                    (self.exc_cleanup()))

            self.code.endif()

            targets = [arg_target1] + [Target() for _ in range(len(args.args))]

            (self.code
                .comment('set positional arguments')
                .jump_table(missing_args,targets))

            t_itr = iter(targets)

            for i in range(len(args.args)-1,-1,-1):
                a = args.args[i]
                tmp = Var(data_type=c_types.PyObject_ptr)

                (self.code
                    (next(t_itr))
                    .mov(tuple_item(r_args,i),tmp)
                    .mov(tmp,self.local_addrs[a.arg])
                    .incref(tmp))

            self.code(next(t_itr))

            def with_dict(s,mark_miss,mark_hit,r_kwds):
                d_index = Var(data_type=c_types.t_long)
                default_item = tuple_item(defaults,0)
                default_item.index = d_index
                default_item.scale = SIZE_PTR

                c_func_self = CType('Function',r_kwds)

                def_len = Var(data_type=c_types.t_long)
                item = Var(data_type=c_types.PyObject_ptr)
                item_b = Var(data_type=c_types.PyObject_ptr)
                s.mov(CType(c_types.PyVarObject,defaults).ob_size,def_len)

                for i,a in enumerate(args.args):
                    name = self.get_name(a.arg)
                    (s
                        .call('PyDict_GetItem',r_kwds,name,store_ret=item)
                        .if_(item)
                            .if_(self.local_addrs[a.arg])
                                .mov(func_self,r_kwds)
                                .call('PyErr_Format',
                                    'PyExc_TypeError',
                                    'DUPLICATE_VAL_MSG',
                                    c_func_self.name,
                                    name)
                                (self.exc_cleanup())
                            .endif()
                            .mov(item,self.local_addrs[a.arg])
                            .incref(item)
                            (mark_hit(name))
                        .elif_(not_(self.local_addrs[a.arg]))
                            .sub(def_len,len(args.args)-i,d_index)
                            .if_(signed(d_index) >= 0)
                                .mov(default_item,item_b)
                                .mov(item_b,self.local_addrs[a.arg])
                                .incref(item_b)
                            .else_()
                                (mark_miss)
                            .endif()
                        .endif())

            def without_dict(s,mark_miss):
                # just set the defaults.

                def_len = Var('def_len',c_types.t_long)
                first_def = Var('first_def',c_types.t_long)

                targets = [Target() for _ in range(len(args.args)+1)]

                (s
                    .mov(CType(c_types.PyVarObject,defaults).ob_size,def_len)
                    .if_(signed(missing_args) > def_len)
                        (mark_miss)
                        .mov(def_len,missing_args)
                    .endif()
                    .sub(len(args.args),missing_args,first_def)

                    # move back the 'defaults' pointer so that the index of the
                    # argument can be used to get the associated default
                    .lea(SIndirect(-len(args.args) * SIZE_PTR,defaults,index=def_len),defaults)

                    .jump_table(first_def,targets))

                for i,a in enumerate(args.args):
                    tmp = Var('tmp',c_types.PyObject_ptr)

                    (s
                        (targets[i])
                        .mov(tuple_item(defaults,i),tmp)
                        .mov(tmp,self.local_addrs[a.arg])
                        .incref(tmp))

                s(targets[-1])

            self.do_kw_args(args,func_self,func_kwds,with_dict,without_dict)
        else:
            if args.vararg:
                (self.code
                    .mov(func_args,self.local_addrs[args.vararg.arg])
                    .incref(func_args))
            else:
                (self.code
                    .if_(CType(c_types.PyVarObject,func_args).ob_size)
                        .call('too_many_positional',func_self,func_args,func_kwds)
                        (self.exc_cleanup())
                    .endif())

            if args.kwonlyargs:
                self.do_kw_args(args,func_self,func_kwds,None,None)
            elif args.kwarg:
                tmp = Var('tmp')
                (self.code
                    .if_(func_kwds)
                        .call('PyDict_Copy',func_kwds,store_ret=tmp)
                    .else_()
                        .call('PyDict_New',store_ret=tmp)
                    .endif()
                    (self.check_err(tmp))
                    .mov(tmp,self.local_addrs[args.kwarg.arg]))
            else:
                size = Var(data_type=c_types.t_long)
                (self.code
                    .if_(func_kwds)
                        .call('PyDict_Size',func_kwds,store_ret=size)
                        .if_(size)
                            .call('excess_keyword',func_self,func_kwds)
                        .endif()
                    .endif())

    def assign_name_expr(self,name,expr,*,check_dest=True):
        s = self.stable.lookup(name)

        if self.is_local(s):
            if self.stable.is_optimized() or self.is_cell(s):
                # TODO: this code checks to see if the local has a value and
                # frees it if it does. This check could be eliminated in most
                # cases if we were to track assignments and deletions (it would
                # still be required if whether the local is assigned-to
                # depended on a branch, and inside exception handlers).

                tmp1 = None

                if s.is_free():
                    assert self.is_cell(s)
                    if tmp1 is None: tmp1 = Var(data_type=c_types.PyObject_ptr)
                    self.code.mov(self.var_free,tmp1)
                    item = SIndirect(SIZE_PTR*self.free_var_indexes[name],tmp1)
                else:
                    item = self.local_addrs[name]

                if check_dest:
                    if tmp1 is None: tmp1 = Var(data_type=c_types.PyObject_ptr)
                    self.code.mov(item,tmp1)

                if self.is_cell(s):
                    if check_dest:
                        tmp2 = tmp1
                        tmp1 = Var(data_type=c_types.PyObject_ptr)
                    else:
                        tmp2 = Var(data_type=c_types.PyObject_ptr)
                    item = CType('PyCellObject',tmp2).ob_ref

                    if check_dest: self.code.mov(item,tmp1)

                self.code.mov(steal(expr),item)

                if check_dest:
                    assert tmp1 is not None
                    self.code.if_(tmp1).decref(tmp1).endif()
            else:
                setitem = Var('setitem',c_types.TPtr(c_types.TFunc([c_types.PyObject_ptr]*3,c_types.PyObject_ptr)))
                ret = Var('ret',c_types.t_int)

                (self.code
                    .if_(not_(self.var_locals))
                        .call('PyErr_Format','PyExc_SystemError','NO_LOCALS_STORE_MSG')
                        (self.exc_cleanup())
                    .endif()
                    .mov('PyObject_SetItem',setitem)
                    .if_('PyDict_Type' == type_of(self.var_locals))
                        .mov('PyDict_SetItem',setitem)
                    .endif()
                    .call(setitem,self.var_locals,self.get_name(name),steal(expr),store_ret=ret)
                    (self.check_err(ret,True)))
        else:
            assert not (s.is_free() or self.is_cell(s))
            ret = Var('ret',c_types.t_int)
            (self.code
                .call('PyDict_SetItem',self.var_globals,self.get_name(name),steal(expr),store_ret=ret)
                (self.check_err(ret,True)))

    def assign_expr(self,target,expr,*,check_dest=True):
        if isinstance(target,ast.Name):
            self.assign_name_expr(target.id,expr,check_dest=check_dest)
        elif isinstance(target,ast.Attribute):
            obj = self.visit(target.value)

            (self.code
                .call('PyObject_SetAttr',obj,self.get_name(target.attr),steal(expr))
                (self.check_err(True)))

            obj.discard(self.code)
        elif isinstance(target,ast.Subscript):
            obj = self.visit(target.value)
            slice = self._visit_slice(target.slice)

            (self.code
                .call('PyObject_SetItem',obj,slice,steal(expr))
                (self.check_err(True)))

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

    def delete_name_check(self,ret,name,must_exist,exc,msg):
        ret2 = Var('ret2',c_types.t_int)
        (self.code
            .if_(ret)
                .call('PyErr_ExceptionMatches','PyExc_KeyError',store_ret=ret2))

        if must_exist:
            (self.code
                .if_(ret2)
                    .call('PyErr_Clear')
                    .call('format_exc_check_arg',exc,msg,self.get_name(name))
                .endif()
                (self.exc_cleanup()))
        else:
            (self.code
                .if_(ret2)
                    .call('PyErr_Clear')
                .else_()
                    (self.exc_cleanup())
                .endif())

        self.code.endif()

    def delete_name(self,name,must_exist=True,s=None):
        symbol = self.stable.lookup(name)

        if s is None:
            s = self.code

        if self.is_local(symbol):
            if self.stable.is_optimized() or self.is_cell(s):
                # TODO: this code checks to see if the local has a value. This
                # check could be eliminated in most cases if we were to track
                # assignments and deletions (it would still be required if
                # whether the local is assigned-to depended on a branch, and
                # inside exception handlers).

                tmp1 = Var(data_type=c_types.PyObject_ptr)

                if symbol.is_free():
                    assert self.is_cell(symbol)
                    s.mov(self.var_free,tmp1)
                    item = SIndirect(SIZE_PTR*self.free_var_indexes[name],tmp1)
                else:
                    item = self.local_addrs[name]

                s.mov(item,tmp1)

                if self.is_cell(symbol):
                    tmp2 = tmp1
                    tmp1 = Var(data_type=c_types.PyObject_ptr)

                    item = CType('PyCellObject',tmp2).ob_ref
                    s.mov(item,tmp1)

                s.mov(0,item)

                s.invalidate_scratch()
                s.if_(tmp1).decref(tmp1)
                if must_exist:
                    if self.is_cell(s):
                        exc = 'PyExc_NameError'
                        msg = 'UNBOUNDFREE_ERROR_MSG'
                    else:
                        exc = 'PyExc_UnboundLocalError'
                        msg = 'UNBOUNDLOCAL_ERROR_MSG'
                    (s
                        .else_()
                            .call('format_exc_check_arg',exc,msg,self.get_name(name)))
                s.endif()
            else:
                locals_ = Var('locals_',c_types.PyObject_ptr)
                delitem = Var('delitem',c_types.TPtr(c_types.TFunc([c_types.PyObject_ptr]*2,c_types.PyObject_ptr)))
                ret = Var('ret',c_types.t_int)

                (s
                    .mov(s.special_addrs['locals'],locals_)
                    .if_(not_(locals_))
                        .clear_args()
                        .call('PyErr_Format',
                            'PyExc_SystemError',
                            'NO_LOCALS_DELETE_MSG',
                            self.get_name(name))
                        (self.exc_cleanup())
                    .endif()
                    .mov('PyObject_DelItem',delitem)
                    .if_('PyDict_Type' == type_of(locals_))
                        .mov('PyDict_DelItem',delitem)
                    .endif()
                    .call(delitem,locals_,self.get_name(name),store_ret=ret))

                self.delete_name_check(ret,name,must_exist,'PyExc_UnboundLocalError','UNBOUNDLOCAL_ERROR_MSG')
        else:
            assert not (symbol.is_free() or self.is_cell(symbol))

            ret = Var(c_types.t_int)
            s.call('PyDict_DelItem',self.var_globals,self.get_name(name),store_ret=ret)
            self.delete_name_check(ret,name,must_exist,'PyExc_NameError','GLOBAL_NAME_ERROR_MSG')

    def add_for_assign(self,a,b,fallback,delete):
        # perform the same optimization that unicode_concatenate() from
        # Python/ceval.c does

        arg1 = self.visit(a)
        arg2 = self.visit(b)

        self.visit_Delete(ast.Delete(delete))

        assert isinstance(arg1,PyObject)
        ttest = type_of(arg1) == 'PyUnicode_Type'
        if not isinstance(b,ast.Str):
            assert isinstance(arg2,PyObject)
            ttest = and_(ttest,type_of(arg2) == 'PyUnicode_Type')

        tmp = Var('tmp',c_types.TPtr(c_types.PyObject_ptr))
        r = PyObject()

        (self.code
            .if_(ttest)
                .lea(arg1,tmp)
                .touched_indirectly(arg1,True,False)
                .call('PyUnicode_Append',tmp,arg2)
                .touched_indirectly(arg1,False,True)
                .mov(steal(arg1),r)
            .else_()
                .call(fallback,arg1,arg2,store_ret=r)
                (arg1.discard)
            .endif()
            (self.check_err(r))
            .own(r)
            (arg2.discard))

        return r

    def make_function(self,node : Union[ast.FunctionDef,ast.ClassDef],args : Optional[ast.arguments]=None,returns : Optional[ast.expr]=None):
        body = self.code.cgen.new_func_body()

        func = compile_eval(node,self.sym_map,self.abi,self.u_funcs,self.entry_points,False,node.name,body,args)

        doc = None # type: Optional[ConstValue]

        if sys.flags.optimize < 2:
            d_str = ast.get_docstring(node)
            if d_str is not None:
                doc = self.get_const(d_str)

        defaults = None # type: Optional[MaybeTrackedValue]
        if args and args.defaults:
            defaults = self.visit_Tuple(ast.Tuple(args.defaults,ast.Load()))

        kwdefaults = None # type: Optional[ObjArrayValue]
        if args and args.kw_defaults:
            kwdefaults = ObjArrayValue('kwdefaults')

            (self.code
                .call('PyMem_Malloc',len(args.kwonlyargs)*self.abi.ptr_size,store_ret=kwdefaults)
                (self.check_err(kwdefaults))
                .own(kwdefaults))

            assert args.kwonlyargs and len(args.kwonlyargs) == len(args.kw_defaults)

            for i,a in enumerate(args.kw_defaults):
                if a is not None:
                    val = self.visit(a)
                    self.code.mov(steal(val),SIndirect(i*SIZE_PTR,kwdefaults.value))
                else:
                    self.code.mov(0,SIndirect(i*SIZE_PTR,kwdefaults.value))

        free_items = None # type: Optional[ObjArrayValue]
        if func.free_names:
            free_items = ObjArrayValue('free_items')

            (self.code
                .call('PyMem_Malloc',len(func.free_names)*self.abi.ptr_size,store_ret=free_items)
                (self.check_err(free_items))
                .own(free_items))

            # cells get moved directly and don't need any special handling
            for i,n in enumerate(func.free_names):
                s = self.stable.lookup(n)
                if s.is_free():
                    src = tuple_item(self.var_free,self.free_var_indexes[n])
                else:
                    src = self.local_addrs[n]

                tmp = Var(data_type=c_types.PyObject_ptr)
                (self.code
                    .mov(src,tmp)
                    .mov(tmp,SIndirect(i*SIZE_PTR,free_items.value)))
                if self.is_cell(s):
                    self.code.incref(tmp)
                else:
                    self.code.if_(tmp).incref(tmp).endif()

        annots = None # type: Optional[MaybeTrackedValue]
        a_keys = [] # type: List[ast.AST]
        a_vals = [] # type: List[ast.AST]

        def add_annot(k,v):
            a_keys.append(ast.Str(k))
            a_vals.append(v)

        def add_annot_arg(a):
            if a.annotation:
                add_annot(a.arg,a.annotation)

        if returns: add_annot('return',returns)
        if args:
            for a in args.args: add_annot_arg(a)
            if args.vararg: add_annot_arg(args.vararg)
            for a in args.kwonlyargs: add_annot_arg(a)
            if args.kwarg: add_annot_arg(args.kwarg)

        if a_keys:
            annots = self.visit_Dict(ast.Dict(a_keys,a_vals))

        fobj = PyObject()
        (self.code
            .call('new_function',
                self.get_const(body),
                self.get_name(node.name),
                self.var_globals,
                doc or 0,
                defaults or 0,
                steal(kwdefaults) if kwdefaults else 0,
                steal(free_items) if free_items else 0,
                annots or 0,
                store_ret=fobj)
            (self.check_err(fobj))
            .own(fobj))

        if defaults is not None: defaults.discard(self.code)
        if annots is not None: annots.discard(self.code)

        return fobj

    @staticmethod
    def apply_decorators(obj,decorators):
        alt_node = _ParsedValue(obj)
        for d in reversed(decorators):
            alt_node = ast.Call(d,[alt_node],[],None,None,lineno=d.lineno,col_offset=d.col_offset)

        return alt_node

    def get_const(self,val,node_var=None):
        c = self.consts.get(val)
        if c is None:
            self.consts[val] = c = (address_of(val),len(self.consts))

        r = PyConst('consts',c[1],c[0])
        if node_var:
            self.code.mov(r,node_var)
            return node_var

        return ConstValue(r)

    def get_name(self,n):
        c = self.names.get(n)
        if c is None:
            self.names[n] = c = (address_of(n),len(self.names))
        return ConstValue(PyConst('names',c[1],c[0]))

    def top_context(self,c_type):
        for c in reversed(self.target_contexts):
            if isinstance(c,c_type): return c
        return None

    def new_context(self,c_type):
        self.code.cleanup.depth += 1
        c = c_type(self.code.cleanup.depth)
        self.target_contexts.append(c)
        return c

    def end_context(self,c):
        self.code.cleanup.depth -= 1
        last = self.target_contexts.pop()
        assert c is last

    def unwind_to(self,target: Target,until: Optional[ScopeContext]=None,s: Optional[Stitch]=None,last_action: Optional[Callable[[Stitch],None]]=None):
        assert not isinstance(until,FinallyContext)

        if s is None: s = self.code

        item_groups = []
        callbacks = []
        max_depth = None

        def clean_context(depth):
            nonlocal max_depth

            items = []
            for item,item_depth in s.state.owned_items():
                if item_depth >= depth and (max_depth is None or item_depth < max_depth):
                    items.append(CleanupItem(item.value,item.cleanup_func))

            items.extend(callbacks)
            max_depth = depth
            return items

        for c in reversed(self.target_contexts):
            for ec in c.extra_callbacks: callbacks.append(ec)

            if isinstance(c,FinallyContext):
                action = CallTargetAction(c.ftarget)
                item_groups.append((clean_context(c.depth),action))
                c.ftarget.add_next_target(action.ret_target)
                callbacks.clear()
            if c is until:
                item_groups.append((clean_context(c.depth),None))
                break
        else:
            assert until is None
            item_groups.append((clean_context(1),last_action))

        for items,action in reversed(item_groups):
            target = s.cleanup.new_section(s,items,target,action)

        s(target)
        s.last_op_is_uncond_jmp = True

    def set_return_value(self,val):
        # bind the current value
        check_var_return = self.check_var_return

        # this just avoids a reference cycle
        var_return = self.var_return

        def inner(s):
            if check_var_return:
                s.if_(var_return).decref(var_return).endif()
            s.mov(val,var_return)
        return inner

    def exc_cleanup(self):
        """Free the stack items created at the current exception handler depth.
        """
        def inner(s):
            dest = self.func_end
            top = self.top_context(ExceptContext)
            if top: dest = top.target
            self.unwind_to(dest,top,s,self.set_return_value(0))
        return inner

    def check_err(self,v,inverted=False):
        if not inverted: v = not_(v)
        return (lambda s:
            s.if_(v)(self.exc_cleanup()).endif())

    # these are cached so that:
    # self.delete_name_cleanup(x) == self.delete_name_cleanup(x)
    # and thus can be deduplicated by StackCleanup
    def delete_name_cleanup(self,name):
        c = self._del_name_cleanups.get(name)
        if c is None:
            c = lambda s: self.delete_name(name,False,s)
            self._del_name_cleanups[name] = c

        return c

    def test_condition(self,node):
        test = self.visit(node)
        istrue = signed(Var('istrue',data_type=c_types.t_int))

        (self.code
            .call('PyObject_IsTrue',test,store_ret=istrue)
            (test.discard)
            .if_(istrue < 0)
                (self.exc_cleanup())
            .endif()
            .if_(istrue != 0))


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

    @visit_extra
    def visit(self,node,*extra):
        return getattr(self,'visit_' + node.__class__.__name__)(node,*extra)

    def _visit_slice(self,slice_):
        if isinstance(slice_,ast.Index):
            return self.visit(slice_.value)

        if isinstance(slice_,ast.Slice):
            start = None
            if slice_.lower:
                start = self.visit(slice_.lower)

            end = None
            if slice_.upper:
                end = self.visit(slice_.upper)

            step = None
            if slice_.step:
                step = self.visit(slice_.step)

            r = PyObject(own=False)
            (self.code
                .call('PySlice_New',
                    start or 0,
                    end or 0,
                    step or 0,
                    store_ret=r)
                (self.check_err(r))
                .own(r))

            if start: start.discard(self.code)
            if end: end.discard(self.code)
            if step: step.discard(self.code)

            return r

        if isinstance(slice_,ast.ExtSlice):
            raise NotImplementedError()

        raise SyntaxTreeError('invalid index type in subscript')

    def visit__ParsedValue(self,node):
        return node.value

    def visit_Module(self,node):
        assert False

    def visit_FunctionDef(self,node):
        fobj = self.make_function(node,node.args,node.returns)

        self.visit(ast.Assign(
            [ast.Name(node.name,ast.Store())],
            self.apply_decorators(fobj,node.decorator_list),
            lineno=node.lineno,
            col_offset=node.col_offset))

        fobj.discard(self.code)

    def visit_ClassDef(self,node):
        #build_class = PyObject('build_class')
        #frame = CType('PyFrameObject',self.var_frame)

        #(self.code
        #    .call('_load_build_class',frame.f_builtins,store_ret=build_class)
        #    (self.check_err(build_class))
        #    .own(build_class))

        body_func = self.make_function(node)
        new_class = self.visit_Call(
            ast.Call(
                #_ParsedValue(build_class),
                _ParsedValue(self.get_const(pyinternals.build_class)),
                [_ParsedValue(body_func),_ParsedValue(self.get_name(node.name))] + node.bases,
                node.keywords,
                None,
                None))

        self.visit(ast.Assign(
            [ast.Name(node.name,ast.Store())],
            self.apply_decorators(new_class,node.decorator_list),
            lineno=node.lineno,
            col_offset=node.col_offset))

        new_class.discard(self.code)

    def visit_Return(self,node):
        if self.check_var_return:
            (self.code
                .if_(self.var_return)
                    .decref(self.var_return)
                .endif()
                .mov(0,self.var_return))

        for c in self.target_contexts:
            if isinstance(c,FinallyContext):
                c.may_return = True

        var_r = PyObject(self.var_return)
        if node.value:
            val = self.visit(node.value,var_r)
        else:
            val = self.get_const(None,var_r)

        if isinstance(val,TrackedValue):
            steal(val)(self.code)

        self.unwind_to(self.func_end)
        #for oval,depth in self.code.state.owned_items():
        #    if oval is not val:
        #        oval.discard(self.code)
        #if isinstance(val,TrackedValue):
        #    self.code.state.disown(val)

        return True

    def visit_Delete(self,node):
        for target in node.targets:
            if isinstance(target,ast.Name):
                self.delete_name(target.id)
            elif isinstance(target,ast.Attribute):
                tmp = Var('tmp',data_type=c_types.t_int)
                obj = self.visit(target.value)

                (self.code
                    .call('PyObject_DelAttr',obj,self.get_name(target.attr),store_ret=tmp)
                    (self.check_err(tmp,True)))

                obj.discard(self.code)
            elif isinstance(target,ast.Subscript):
                tmp = Var('tmp',data_type=c_types.t_int)
                obj = self.visit(target.value)
                slice_ = self._visit_slice(target.slice)

                (self.code
                    .call('PyObject_DelItem',obj,slice_,store_ret=tmp)
                    (self.check_err(tmp,True)))

                slice_.discard(self.code)
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

            expr = PyObject()

            if op is ast.Pow:
                (self.code
                    .call('PyNumber_InPlacePower',arg2,arg1,'Py_None',store_ret=expr)
                    (self.check_err(expr)))
            else:
                (self.code
                    .call(BINOP_IFUNCS[op],arg1,arg2,store_ret=expr)
                    (self.check_err(expr)))

            self.code.own(expr)

            arg1.discard(self.code)
            arg2.discard(self.code)

        self.assign_expr(node.target,expr,check_dest=check_dest)

    def visit_For(self,node):
        exhaust = Target()
        i_type = CType('PyTypeObject',Var('i_type',data_type=c_types.PyTypeObject_ptr))
        itr = PyObject('itr')

        itr_val = self.visit(node.iter)

        (self.code
            .call('PyObject_GetIter',itr_val,store_ret=itr)
            (itr_val.discard)
            (self.check_err(itr))
            .own(itr))

        next_val = PyObject('next_val')
        occurred = Var('occurred',data_type=c_types.PyObject_ptr)
        matches = Var('matches',data_type=c_types.t_int)

        context = self.new_context(LoopContext)

        (self.code
            (context.begin)
            .mov(type_of(itr),i_type)
            .call(i_type.tp_iternext,itr,store_ret=next_val)
            .if_(not_(next_val))
                .call('PyErr_Occurred',store_ret=occurred)
                .if_(occurred)
                    .call('PyErr_ExceptionMatches','PyExc_StopIteration',store_ret=matches)
                    (self.check_err(matches))
                    .call('PyErr_Clear')
                .endif()
                .jmp(exhaust)
            .endif())

        self.assign_expr(node.target,next_val)
        next_val.discard(self.code)
        for stmt in node.body:
            self.visit(stmt)

        (self.code
            .jmp(context.begin)
            (exhaust))

        self.end_context(context)
        for stmt in node.orelse:
            self.visit(stmt)

        (self.code
            (context.break_)
            (itr.discard))

    def visit_While(self,node):
        context = self.new_context(LoopContext)

        if_quit = else_quit = False

        self.code(context.begin)
        self.test_condition(node.test)

        for stmt in node.body:
            if self.visit(stmt):
                if_quit = True
                break

        self.code.jmp(context.begin)
        self.end_context(context)

        if node.orelse:
            self.code.else_()
            for stmt in node.orelse:
                if self.visit(stmt):
                    else_quit = True
                    break

        self.code(context.break_)

        return if_quit and else_quit

    def visit_If(self,node):
        self.test_condition(node.test)

        if_quit = else_quit = False

        for stmt in node.body:
            if self.visit(stmt):
                if_quit = True
                break

        if node.orelse:
            self.code.else_()
            for stmt in node.orelse:
                if self.visit(stmt):
                    else_quit = True
                    break

        self.code = self.code.endif()

        return if_quit and else_quit

    def visit_With(self,node):
        items_head,*items_tail = node.items

        val = self.visit(items_head.context_expr)
        exit_ = PyObject('exit')
        enter = PyObject('enter')
        as_obj = PyObject('as_obj')

        (self.code
            .call('special_lookup',val,self.get_name('__exit__'),store_ret=exit_)
            (self.check_err(exit_))
            .own(exit_)
            .call('special_lookup',val,self.get_name('__enter__'),store_ret=enter)
            (val.discard)
            (self.check_err(enter))
            .own(enter)
            .call('PyObject_CallObject',enter,0,store_ret=as_obj)
            (enter.discard)
            .own(as_obj))

        t_context = self.new_context(TryContext)
        f_context = self.new_context(FinallyContext)

        if items_head.optional_vars is not None:
            self.assign_expr(items_head.optional_vars,as_obj)
        as_obj.discard(self.code)

        if items_tail:
            self.visit_With(ast.With(items_tail,node.body))
        else:
            for stmt in node.body:
                self.visit(stmt)

        self.unwind_to(t_context.target,t_context)

        self.end_context(f_context)
        with FinallyBody(self,f_context):
            exit_r = PyObject('exit_r')

            exc_vals = [PyObject(nullable=True) for _ in range(3)]
            addr_type = c_types.TPtr(c_types.PyObject_ptr)
            exc_addrs = [Var(data_type=addr_type) for _ in range(3)]

            for val,addr in zip(exc_vals,exc_addrs):
                self.code.lea(val,addr)

            self.code.call('PyErr_Fetch',*exc_addrs)

            for val in exc_vals:
                self.code.touched_indirectly(val,False,True).own(val)

            args_tup = PyObject()
            (self.code
                .call('PyTuple_New',3,store_ret=args_tup)
                (self.check_err(args_tup))
                .own(args_tup)
                .if_(exc_vals[0]))

            for i,val in enumerate(exc_vals):
                (self.code
                    .mov(val,tuple_item(args_tup,i))
                    .incref(val))

            tmp = Var(data_type=c_types.PyObject_ptr)
            (self.code
                .else_()
                    .mov('Py_None',tmp))
            for i in range(len(exc_vals)): self.code.mov(tmp,tuple_item(args_tup,i))
            (self.code
                    .incref(tmp,3)
                .endif())

            (self.code
                .call('PyObject_CallObject',exit_,args_tup,store_ret=exit_r)
                (args_tup.discard)
                (self.check_err(exit_r))
                .own(exit_r)
                .if_(exc_vals[0])
                    .call('PyObject_IsTrue',exit_r,store_ret=tmp)
                    .if_(signed(tmp) < 0)
                        (self.exc_cleanup())
                    .endif()
                    .if_(tmp)
                        (exc_vals[0].discard_non_null)
                        (exc_vals[1].discard_non_null)
                        (exc_vals[2].discard))

            self.unwind_to(t_context.target,t_context)

            (self.code
                    .else_()
                        .call('PyErr_Restore',*[steal(v) for v in exc_vals])
                    .endif()
                .endif()
                (exit_r.discard))

        self.end_context(t_context)
        self.code(t_context.target)(exit_.discard)

    def visit_Raise(self,node):
        raise NotImplementedError()

    def visit_Try(self,node):
        top_context = t_context = self.new_context(TryContext)
        e_context = f_context = None

        if node.finalbody:
            top_context = f_context = self.new_context(FinallyContext)

        if node.handlers:
            e_context = self.new_context(ExceptContext)

        for stmt in node.body:
            if self.visit(stmt): break

        if node.handlers:
            self.end_context(e_context)

        for stmt in node.orelse:
            if self.visit(stmt): break
        else:
            self.unwind_to(t_context.target,t_context)

        if node.handlers:
            exc_val_block = Block(6,base_type=c_types.PyObject_ptr)
            exc_vals = [PyObject(v,nullable=i>1) for i,v in enumerate(exc_val_block)]
            exc_type,exc_val,exc_tb = exc_vals[0:3]

            tmp = Var(data_type=c_types.TPtr(c_types.PyObject_ptr))
            (self.code
                (e_context.target)
                .lea(exc_val_block,tmp)
                .call('prepare_exc_handler',tmp)
                .touched_indirectly(exc_val_block,False,True))

            for v in exc_vals: self.code.own(v)

            def do_handler(handlers):
                exc = handlers.pop(0)
                assert exc.type or not handlers

                if exc.type:
                    e_type = self.visit(exc.type)
                    is_sub = Var(data_type=c_types.t_int)
                    self.code.call('PyObject_IsSubclass',exc_type,e_type,store_ret=is_sub)
                    e_type.discard(self.code)
                    self.code.if_(is_sub)
                    if exc.name:
                        self.assign_name_expr(exc.name,exc_val)
                        cleanup = self.delete_name_cleanup(exc.name)
                        top_context.extra_callbacks.append(cleanup)

                for val in exc_vals[0:3]: val.discard(self.code)


                for stmt in exc.body:
                    if self.visit(stmt): break

                if exc.type:
                    if exc.name:
                        # noinspection PyUnboundLocalVariable
                        top_context.extra_callbacks.remove(cleanup)
                        cleanup(self.code)
                    self.code.else_()

                    if handlers:
                        do_handler(handlers)
                    else:
                        (self.code
                            .call('PyErr_Restore',
                                steal(exc_type),
                                steal(exc_val),
                                steal(exc_tb))
                            (self.exc_cleanup()))

                    self.code.endif()

            do_handler(node.handlers[:])

            tmp = Var(data_type=c_types.TPtr(c_types.PyObject_ptr))
            (self.code
                .lea(exc_vals[3],tmp)
                .call('end_exc_handler',tmp))

            for val in exc_vals[3:]: self.code.disown(val)

            self.unwind_to(t_context.target,t_context)

        if node.finalbody:
            self.end_context(f_context)
            with FinallyBody(self,f_context) as fb:
                exc_vals = [PyObject(),PyObject(),PyObject()]
                addr_type = c_types.TPtr(c_types.PyObject_ptr)
                exc_addrs = [Var(data_type=addr_type) for _ in range(3)]

                for val,addr in zip(exc_vals,exc_addrs):
                    self.code.lea(val,addr)

                self.code.call('PyErr_Fetch',*exc_addrs)

                for val in exc_vals:
                    self.code.touched_indirectly(val,False,True).own(val)

                for stmt in node.finalbody:
                    if self.visit(stmt):
                        for v in exc_vals: v.discard(self.code)
                        fb.no_return = True
                        break
                else:
                    (self.code
                        .if_(exc_vals[0])
                            .call('PyErr_Restore',*[steal(v) for v in exc_vals])
                        .endif())

        self.end_context(t_context)
        self.code(t_context.target)

    def visit_Assert(self,node):
        raise NotImplementedError()

    def visit_Import(self,node):
        raise NotImplementedError()

    def visit_ImportFrom(self,node):
        raise NotImplementedError()

    def visit_Pass(self,node):
        pass

    # these are handled by the symbol table, so we don't have to do anything
    # here
    visit_Global = visit_Pass
    visit_Nonlocal = visit_Pass

    def visit_Expr(self,node):
        self.visit(node.value).discard(self.code)

    def visit_Break(self,node):
        context = self.top_context(LoopContext)
        self.unwind_to(context.break_,context)
        return True

    def visit_Continue(self,node):
        context = self.top_context(LoopContext)
        self.unwind_to(context.begin,context)
        return True

    def  _boolop_iteration(self,op,values,r):
        if len(values) > 1:
            val = self.visit(values[0])
            istrue = Var(data_type=c_types.t_int)

            (self.code
                .call('PyObject_IsTrue',val,store_ret=istrue)
                .if_(signed(istrue) < 0)
                    (self.exc_cleanup())
                .elif_(istrue if isinstance(op,ast.Or) else not_(istrue))
                    .mov(steal(val),r)
                    .own(r)
                .else_()
                    (val.discard))
            self._boolop_iteration(op,values[1:],r)
            (self.code
                .endif())
        else:
            self.code.mov(steal(self.visit(values[0])),r).own(r)

    def visit_BoolOp(self,node,node_var=None):
        r = PyObject(node_var)
        self._boolop_iteration(node.op,node.values,r)
        return r

    def visit_BinOp(self,node,node_var=None):
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

        r = PyObject(node_var)

        if op is ast.Add and maybe_unicode(node.left) and maybe_unicode(node.right):
            str_type = Symbol('PyUnicode_Type')

            if isinstance(node.left,ast.Str):
                ttest = type_of(arg2) == str_type
            else:
                ttest = type_of(arg1) == str_type
                if not isinstance(node.right,ast.Str):
                    ttest = and_(ttest,type_of(arg2) == str_type)

            f = Var(data_type=c_types.TPtr(c_types.TFunc([c_types.PyObject_ptr,c_types.PyObject_ptr],c_types.PyObject_ptr)))
            (self.code
                .mov('PyNumber_Add',f)
                .if_(ttest)
                    .mov('PyUnicode_Concat',f)
                .endif()
                .call(f,arg1,arg2,store_ret=r)
                (self.check_err(r)))
        elif op is ast.Mod:
            func = Var(data_type=c_types.TPtr(c_types.TFunc([c_types.PyObject_ptr,c_types.PyObject_ptr],c_types.PyObject_ptr)))

            uaddr = Symbol('PyUnicode_Type')

            (self.code
                .mov('PyUnicode_Format',func)
                .if_(uaddr != type_of(arg1))
                    .mov('PyNumber_Remainder',func)
                .endif()
                .call(func,arg1,arg2,store_ret=r)
                (self.check_err(r)))
        elif op is ast.Pow:
            (self.code
                .call('PyNumber_Power',arg2,arg1,'Py_None',store_ret=r)
                (self.check_err(r)))
        else:
            (self.code
                .call(BINOP_FUNCS[op],arg1,arg2,store_ret=r)
                (self.check_err(r)))

        self.code.own(r)

        arg1.discard(self.code)
        arg2.discard(self.code)

        return r

    def visit_UnaryOp(self,node,node_var=None):
        arg = self.visit(node.operand)
        r = PyObject(node_var)

        op = type(node.op)
        if op is ast.Not:
            istrue = signed(Var(data_type=c_types.t_int))
            (self.code
                .call('PyObject_IsTrue',arg,store_ret=istrue)
                .mov('Py_True',r)
                .if_(istrue > 0)
                    .mov('Py_False',r)
                .elif_(istrue != 0)
                    (self.exc_cleanup())
                .endif()
                .incref(r))
        else:
            if op is ast.Invert:
                func = 'PyNumber_Invert'
            elif op is ast.UAdd:
                func = 'PyNumber_Positive'
            elif op is ast.USub:
                func = 'PyNumber_Negative'
            else:
                raise SyntaxTreeError('unrecognized unary operation type encountered')

            self.code.call(func,arg,store_ret=r)(self.check_err(r))

        self.code.own(r)(arg.discard)
        return r

    def visit_Lambda(self,node,node_var=None):
        raise NotImplementedError()
    def visit_IfExp(self,node,node_var=None):
        r = PyObject() if node_var is None else node_var

        if isinstance(node.body,CONST_NODES) and isinstance(node.orelse,CONST_NODES):
            # if both expressions are constants, we set the result to the
            # "else" value and overwrite it if the condition is true

            t_val = self.visit(node.body)
            f_val = self.visit(node.orelse)

            assert isinstance(t_val,ConstValue) and isinstance(f_val,ConstValue)

            self.code.mov(f_val,r)
            self.test_condition(node.test)
            self.code.mov(t_val,r)
            self.code.endif()
        else:
            self.test_condition(node.test)
            self.visit(node.body,r)
            self.code.else_()
            self.visit(node.orelse)
            self.code.endif()

        return r

    def visit_Dict(self,node,node_var=None):
        r = PyObject() if node_var is None else node_var
        (self.code
            .call('_PyDict_NewPresized',len(node.keys),store_ret=r)
            (self.check_err(r))
            .own(r))

        for k,v in zip(node.keys,node.values):
            kobj = self.visit(k)
            vobj = self.visit(v)
            err = Var(data_type=c_types.t_int)

            (self.code
                .call('PyDict_SetItem',r,kobj,vobj,store_ret=err)
                (self.check_err(err,True)))

            kobj.discard(self.code)
            vobj.discard(self.code)

        return r

    def visit_Set(self,node,node_var=None):
        raise NotImplementedError()
    def visit_ListComp(self,node,node_var=None):
        raise NotImplementedError()
    def visit_SetComp(self,node,node_var=None):
        raise NotImplementedError()
    def visit_DictComp(self,node,node_var=None):
        raise NotImplementedError()
    def visit_GeneratorExp(self,node,node_var=None):
        raise NotImplementedError()
    def visit_Yield(self,node,node_var=None):
        raise NotImplementedError()
    def visit_YieldFrom(self,node,node_var=None):
        raise NotImplementedError()

    def _compare_check_bool(self,istrue,invert_test,b,ops,comparators,r,end):
        (self.code
            .if_((istrue > 0) if invert_test else (istrue == 0))
            .mov('Py_False',r)
            .incref(r)
            .jmp(end)
            .elif_(istrue < 0)
                (self.exc_cleanup())
            .endif())

        if len(ops) > 1:
            self._compare_iteration(b,ops[1:],comparators[1:],r,end)
        else:
            b.discard(self.code)
            (self.code
                .mov('Py_True',r)
                .incref(r))

    def _compare_iteration(self,a,ops,comparators,r,end):
        assert len(ops) == len(comparators)

        b = self.visit(comparators[0])
        istrue = signed(Var('istrue',data_type=c_types.t_int))

        op_t = ops[0].__class__
        if op_t in (ast.In,ast.NotIn):
            (self.code
                .call('PySequence_Contains',b,a,store_ret=istrue)
                (a.discard))

            self._compare_check_bool(istrue,op_t == ast.NotIn,b,ops,comparators,r,end)
        elif op_t in (ast.Is,ast.IsNot):
            # we don't need its value, only its address
            a.discard(self.code)

            isfalse = (a != b) if op_t == ast.Is else (a == b)

            if len(ops) > 1:
                (self.code
                    .if_(isfalse)
                        .mov('Py_False',r)
                        .incref(r)
                        .jmp(end)
                    .endif())

                self._compare_iteration(b,ops[1:],comparators[1:],r,end)
            else:
                (self.code
                    (b.discard)
                    .mov('Py_True',r)
                    .if_(isfalse)
                        .mov('Py_False',r)
                    .endif()
                    .incref(r))
        else:
            assert op_t in RICH_CMP_NODES
            (self.code
                .call('PyObject_RichCompare',a,b,RICH_CMP_NODES.index(op_t),store_ret=r)
                (a.discard))

            if len(ops) > 1:
                (self.code
                    (self.check_err(r))
                    .call('PyObject_IsTrue',r,store_ret=istrue)
                    .decref(r))

                self._compare_check_bool(istrue,False,b,ops,comparators,r,end)
            else:
                (self.code
                    (self.check_err(r))
                    (b.discard))

    def visit_Compare(self,node,node_var=None):
        a = self.visit(node.left)
        r = PyObject() if node_var is None else node_var
        end = Target()

        self._compare_iteration(a,node.ops,node.comparators,r,end)
        self.code(end).own(r)
        return r

    def visit_Call(self,node,node_var=None):
        fobj = self.visit(node.func)

        args_obj = 0
        if node.args:
            args_obj = PyObject('args_obj')

            (self.code
                .call('PyTuple_New',len(node.args),store_ret=args_obj)
                (self.check_err(args_obj))
                .own(args_obj))

            for i,a in enumerate(node.args):
                aobj = self.visit(a)
                self.code.mov(steal(aobj),tuple_item(args_obj,i))

        if node.starargs:
            s_args_obj = self.visit(node.starargs)

            if args_obj:
                args_obj = steal(args_obj)

            new_args_obj = PyObject()

            (self.code
                .call('append_tuple_for_call',fobj,args_obj,s_args_obj,store_ret=new_args_obj)
                (self.check_err(new_args_obj))
                .own(new_args_obj))

            args_obj = new_args_obj

            s_args_obj.discard(self.code)

        args_kwds = 0
        if node.keywords:
            args_kwds = PyObject('args_kwds')
            (self.code
                .call('_PyDict_NewPresized',len(node.keywords),store_ret=args_kwds)
                (self.check_err(args_kwds))
                .own(args_kwds))

            for kwds in node.keywords:
                obj = self.visit(kwds.value)
                err = Var(data_type=c_types.t_int)

                (self.code
                    .call('PyDict_SetItem',args_kwds,self.get_name(kwds.arg),obj,store_ret=err)
                    (self.check_err(err,True)))

                obj.discard(self.code)

        if node.kwargs:
            s_kwds = self.visit(node.kwargs)

            if args_kwds:
                args_kwds = steal(args_kwds)

            new_args_kwds = PyObject()

            (self.code
                .call('append_dict_for_call',fobj,args_kwds,s_kwds,store_ret=new_args_kwds)
                (self.check_err(new_args_kwds))
                .own(new_args_kwds))

            args_kwds = new_args_kwds

            s_kwds.discard(self.code)

        r = PyObject() if node_var is None else node_var
        (self.code
            .call('PyObject_Call',
                fobj,
                args_obj or self.get_const(()),
                args_kwds,
                store_ret=r)
            (self.check_err(r))
            .own(r))

        if args_kwds: args_kwds.discard(self.code)
        if args_obj: args_obj.discard(self.code)
        fobj.discard(self.code)

        return r

    def visit_Num(self,node,node_var=None):
        return self.get_const(node.n,node_var)

    def visit_Str(self,node,node_var=None):
        return self.get_const(node.s,node_var)

    def visit_Bytes(self,node,node_var=None):
        return self.get_const(node.s,node_var)

    def visit_NameConstant(self,node,node_var=None):
        c = ConstValue(Immediate(address_of(node.value),c_types.PyObject_ptr))
        if node_var:
            self.code.mov(c,node_var)
            return node_var

        return c

    def visit_Ellipsis(self,node,node_var=None):
        return self.get_const(...,node_var)

    def visit_Attribute(self,node,node_var=None):
        obj = self.visit(node.value)

        r = PyObject() if node_var is None else node_var
        (self.code
            .call('PyObject_GetAttr',obj,self.get_name(node.attr),store_ret=r)
            (self.check_err(r))
            .own(r))

        obj.discard(self.code)

        return r

    def visit_Subscript(self,node,node_var=None):
        value = self.visit(node.value)
        slice_ = self._visit_slice(node.slice)

        r = PyObject() if node_var is None else node_var
        (self.code
            .call('PyObject_GetItem',value,slice_,store_ret=r)
            (self.check_err(r))
            .own(r))

        value.discard(self.code)
        slice_.discard(self.code)

        return r

    def visit_Starred(self,node,node_var=None):
        raise NotImplementedError()
    def visit_Name(self,node,node_var=None):
        ovr = self.name_overrides.get(node.id)
        s = self.stable.lookup(node.id)

        dbg_name = 'py:'+node.id

        if ovr is not None or s.is_free() or (self.is_local(s) and self.stable.is_optimized()) or self.is_cell(s):
            r = PyObject(dbg_name) if node_var is None else node_var

            if ovr is not None:
                self.code.mov(ovr,r)
            else:
                if s.is_free():
                    (self.code
                        .mov(self.var_free,r)
                        .mov(SIndirect(self.free_var_indexes[node.id]*SIZE_PTR,r.value),r))
                else:
                    self.code.mov(self.local_addrs[node.id],r)

                if self.is_cell(s):
                    self.code.mov(CType('PyCellObject',r).ob_ref,r)

            (self.code
                .if_(not_(r))
                    .call('format_exc_check_arg',
                        'PyExc_UnboundLocalError',
                        'UNBOUNDLOCAL_ERROR_MSG',
                        self.get_name(node.id))
                    (self.exc_cleanup())
                .endif())
            # we don't call own because we didn't increment the reference count

            return r

        r = PyObject(dbg_name) if node_var is None else node_var

        (self.code
            .call(self.u_funcs['local_name' if self.is_local(s) else 'global_name'].addr,
                self.get_name(node.id),
                self.var_frame,
                store_ret=r,
                callconv=CallConvType.utility)
            (self.check_err(r))
            .own(r))
        return r

    def visit_List(self,node,node_var=None):
        item_offset = pyinternals.member_offsets['PyListObject']['ob_item']

        r = PyObject() if node_var is None else node_var
        tmp = Var(data_type=c_types.TPtr(c_types.PyObject_ptr))
        (self.code
            .call('PyList_New',len(node.elts),store_ret=r)
            (self.check_err(r))
            .own(r)
            .mov(SIndirect(item_offset,r),tmp))

        for i,item in enumerate(node.elts):
            obj = self.visit(item)

            self.code.mov(steal(obj),SIndirect(SIZE_PTR * i,tmp))

        return r

    def visit_Tuple(self,node,node_var=None):
        r = PyObject() if node_var is None else node_var
        (self.code
            .call('PyTuple_New',len(node.elts),store_ret=r)
            (self.check_err(r))
            .own(r))

        for i,item in enumerate(node.elts):
            self.code.mov(steal(self.visit(item)),tuple_item(r,i))

        return r

def local_name_func(abi : Abi):
    ret = Target()
    inc_ret = Target()
    format_exc = Target()

    name = Var(data_type=c_types.PyObject_ptr)
    frame = CType('PyFrameObject',Var(data_type=c_types.PyFrameObject_ptr))
    locals_ = Var(data_type=c_types.PyObject_ptr)
    builtins = Var(data_type=c_types.PyObject_ptr)
    r = Var(data_type=c_types.PyObject_ptr)

    s = Stitch(abi.code_gen(abi,[name,frame.base],CallConvType.utility))

    (s
        .mov(frame.f_locals,locals_)
        .if_(not_(locals_))
            .call('PyErr_Format','PyExc_SystemError','NO_LOCALS_LOAD_MSG',name,store_ret=r)
            .jmp(ret)
        .endif()

        .if_('PyDict_Type' != type_of(locals_))
            .call('PyObject_GetItem',locals_,name,store_ret=r)
            .jump_if(r,ret)

            .call('PyErr_ExceptionMatches','PyExc_KeyError',store_ret=r)
            .jump_if(not_(r),ret)
            .call('PyErr_Clear')
        .else_()
            .call('PyDict_GetItem',locals_,name,store_ret=r)
            .jump_if(r,inc_ret)
        .endif()

        .call('PyDict_GetItem',frame.f_globals,name,store_ret=r)
        .if_(not_(r))
            .mov(frame.f_builtins,builtins)
            .if_('PyDict_Type' != type_of(builtins))
                .call('PyObject_GetItem',builtins,name,store_ret=r)
                .jump_if(r,ret)

                .call('PyErr_ExceptionMatches','PyExc_KeyError',store_ret=r)
                .jump_if(not_(r),ret)
                .jmp(format_exc)
            .else_()
                .call('PyDict_GetItem',builtins,name,store_ret=r)
                .jump_if(r,inc_ret)

                (format_exc)
                .call('format_exc_check_arg',
                    'PyExc_NameError',
                    'NAME_ERROR_MSG',
                    name)
                .jmp(ret)
            .endif()
        .endif()

        (inc_ret)
        .incref(r)
        (ret))

    return s.cgen.compile('local_name',destitch(s),r)

def global_name_func(abi : Abi):
    ret = Target()
    name_err = Target()

    name = Var('name',data_type=c_types.PyObject_ptr)
    frame = CType('PyFrameObject',Var(data_type=c_types.PyFrameObject_ptr))
    globals_ = Var('globals')
    builtins = Var('builtins_')

    s = Stitch(abi.code_gen(abi,[name,frame.base],CallConvType.utility))
    r = Var('r',data_type=c_types.PyObject_ptr)

    (s
        .mov(frame.f_globals,globals_)
        .mov(frame.f_builtins,builtins)
        .if_(and_('PyDict_Type' == type_of(globals_),'PyDict_Type' == type_of(builtins)))
            .call('_PyDict_LoadGlobal',globals_,builtins,name,store_ret=r)
            .if_(not_(r))
                .call('PyErr_Occurred',store_ret=r)
                .jump_if(not_(r),name_err)
                .mov(0,r)
                .jmp(ret)
            .endif()
            .incref(r)
        .else_()
            .call('PyObject_GetItem',globals_,name,store_ret=r)
            .if_(not_(r))
                .call('PyObject_GetItem',builtins,name,store_ret=r)
                .if_(not_(r))
                    .call('PyErr_ExceptionMatches','PyExc_KeyError',store_ret=r)
                    .if_(not_(r))
                        (name_err)
                        .call('format_exc_check_arg',
                            'PyExc_NameError',
                            'GLOBAL_NAME_ERROR_MSG',
                            name)
                    .endif()
                    .mov(0,r)
                    .jmp(ret)
                .endif()
            .endif()
        .endif()
        (ret))

    return s.cgen.compile('global_name',destitch(s),r)

def resume_generator_func(abi : Abi):
    # the parameter is expected to be an address to a Generator instance

    generator = CType('Generator')
    body = CType('FunctionBody')
    stack = Var('stack')
    entry = Var('entry')
    stack_b_size = Var('stack_b_size')
    i = Var('i')

    s = Stitch(abi.code_gen(abi,[generator.base]))

    (s
        .mov(generator.stack_size,i)
        .mov(generator.body,body)
        .mov(generator.stack,stack)
        .mul(i,8,stack_b_size)
        .mov(body.entry,entry)
        #.sub(R_SP,stack_b_size)
        .add(entry,generator.offset)
        .do()
            .sub(i,1)
            #.push(SIndirect(0,stack,i))
        .while_(signed(i) != 0)
        .jmp(entry))

    return s.cgen.compile('resume_generator',destitch(s))


ProtoFunction = namedtuple('ProtoFunction',['name','code'])

class PyFunction:
    def __init__(self,fb_obj,func,cgen,ret_var,names,args,free_names,cells,consts):
        self.fb_obj = cgen.new_func_body() if fb_obj is None else fb_obj
        self.func = func
        self.cgen = cgen
        self.ret_var = ret_var

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

class UtilityFunction:
    def __init__(self,name,code,offset):
        self.name = name
        self.code = code
        self.offset = offset

    @property
    def addr(self):
        return Symbol(self.name,address=self.code.start_addr + self.offset)


def ordered_consts(consts):
    empty = object()

    r = [empty] * len(consts)
    for n,c in consts.items():
        assert r[c[1]] is empty
        r[c[1]] = n

    assert empty not in r
    return tuple(r)

def compile_eval(
        scope_ast : Union[ast.Module,ast.FunctionDef,ast.ClassDef],
        sym_map : Dict[int,symtable.SymbolTable],
        abi_ : Abi,
        u_funcs : Dict[str,UtilityFunction],
        entry_points : List,
        global_scope : bool,
        name : str='<module>',
        fb_obj : Optional[pyinternals.FunctionBody]=None,
        args : Optional[ast.arguments]=None) -> PyFunction:
    """Compile scope_ast into an intermediate representation"""

    #move_throw_flag = (code.co_flags & CO_GENERATOR and len(abi.r_arg) >= 2)
    #mov_throw_flag = False

    f_obj = CType('PyFrameObject',Var('var_frame',c_types.PyFrameObject_ptr,dbg_symbol='__frame'))
    func_self = CType('Function',Var('fun_self',c_types.Function_ptr,dbg_symbol='__func'))
    func_args = Var('func_args',c_types.PyObject_ptr,dbg_symbol='__args')
    func_kwds = Var('func_kwds',c_types.PyObject_ptr,dbg_symbol='__kwds')

    param_vars = [f_obj.base,func_self.base,func_args,func_kwds]
    cgen = abi_.code_gen(abi_,param_vars,pyinfo=PyFuncInfo(*param_vars))

    # noinspection PyUnresolvedReferences
    ec = ExprCompiler(cgen,f_obj.base,sym_map[scope_ast._raw_id],sym_map,u_funcs,global_scope,entry_points)

    #if ec.stable.get_globals():
    if True:
        ec.code.mov(f_obj.f_globals,ec.var_globals)

    if not ec.stable.is_optimized():
        ec.code.mov(f_obj.f_locals,ec.var_locals)

    has_free = isinstance(ec.stable,symtable.Function) and ec.stable.get_frees()
    if has_free:
        ec.code.mov(func_self.closure,ec.var_free)

    # we set this here so that ec.var_return is never only conditionally set,
    # which would otherwise be the case if e.g. a return statement appears
    # inside a try-finally statement
    ec.code.mov(0,ec.var_return)

    #naddr = steal(ec.code.fit_addr('Py_None',R_RET))

    # if False: #STATE.code.co_flags & CO_GENERATOR:
    #     (ec.code
    #         .mov(f_obj.f_exc_type,R_SCRATCH2)
    #         .if_(and_(R_SCRATCH2,R_SCRATCH2 != naddr))
    #             .inner_call(STATE.util_funcs.swap_exc_state)
    #         .else_()
    #             .mov(tstate.exc_type,R_RET)
    #             .mov(tstate.exc_value,R_PRES2)
    #             .mov(tstate.exc_traceback,R_SCRATCH2)
    #
    #             .if_(R_RET).incref(R_RET).endif()
    #             .if_(R_PRES2).incref(R_PRES2).endif()
    #             .if_(R_SCRATCH2).incref(R_SCRATCH2).endif()
    #
    #             .mov(f_obj.f_exc_type,R_SCRATCH1)
    #             .mov(R_RET,f_obj.f_exc_type)
    #             .mov(f_obj.f_exc_value,R_RET)
    #             .mov(R_PRES2,f_obj.f_exc_value)
    #             .mov(f_obj.f_exc_traceback,R_PRES2)
    #             .mov(R_SCRATCH2,f_obj.f_exc_traceback)
    #
    #             .if_(R_SCRATCH1).decref(R_SCRATCH1,preserve_reg=R_RET).endif()
    #             .if_(R_RET).decref(R_RET).endif()
    #             .if_(R_PRES2).decref(R_PRES2).endif()
    #         .endif()
    #         .mov(f_obj.f_lasti,dword(R_SCRATCH1))
    #         .if_(dword(R_SCRATCH1) != -1))
    #
    #     rip = getattr(abi,'rip',None)
    #     if rip is not None:
    #         (ec.code
    #             .lea(SIndirect(0,rip),R_RET)
    #             (STATE.yield_start))
    #     else:
    #         (ec.code
    #             .call(abi.Displacement(0))
    #             (STATE.yield_start)
    #             .pop(R_RET))
    #
    #     (ec.code
    #             .add(R_RET,R_SCRATCH1)
    #         .jmp(R_SCRATCH1)
    #         .endif()
    #         .cmpl(0,STATE.throw_flag_store)
    #         .if_cond(TEST_NE)
    #             .mov(0,ret_temp_store)
    #             .jmp(fast_end)
    #         .endif())

    #if move_throw_flag:
        # use the original address for temp_store again
        #state.temp_store = state.throw_flag_store

    ec.allocate_locals(args)

    if args:
        ec.handle_args(args,func_self,func_args,func_kwds)

    for n in scope_ast.body:
        ec.visit(n)

    (ec.code
        # if this is jumped over, it will be elided in a subsequent pass
        .mov(ec.get_const(None),ec.var_return)
        .incref(ec.var_return)

        (ec.func_end)
        .comment('epilogue')
        (ec.deallocate_locals))

    names = [None] * len(ec.names)
    for n,c in ec.names.items():
        assert names[c[1]] is None
        names[c[1]] = n

    r = PyFunction(
        fb_obj,
        ProtoFunction(name,DelayedCompile.process(cgen,ec.code.code)),
        cgen,
        ec.var_return,
        ordered_consts(ec.names),
        args,
        tuple(ec.free_var_indexes.keys()),
        0,
        ordered_consts(ec.consts))

    entry_points.append(r)
    return r


def compile_utility_funcs_raw(abi):
    functions = [
        global_name_func(abi),
        local_name_func(abi),
        resume_generator_func(abi)]

    offset = 0
    for func in functions:
        func.offset = offset
        offset += len(func)

    return CompilationUnit(functions)


def compile_raw(code,abi,u_funcs) -> Tuple[CompilationUnit,List[PyFunction]]:
    entry_points = [] # type: List[PyFunction]
    mod_ast,sym_map = ast_and_symtable_map(code,'<string>','exec')
    compile_eval(mod_ast,sym_map,abi,u_funcs,entry_points,True)

    functions = []

    for pyfunc in entry_points:
        name,fcode = pyfunc.func
        pyfunc.func = pyfunc.cgen.compile(name,fcode,pyfunc.ret_var)
        functions.insert(0,pyfunc.func)

    offset = 0
    for func in functions:
        func.offset = offset
        offset += len(func)

    return CompilationUnit(functions),entry_points


def native_abi(*,assembly=False):
    if pyinternals.ARCHITECTURE in ('X86','X86_64'):
        x86ops = importlib.import_module('.x86_ops',__package__)
        if pyinternals.ARCHITECTURE == "X86":
            return x86ops.CdeclAbi(assembly=assembly)
        if sys.platform in ('win32','cygwin'):
            return x86ops.MicrosoftX64Abi(assembly=assembly)

        return x86ops.SystemVAbi(assembly=assembly)

    raise ValueError("native compilation is not supported on this CPU")


def compile_utility_funcs(abi):
    global utility_funcs
    if not utility_funcs:
        utility_cu = compile_utility_funcs_raw(abi)

        if debug.GDB_JIT_SUPPORT and not DISABLE_DEBUG:
            out = debug.generate(abi,utility_cu,utility_cu.functions)
            if DUMP_OBJ_FILE:
                with open('OBJ_UTILITY_DUMP_{}'.format(os.getpid()),'wb') as f:
                    f.write(out.buff.getbuffer())
        else:
            out = [f.code for f in utility_cu.functions]

        compiled = pyinternals.CompiledCode(out)
        utility_funcs = {f.name:UtilityFunction(f.name,compiled,f.offset) for f in utility_cu.functions}
        pyinternals.set_utility_funcs(utility_funcs)
    return utility_funcs


def compile(code : str,globals=None) -> pyinternals.Function:
    """Compile "code" and return a function taking no arguments.

    :param code: A string containing Python code to compile
    :param globals: A mapping to use as the "globals" object

    """
    global DUMP_OBJ_FILE

    abi = native_abi()
    u_funcs = compile_utility_funcs(abi)
    cu,entry_points = compile_raw(code,abi,u_funcs)

    if debug.GDB_JIT_SUPPORT and not DISABLE_DEBUG:
        out = debug.generate(abi,cu,(e.func for e in entry_points))
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


def compile_asm(code : str) -> str:
    """Compile "code" and return the assembly representation.

    This is for debugging purposes and is not necessarily usable for
    assembling. In particular, certain instructions in 64-bit mode have
    different names from their 32-bit equivalents, despite having the same
    binary representation, but the assembly returned by this function always
    uses the 32-bit names (since using the 32-bit versions of those
    instructions in 64-bit mode requires using a special prefix, which this
    library doesn't even have support for, there is no ambiguity).

    """
    class DummyUFunc:
        addr = 0

    abi = native_abi(assembly=True)

    parts = []
    u_funcs = compile_utility_funcs_raw(abi).functions
    for f in chain(compile_raw(code,abi,{f.name:DummyUFunc for f in u_funcs})[0].functions,u_funcs):
        if f.name: parts.append(f.name+':')
        parts.append(f.code.emit())

    return '\n\n'.join(parts)
