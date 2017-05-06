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


import collections
from typing import DefaultDict,Dict,Iterable,List,NamedTuple,Optional,Sequence,Set,Union

from . import compile_mod
from . import c_types
from . import pyinternals
from .intermediate import *

class _Indent:
    def __init__(self,amount):
        self.amount = amount
indent = _Indent(1)
dedent = _Indent(-1)

IRCOp = Union[str,_Indent,Target]
IRCCode = List[IRCOp]


INTERNAL_FUNCS = [
    'new_function',
    'new_generator',
    'free_pyobj_array',
    'missing_arguments',
    'too_many_positional',
    'excess_keyword',
    'append_tuple_for_call',
    'append_dict_for_call',
    'prepare_exc_handler',
    'end_exc_handler',
    'format_exc_check_arg',
    'format_exc_unbound',
    '_unpack_iterable',
    '_exception_cmp',
    '_do_raise',
    'import_all_from',
    'special_lookup',
    'call_exc_trace',
    '_print_expr',
    '_load_build_class',
    'c_global_name',
    'c_local_name']

STR_CONSTS = [
    'NAME_ERROR_MSG',
    'GLOBAL_NAME_ERROR_MSG',
    'UNBOUNDLOCAL_ERROR_MSG',
    'UNBOUNDFREE_ERROR_MSG',
    'NO_LOCALS_LOAD_MSG',
    'NO_LOCALS_STORE_MSG',
    'NO_LOCALS_DELETE_MSG',
    'BUILD_CLASS_ERROR_MSG',
    'CANNOT_IMPORT_MSG',
    'IMPORT_NOT_FOUND_MSG',
    'BAD_EXCEPTION_MSG',
    'UNEXPECTED_KW_ARG_MSG',
    'DUPLICATE_VAL_MSG']

OBJECTS = ['build_class']

BY_VALUE = {
    'PyDict_Type',
    'PyList_Type',
    'PyTuple_Type',
    'PyUnicode_Type',
    'PyMethod_Type',
    'PyGen_Type'}

def ptr_expr(x):
    if x.ptr_factor:
        r = ['(sizeof(char*)']
        if x.ptr_factor != 1:
            r.append(' * ')
            r.append(str(x.ptr_factor))

        if x.val:
            r.append(' + ')
            r.append(str(x.val))

        r.append(')')
        return ''.join(r)

    return str(x.val)

class UniqueNameGen:
    def __init__(self) -> None:
        self.used_names = collections.defaultdict(lambda: 0)  # type: DefaultDict[str,int]

    def __call__(self,base: str,always_number: bool=False) -> str:
        count = self.used_names[base] + 1
        self.used_names[base] = count
        return base if count == 1 and not always_number else base + str(count)

class CFunctionBody:
    name = None

CCodeImpl = NamedTuple('CCodeImpl',[('c_name',str),('code',str)])

class COpGen(OpGen[IRCCode]):
    def __init__(self,abi: 'CAbi',func_args: Iterable[Var] = (),callconv : CallConvType=CallConvType.default,pyinfo: Optional[PyFuncInfo]=None) -> None:
        super().__init__(abi,func_args,callconv,pyinfo)

        self.vars = {} # type: Dict[Union[Var,Block],str]
        self.targets = {} # type: Dict[Target,str]
        self.unique_names = UniqueNameGen()

        self.case_labels = set() # type: Set[Target]
        self.final_case_labels = set() # type: Set[Target]

        self.next_func_args = [] # type: List[Var]

        # the function parameters get the first names
        for v in self.func_arg_vars:
            self.arg(v)

        # the name of the C function
        self.c_name = abi.name_gen('_func_impl')

    def _indrect_var(self,a: IndirectVar) -> str:
        if a.base and a.offset.ptr_factor == 0:
            b_type = c_types.real_type(a.base.data_type)
            if isinstance(b_type,c_types.TPtr):
                t = c_types.real_type(b_type.base)
                if isinstance(t,c_types.TStruct):
                    a_offset = a.offset.val
                    for attr in t.attrs:
                        if a_offset == attr.offset:
                            r = '{}->{}'.format(self.arg(a.base),attr.name)

                            if a.index:
                                a_type = c_types.real_type(a.data_type)
                                if isinstance(a_type,c_types.TPtr) and a_type.base.size(self.abi) == a.scale:
                                    r = '{}[{}]'.format(r,self.arg(a.index))
                                else:
                                    r = '((char*){} + (char*){}{})'.format(
                                        r,
                                        self.arg(a.index),
                                        '' if a.scale == 1 else ' * '+ptr_expr(a.scale))

                            return r

        parts = []  # type: List[str]
        if a.base:
            parts.append('((char*){})'.format(self.arg(a.base)))
        if a.index:
            part = '((long){})'.format(self.arg(a.index))
            if a.scale != 1:
                part += ' * ' + ptr_expr(a.scale)
            parts.append(part)
        if a.offset or not parts:
            parts.append(ptr_expr(a.offset))
        return '*({})({})'.format(c_types.TPtr(a.data_type),' + '.join(parts))

    def arg(self,a: Union[Value,Target]) -> str:
        if isinstance(a,Target):
            name = self.targets.get(a)
            if name is None:
                self.targets[a] = name = self.unique_names('target',True)
            return name

        if isinstance(a,VarPart):
            return '{}[{}]'.format(self.arg(a.block),a.offset)

        if isinstance(a,(Var,Block)):
            name = self.vars.get(a)
            if name is None:
                if a.dbg_symbol:
                    name = self.unique_names(a.dbg_symbol,False)
                else:
                    name = self.unique_names('var',True)
                self.vars[a] = name
            return name

        if isinstance(a,IndirectVar):
            return self._indrect_var(a)

        if isinstance(a,Symbol):
            if a.name in BY_VALUE:
                return '&'+a.name
            return a.name

        if isinstance(a,PyConst):
            if self.pyinfo is None:
                raise ValueError('Python-specific information was not supplied to this class')
            return 'PyTuple_GET_ITEM({}->body->{},{})'.format(self.arg(self.pyinfo.func_var),a.tuple_name,a.index)

        if isinstance(a,Immediate):
            return ptr_expr(a.val)

        return ''

    def bin_op(self,a: Value,b: Value,dest: MutableValue,op_type: OpType) -> IRCCode:
        op = {
            OpType.add : '+',
            OpType.sub : '-',
            OpType.mul : '*',
            OpType.div : '/',
            OpType.and_ : '&',
            OpType.or_ : '|',
            OpType.xor : '^'}[op_type]

        return ['{} = {} {} {};'.format(self.arg(dest),self.arg(a),op,self.arg(b))]

    def unary_op(self,a: Value,dest: MutableValue,op_type: UnaryOpType) -> IRCCode:
        op = {
            UnaryOpType.neg: '-'}[op_type]

        return ['{} = {}{};'.format(self.arg(dest),op,self.arg(a))]

    def load_addr(self,addr: MutableValue,dest: MutableValue) -> IRCCode:
        return ['{} = {}{};'.format(
            self.arg(dest),
            '' if isinstance(addr,Block) else '&',
            self.arg(addr))]

    def call(self,func,args: Sequence[Value]=(),store_ret: Optional[Var]=None,callconv: CallConvType=CallConvType.default) -> IRCCode:
        store_str = ''
        if store_ret is not None:
            store_str = self.arg(store_ret) + ' = '
        return ['{}{}({});'.format(store_str,self.arg(func),','.join(self.arg(a) for a in args))]

    def call_preloaded(self,func,args: int,store_ret: Optional[Var]=None,callconv: CallConvType=CallConvType.default) -> IRCCode:
        if args != len(self.next_func_args):
            raise ValueError('"args" must equal the number of arguments referenced via "gen_func_args"')

        r = self.call(func,self.next_func_args,store_ret,callconv)
        del self.next_func_args[:]
        return r

    def get_func_arg(self,i: int,prev_frame: bool=False,callconv: Optional[CallConvType]=None) -> MutableValue:
        if prev_frame:
            return self.func_arg_vars[i]

        while i >= len(self.next_func_args): self.next_func_args.append(Var())

        return self.next_func_args[i]

    def jump(self,dest: Union[Target,Value],targets: Union[Iterable[Target],Target,None]=None) -> IRCCode:
        return ['goto {}{};'.format(
            '*' if isinstance(dest,Value) else '',
            self.arg(dest))]

    def _cond(self,cond: Cmp) -> str:
        if isinstance(cond,AndCmp):
            return '({}) && ({})'.format(self._cond(cond.a),self._cond(cond.b))

        if isinstance(cond,OrCmp):
            return '({}) || ({})'.format(self._cond(cond.a),self._cond(cond.b))

        assert isinstance(cond,BinCmp)
        op = {
            CmpType.eq : '==',
            CmpType.ne : '!=',
            CmpType.ge : '>=',
            CmpType.gt : '>',
            CmpType.le : '<=',
            CmpType.lt : '<'}[cond.op]
        convert = 'long' if cond.signed else 'unsigned long'
        return '({0}){1} {2} ({0}){3}'.format(convert,self.arg(cond.a),op,self.arg(cond.b))

    def jump_if(self,dest: Target,cond: Cmp) -> IRCCode:
        return ['if({}) goto {};'.format(self._cond(cond),self.arg(dest))]

    def if_(self,cond: Cmp,on_true: IRCCode,on_false: Optional[IRCCode]) -> IRCCode:
        r = ['if({}) {{'.format(self._cond(cond)),indent] + on_true
        if on_false is not None:
            r.append(dedent)
            r.append('} else {')
            r.append(indent)
            r.extend(on_false)

        r.append(dedent)
        r.append('}')
        return r

    def do_while(self,action: IRCCode,cond: Cmp) -> IRCCode:
        return ['do {',indent] + action + [dedent,'}} while({})'.format(self._cond(cond))]

    def jump_table(self,val: Value,targets: Sequence[Target]) -> IRCCode:
        return ['goto *(((void*[]){{{}}})[{}]);'.format(','.join('&&'+self.arg(t) for t in targets),self.arg(val))]

    def move(self,src: Value,dest: MutableValue) -> IRCCode:
        return ['{} = {};'.format(self.arg(dest),self.arg(src))]

    def shift(self,src: Value,shift_dir: ShiftDir,amount: Value,dest: MutableValue) -> IRCCode:
        return ['{} = {} {} {};'.format(
            self.arg(dest),
            self.arg(src),
            '<<' if shift_dir == ShiftDir.left else '>>',
            self.arg(amount))]

    def enter_finally(self,f: FinallyTarget,next_t: Optional[Target] = None) -> IRCCode:
        t = Target() if next_t is None else next_t
        r = ['{} = &&{};'.format(self.arg(f.next_var),self.arg(t))] + self.jump(f.start) # type: IRCCode
        if next_t is None:
            r.append(t)
            f.add_next_target(t)
        return r

    def finally_body(self,f: FinallyTarget,body: IRCCode) -> IRCCode:
        return [f.start] + body + self.jump(f.next_var,f.next_targets)

    def compile(self,func_name: str,code: IRCCode,ret_var: Optional[Var]=None,end_targets=()):
        lines = ['{} {}({}){{'.format(
            'void' if ret_var is None else str(ret_var.data_type),
            self.c_name,
            ','.join(v.data_type.typestr(self.arg(v)) for v in self.func_arg_vars))]

        non_arg_vars = set(self.vars.keys()).difference(self.func_arg_vars)
        for var,name in sorted(((v,self.vars[v]) for v in non_arg_vars),key=(lambda vn: vn[1])):
            lines.append('    {};'.format(var.data_type.typestr(name)))

        lines.append('')

        indent = 1

        for op in code:
            assert isinstance(op,(str,_Indent,Target,IndirectMod))

            if isinstance(op,str):
                lines.append('    '*indent + op)
            elif isinstance(op,_Indent):
                indent += op.amount
                assert indent >= 1
            elif isinstance(op,Target):
                ind_str = '    '*(indent-1)
                frmt = '{}case {}:' if op in self.case_labels else '{}{}:;'
                lines.append(frmt.format(ind_str,self.arg(op)))

        lines.append('    return {};'.format(self.arg(ret_var)))
        lines.append('}')

        return Function(
            CCodeImpl(self.c_name,'\n'.join(lines)),
            name=func_name,
            returns=c_types.t_void if ret_var is None else ret_var.data_type,
            params=[Param(self.arg(v),v.data_type) for v in self.func_arg_vars],
            callconv=self.callconv)

    def new_func_body(self):
        return CFunctionBody()


class CAbi(Abi):
    code_gen = COpGen
    name_gen = UniqueNameGen()

def compile_utility_funcs() -> str:
    abi = CAbi()
    utility_cu = compile_mod.compile_utility_funcs_raw(abi)

    return '\n\n'.join(f.code.code for f in utility_cu.functions)

class UtilityPseudoFunc:
    def __init__(self,name):
        self.name = name

    @property
    def addr(self):
        return Symbol(self.name)

def init_external_ref(init_lines,name,ptr_name,ptr_type):
    init_lines.append(
        'if(!(tmp = PyDict_GetItemString(addrs,"{}"))) {{'.format(name))
    init_lines.append(
        '    PyErr_SetString(PyExc_KeyError,"{}");'.format(name))
    init_lines.append('    goto init_end;')
    init_lines.append('}')
    init_lines.append('tmp_l = PyLong_AsUnsignedLong(tmp);')
    init_lines.append('if(PyErr_Occurred()) goto init_end;')
    init_lines.append('{} = ({})tmp_l;'.format(ptr_name,ptr_type))

def external_refs():
    refs = []
    init_lines = []
    for intern_f in INTERNAL_FUNCS:
        sig = c_types.func_signatures[intern_f]
        ptr_type = c_types.TPtr(sig)
        ptr_name = '_ptr_'+intern_f

        param_names = ['_'+str(i) for i in range(len(sig.params))]

        call = '(*{})'.format(ptr_name)
        if sig.returns != c_types.t_void:
            call = 'return' + call

        refs.append('static {};\nstatic {} {}({}) {{ {}({}); }}'.format(
            ptr_type.typestr(ptr_name),
            sig.returns,
            intern_f,
            ','.join(p.typestr(n) for p,n in zip(sig.params,param_names)),
            call,
            ','.join(param_names)
        ))

        init_external_ref(init_lines,intern_f,ptr_name,ptr_type)

    for const in STR_CONSTS:
        refs.append('static const char *{};'.format(const))
        init_external_ref(init_lines,const,const,c_types.const_char_ptr)

    return refs,init_lines

def obj_inits():
    for i,obj in enumerate(OBJECTS):
        yield 'if(!(other_objs[{}] = PyObject_GetAttrString(intern_mod,"{}"))) goto err;'.format(i,obj)

def format_and_literal(val):
    if isinstance(val,str):
        return 's','"{}"'.format(val.encode('raw_unicode_escape').decode('latin_1'))
    if isinstance(val,int):
        if isinstance(val,bool):
            val = int(val)
        if -0x7fffffffffffffff <= val <= 0xffffffffffffffff:
            if not (-0x7fffffff <= val <= 0xffffffff):
                return ('L' if val < 0 else 'K'),str(val) + 'LL'
            return ('l' if val < 0 else 'k'),str(val)

        return 'O&','int_converter,"{}"'.format(val)
    if isinstance(val,float):
        return 'd',str(val)
    if isinstance(val,CFunctionBody):
        assert val.name is not None
        return 'O',val.name
    if isinstance(val,CCodeImpl):
        return 'k','(unsigned long)'+val.c_name
    if val is None:
        return 'O','Py_None'
    if val is ...:
        return 'O','Py_Ellipsis'
    if val is pyinternals.build_class:
        return 'O','other_objs[0]'

    raise TypeError('unexpected constant type: '+val.__class__.__name__)

def compound_format_and_literal(*values):
    format = []
    args = []

    def handle_tuple(values):
        for v in values:
            if type(v) in (list,tuple):
                format.append('(')
                handle_tuple(v)
                format.append(')')
            else:
                f,a = format_and_literal(v)
                format.append(f)
                args.append(a)
    handle_tuple(values)

    return ''.join(format),','.join(args)

def func_bodies(entry_points):
    for i,e in enumerate(entry_points):
        args = 0
        vararg = False
        kwargs = 0
        varkw = False
        if e.args:
            args = len(e.args.args)
            vararg = e.args.vararg
            kwargs = len(e.args.kwonlyargs)
            varkw = e.args.kwarg

        yield "if(!(init_fb = PyObject_GetAttr(bodies[{}],init_str))) goto err;".format(i)
        yield 'tmp = PyObject_CallFunction(init_fb,"{}",{});'.format(
            *compound_format_and_literal(
                None,
                e.func.code,
                e.name,
                e.names,
                args,
                vararg,
                kwargs,
                varkw,
                e.free_names,
                e.cells,
                e.consts))
        yield 'Py_DECREF(init_fb);'
        yield 'if(!tmp) goto err;'
        yield 'Py_DECREF(tmp);'


def compile(code: str,modname: str='module') -> str:
    u_funcs = {}
    for f in ['global_name','local_name','resume_generator']:
        u_funcs[f] = UtilityPseudoFunc('c_' + f)
    cu,entry_points = compile_mod.compile_raw(code,CAbi(),u_funcs)

    for i,e in enumerate(entry_points):
        e.fb_obj.name = 'bodies[{}]'.format(i)

    w_funcs,init_lines = external_refs()

    r = [
        '#include <Python.h>\n' +
        '#include <frameobject.h>',
        '#include "nativecompile/pyinternals.h"',
        'static PyObject *intern_mod = NULL;',
        'static PyObject *int_converter(const char *x) { return PyLong_FromString(x,NULL,10); }']
    r.extend(w_funcs)
    r.extend(f.code.code for f in cu.functions)
    r.append("""
static int mod_traverse(PyObject *m,visitproc visit,void *arg) {{
    Py_VISIT(intern_mod);
    return 0;
}}

static int mod_clear(PyObject *m) {{
    Py_CLEAR(intern_mod);
    return 0;
}}

static void mod_free(void *m) {{
    Py_XDECREF(intern_mod);
}}

static PyModuleDef module = {{
    PyModuleDef_HEAD_INIT,
    "{modname}",
    NULL,
    0,
    NULL,
    NULL,
    mod_traverse,
    mod_clear,
    mod_free
}};

PyMODINIT_FUNC PyInit_{modname}(void) {{
    int i;
    PyObject *mod, *tmp, *init_str, *init_fb, *new_fb=NULL, *fb_type=NULL, *new_args_tup=NULL, *f_type=NULL;
    PyObject *other_objs[{obj_count}] = {{NULL}};
    PyObject *bodies[{body_count}] = {{NULL}};

    if(intern_mod) Py_INCREF(intern_mod);
    else {{
        unsigned long tmp_l;
        PyObject *addrs;

        if(!(intern_mod = PyImport_ImportModule("nativecompile.pyinternals"))) return NULL;

        addrs = PyObject_GetAttrString(intern_mod,"raw_addresses");

        if(!addrs) {{
            Py_DECREF(intern_mod);
            return NULL;
        }}

        {initcode}

    init_end:
        Py_DECREF(addrs);

        if(PyErr_Occurred()) {{
            Py_DECREF(intern_mod);
            return NULL;
        }}
    }}

    if(!(mod = PyModule_Create(&module))) return NULL;

    tmp = PyEval_GetBuiltins();
    Py_INCREF(tmp);
    if(PyModule_AddObject(mod,"__builtins__",tmp)) goto err;

    if(!(init_str = PyUnicode_FromString("__init__"))) goto err;
    if(!(fb_type = PyObject_GetAttrString(intern_mod,"FunctionBody"))) goto err;
    if(!(new_fb = PyObject_GetAttrString(fb_type,"__new__"))) goto err;
    if(!(new_args_tup = PyTuple_Pack(1,fb_type))) goto err;

    for(i=0;i<{body_count};++i) {{
        if(!(bodies[i] = PyObject_Call(new_fb,new_args_tup,NULL))) goto err;
    }}

    {obj_init}
    {fb_init}

    if(!(f_type = PyObject_GetAttrString(intern_mod,"Function"))) goto err;
    if(!(tmp = PyObject_CallFunction(f_type,"OsO",bodies[{last_body}],"__main_func",PyModule_GetDict(mod)))) goto err;

    if(PyModule_AddObject(mod,"__main_func",tmp)) goto err;
end:
    Py_XDECREF(f_type);
    for(i=0;i<{obj_count};++i) Py_XDECREF(other_objs[i]);
    for(i=0;i<{body_count};++i) Py_XDECREF(bodies[i]);
    Py_XDECREF(new_args_tup);
    Py_XDECREF(new_fb);
    Py_XDECREF(fb_type);
    Py_XDECREF(init_str);

    return mod;
err:
    Py_DECREF(mod);
    mod = NULL;
    goto end;
}}""".format(
        modname=modname,
        obj_count=len(OBJECTS),
        body_count=len(entry_points),
        initcode='\n        '.join(init_lines),
        obj_init='\n    '.join(obj_inits()),
        fb_init='\n    '.join(func_bodies(entry_points)),
        last_body=len(entry_points)-1))

    return '\n\n'.join(r)
