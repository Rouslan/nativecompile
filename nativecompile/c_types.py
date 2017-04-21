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


from . import dwarf
from . import pyinternals

RefType = dwarf.FORM_ref4

class CType:
    def die(self,dcu,abi,st,cache):
        if cache is None: cache = {}

        r = cache.get(self)
        if r is None:
            r = self._die(dcu,abi,st,cache)
            cache[self] = r
            dcu.children.append(r)
        return r

    def _die(self,dcu,abi,st,cache):
        raise NotImplementedError()

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.typestr('')

    def size(self,abi):
        raise NotImplementedError()

    def typestr(self,deriv):
        raise NotImplementedError()

def append_deriv(x,deriv):
    return ' '.join((x,deriv)) if deriv else x

# noinspection PyAbstractClass
class BasicType(CType):
    def full_name(self):
        raise NotImplementedError()

    def typestr(self,deriv):
        return append_deriv(self.full_name(),deriv)

class TVoid(BasicType):
    def full_name(self):
        return 'void'

    def __repr__(self):
        return 'TVoid()'

    def __eq__(self,b):
        return isinstance(b,TVoid)

    __hash__ = CType.__hash__

    def _die(self,dcu,abi,st,cache):
        return dwarf.DIE('unspecified_type',name=st['void'])

    def size(self,abi):
        raise TypeError('"void" does not have a size')

t_void = TVoid()

class TInt(BasicType):
    def __init__(self,base_name,f_size,signed):
        self.base_name = base_name
        self.f_size = f_size
        self.signed = signed

    def full_name(self):
        r = self.base_name
        if not self.signed: r = 'unsigned ' + r
        return r

    def __repr__(self):
        return 'TInt({!r},{!r},{!r})'.format(self.base_name,self.f_size,self.signed)

    def __eq__(self,b):
        return isinstance(b,TInt) and self.base_name == b.base_name and self.signed == b.signed

    __hash__ = CType.__hash__

    def _die(self,dcu,abi,st,cache):
        return dwarf.DIE('base_type',
            name=st[self.__str__()],
            encoding=dwarf.ATE.signed if self.signed else dwarf.ATE.unsigned,
            byte_size=dwarf.FORM_data1(self.f_size(abi)))

    def size(self,abi):
        return self.f_size(abi)

t_int = TInt('int',(lambda abi: abi.int_size),True)
t_uint = TInt(t_int.base_name,t_int.f_size,False)
t_long = TInt('long',(lambda abi: abi.long_size),True)
t_ulong = TInt(t_long.base_name,t_long.f_size,False)

class TFixedSizeBase(BasicType):
    def __init__(self,name,size,signed,d_encoding):
        self.name = name
        self._size = size
        self.signed = signed
        self.d_encoding = d_encoding

    def full_name(self):
        return self.name

    def __repr__(self):
        return 'TFixedSizeBase({!r},{},{!r},{!r})'.format(self.name,self.size,self.signed,self.d_encoding)

    def __eq__(self,b):
        return isinstance(b,TFixedSizeBase) and self.name == b.base

    __hash__ = CType.__hash__

    def _die(self,dcu,abi,st,cache):
        return dwarf.DIE('base_type',
            name=st[self.name],
            encoding=self.d_encoding,
            byte_size=dwarf.FORM_data1(self._size))

    def size(self,abi):
        return self._size

t_char = TFixedSizeBase('char',1,True,dwarf.ATE.signed_char)
t_schar = TFixedSizeBase('signed char',1,True,dwarf.ATE.signed_char)
t_uchar = TFixedSizeBase('unsigned char',1,False,dwarf.ATE.unsigned_char)

class TConst(CType):
    def __init__(self,base):
        assert not isinstance(base,TConst)
        self.base = base

    def typestr(self,deriv):
        return self.base.typestr(append_deriv('const',deriv))

    def __repr__(self):
        return 'TConst({!r})'.format(self.base)

    def __eq__(self,b):
        return isinstance(b,TPtr) and self.base == b.base

    __hash__ = CType.__hash__

    def _die(self,dcu,abi,st,cache):
        return dwarf.DIE('const_type',type=RefType(self.base.die(dcu,abi,st,cache)))

    def size(self,abi):
        return self.base.size(abi)

class TPtr(CType):
    def __init__(self,base):
        self.base = base

    def typestr(self,deriv):
        return self.base.typestr('*'+deriv)

    def __repr__(self):
        return 'TPtr({!r})'.format(self.base)

    def __eq__(self,b):
        return isinstance(b,TPtr) and self.base == b.base

    __hash__ = CType.__hash__

    def _die(self,dcu,abi,st,cache):
        return dwarf.DIE('pointer_type',
            type=RefType(self.base.die(dcu,abi,st,cache)),
            byte_size=dwarf.FORM_data1(abi.ptr_size))

    def size(self,abi):
        return abi.ptr_size

class TArray(CType):
    def __init__(self,base,length):
        assert length > 0
        self.base = base
        self.length = length

    def typestr(self,deriv):
        part = '[{}]'.format(self.length)
        return self.base.typestr('({}){}'.format(deriv,part) if deriv else part)

    def __repr__(self):
        return 'TArray({!r},{})'.format(self.base,self.length)

    def __eq__(self,b):
        return isinstance(b,TArray) and self.base == b.base and self.length == b.length

    __hash__ = CType.__hash__

    def _die(self,dcu,abi,st,cache):
        r = dwarf.DIE('array_type',
            type=RefType(self.base.die(dcu,abi,st,cache)))
        r.children.append(dwarf.DIE('subrange_type',
            type=RefType(t_ulong.die(dcu,abi,st,cache)),
            upper_bound=dwarf.smallest_data_form(self.length-1)))
        return r

    def size(self,abi):
        return self.base.size(abi) * self.length


class TFunc(CType):
    def __init__(self,params,returns):
        self.params = params
        self.returns = returns

    def typestr(self,deriv):
        return '{} ({})({})'.format(str(self.returns),deriv,','.join(map(str,self.params)))

    def size(self,abi):
        raise TypeError("it does not make sense to ask a function's size")

    # This method throws an instance of ValueError and not NotImplementedError
    # because it is not an abstract method. It simply hasn't been written, but
    # it may be useful in the future to implement it.
    def _die(self,dcu,abi,st,cache):
        raise ValueError('not implemented')

class TStruct(BasicType):
    def __init__(self,name,members):
        self.name = name
        self.members = members

    def full_name(self):
        if self.name:
            return 'struct ' + self.name

        return '<anonymous struct>'

    # NOTE: for now, this just mimics void, since we don't really need a full
    # implementation
    def _die(self,dcu,abi,st,cache):
        return t_void.die(dcu,abi,st,cache)

    def size(self,abi):
        raise ValueError('not implemented')

class TTypedef(BasicType):
    def __init__(self,name,base):
        self.name = name
        self.base = base

    def full_name(self):
        return self.name

    def size(self,abi):
        return self.base.size(abi)

    def _die(self,dcu,abi,st,cache):
        return dwarf.DIE('typedef',
            name=st[self.name],
            type=RefType(self.base.die(dcu,abi,st,cache)))

def real_type(x):
    while isinstance(x,TTypedef):
        x = x.base
    return x

def real_isinstance(x,t):
    return isinstance(real_type(x),t)

def typedef_struct(typedef,name=None):
    return TTypedef(typedef,TStruct(name,pyinternals.member_offsets[typedef]))

PyObject = typedef_struct('PyObject','_object')
PyVarObject = typedef_struct('PyVarObject')
PyTypeObject = typedef_struct('PyTypeObject','_typeobject')
PyListObject = typedef_struct('PyListObject')
PyTupleObject = typedef_struct('PyTupleObject')
PyFrameObject = typedef_struct('PyFrameObject','_frame')
PyThreadState = typedef_struct('PyThreadState','_ts')
PyCellObject = typedef_struct('PyCellObject')
PyMethodObject = typedef_struct('PyMethodObject')
CompiledCode = typedef_struct('CompiledCode','_CompiledCode')
FunctionBody = typedef_struct('FunctionBody')
Function = typedef_struct('Function')
Generator = typedef_struct('Generator')
PyDictObject = TTypedef('PyDictObject',TStruct(None,{}))
PyCodeObject = TTypedef('PyCodeObject',TStruct(None,{}))

t_void_ptr = TPtr(t_void)

PyObject_ptr = TPtr(PyObject)
PyTypeObject_ptr = TPtr(PyTypeObject)
Py_tracefunc = t_void_ptr
PyThreadState_ptr = TPtr(PyThreadState)
PyFrameObject_ptr = TPtr(PyFrameObject)
PyCodeObject_ptr = TPtr(PyCodeObject)
PyDictObject_ptr = TPtr(PyDictObject)
const_char_ptr = TPtr(TConst(t_char))
Function_ptr = TPtr(Function)
FunctionBody_ptr = TPtr(FunctionBody)
Py_ssize_t = t_long
size_t = t_ulong

func_signatures = {
    'PyMem_Malloc' : TFunc([size_t],t_void_ptr),
    'Py_IncRef' : TFunc([PyObject_ptr],t_void),
    'Py_DecRef' : TFunc([PyObject_ptr],t_void),
    'Py_AddPendingCall' : TFunc([TPtr(TFunc([t_void_ptr],t_int)),t_void_ptr],t_int),
    'PyDict_GetItem' : TFunc([PyObject_ptr,PyObject_ptr],PyObject_ptr),
    'PyDict_SetItem' : TFunc([PyObject_ptr,PyObject_ptr,PyObject_ptr],t_int),
    'PyDict_DelItem' : TFunc([PyObject_ptr,PyObject_ptr],t_int),
    'PyDict_GetItemString' : TFunc([PyObject_ptr,const_char_ptr],PyObject_ptr),
    'PyDict_Size' : TFunc([PyObject_ptr],Py_ssize_t),
    'PyDict_Copy' : TFunc([PyObject_ptr],PyObject_ptr),
    'PyDict_New' : TFunc([],PyObject_ptr),
    '_PyDict_NewPresized' : TFunc([Py_ssize_t],PyObject_ptr),
    '_PyDict_LoadGlobal' : TFunc([PyDictObject_ptr,PyDictObject_ptr,PyObject_ptr],PyObject_ptr),
    'PyObject_IsSubclass' : TFunc([PyObject_ptr,PyObject_ptr],t_int),
    'PyObject_GetItem' : TFunc([PyObject_ptr,PyObject_ptr],PyObject_ptr),
    'PyObject_SetItem' : TFunc([PyObject_ptr,PyObject_ptr,PyObject_ptr],t_int),
    'PyObject_DelItem' : TFunc([PyObject_ptr,PyObject_ptr],t_int),
    'PyObject_GetIter' : TFunc([PyObject_ptr],PyObject_ptr),
    'PyObject_GetAttr' : TFunc([PyObject_ptr,PyObject_ptr],PyObject_ptr),
    'PyObject_SetAttr' : TFunc([PyObject_ptr,PyObject_ptr,PyObject_ptr],t_int),
    'PyObject_IsTrue' : TFunc([PyObject_ptr],t_int),
    'PyObject_RichCompare' : TFunc([PyObject_ptr,PyObject_ptr,t_int],t_int),
    'PyObject_Call' : TFunc([PyObject_ptr,PyObject_ptr,PyObject_ptr],PyObject_ptr),
    'PyObject_CallObject' : TFunc([PyObject_ptr,PyObject_ptr],PyObject_ptr),
    'PyEval_GetGlobals' : TFunc([],t_void),
    'PyEval_GetBuiltins' : TFunc([],t_void),
    'PyEval_GetLocals' : TFunc([],t_void),
    'PyEval_AcquireThread' : TFunc([],t_void),
    '_PyEval_SignalAsyncExc' : TFunc([],t_void),
    'PyErr_Occurred' : TFunc([],t_void),
    'PyErr_ExceptionMatches' : TFunc([],t_void),
    'PyErr_Clear' : TFunc([],t_void),
    'PyErr_Format' : TFunc([],t_void),
    'PyErr_SetString' : TFunc([],t_void),
    'PyErr_Fetch' : TFunc([],t_void),
    'PyErr_Restore' : TFunc([],t_void),
    'PyErr_NormalizeException' : TFunc([],t_void),
    'PyException_SetTraceback' : TFunc([],t_void),
    'PyNumber_Multiply' : TFunc([],t_void),
    'PyNumber_TrueDivide' : TFunc([],t_void),
    'PyNumber_FloorDivide' : TFunc([],t_void),
    'PyNumber_Add' : TFunc([],t_void),
    'PyNumber_Subtract' : TFunc([],t_void),
    'PyNumber_Lshift' : TFunc([],t_void),
    'PyNumber_Rshift' : TFunc([],t_void),
    'PyNumber_And' : TFunc([],t_void),
    'PyNumber_Xor' : TFunc([],t_void),
    'PyNumber_Or' : TFunc([],t_void),
    'PyNumber_InPlaceMultiply' : TFunc([],t_void),
    'PyNumber_InPlaceTrueDivide' : TFunc([],t_void),
    'PyNumber_InPlaceFloorDivide' : TFunc([],t_void),
    'PyNumber_InPlaceRemainder' : TFunc([],t_void),
    'PyNumber_InPlaceAdd' : TFunc([],t_void),
    'PyNumber_InPlaceSubtract' : TFunc([],t_void),
    'PyNumber_InPlaceLshift' : TFunc([],t_void),
    'PyNumber_InPlaceRshift' : TFunc([],t_void),
    'PyNumber_InPlaceAnd' : TFunc([],t_void),
    'PyNumber_InPlaceXor' : TFunc([],t_void),
    'PyNumber_InPlaceOr' : TFunc([],t_void),
    'PyNumber_Positive' : TFunc([],t_void),
    'PyNumber_Negative' : TFunc([],t_void),
    'PyNumber_Invert' : TFunc([],t_void),
    'PyNumber_Remainder' : TFunc([],t_void),
    'PyNumber_Power' : TFunc([],t_void),
    'PyLong_AsLong' : TFunc([],t_void),
    'PyLong_FromLong' : TFunc([],t_void),
    'PyList_New' : TFunc([],t_void),
    'PyList_Append' : TFunc([],t_void),
    'PyTuple_New' : TFunc([],t_void),
    'PyTuple_Pack' : TFunc([],t_void),
    'PySet_Add' : TFunc([],t_void),
    'PySet_New' : TFunc([],t_void),
    'PySlice_New' : TFunc([],t_void),
    'PySequence_Contains' : TFunc([],t_void),
    'PyTraceBack_Here' : TFunc([],t_void),
    'PyUnicode_Format' : TFunc([],t_void),
    'PyUnicode_Append' : TFunc([],t_void),
    'PyUnicode_Concat' : TFunc([],t_void),
    'PyCell_Get' : TFunc([],t_void),
    'PyCell_Set' : TFunc([],t_void),
    'PyCell_New' : TFunc([],t_void),
    '_PyGen_FetchStopIterationValue' : TFunc([],t_void),

    'new_function' : TFunc([FunctionBody_ptr,PyObject_ptr,PyObject_ptr,PyObject_ptr,PyObject_ptr,TPtr(PyObject_ptr),TPtr(PyObject_ptr),PyObject_ptr],PyObject_ptr),
    'new_generator' : TFunc([PyFrameObject_ptr,Function_ptr,size_t],PyObject_ptr),
    'free_pyobj_array' : TFunc([TPtr(PyObject_ptr),Py_ssize_t],t_void),
    'missing_arguments' : TFunc([Function_ptr,TPtr(PyObject_ptr)],t_void),
    'too_many_positional' : TFunc([Function_ptr,t_long,PyObject_ptr],t_void),
    'excess_keyword' : TFunc([Function_ptr,PyObject_ptr],t_void),
    'append_tuple_for_call' : TFunc([PyObject_ptr,PyObject_ptr,PyObject_ptr],PyObject_ptr),
    'append_dict_for_call' : TFunc([PyObject_ptr,PyObject_ptr,PyObject_ptr],PyObject_ptr),
    'prepare_exc_handler' : TFunc([TPtr(PyObject_ptr)],t_void),
    'end_exc_handler' : TFunc([TPtr(PyObject_ptr)],t_void),
    'format_exc_check_arg' : TFunc([PyObject_ptr,const_char_ptr,PyObject_ptr],PyObject_ptr),
    'format_exc_unbound' : TFunc([PyCodeObject_ptr,t_int],PyObject_ptr),
    '_unpack_iterable' : TFunc([PyObject_ptr,t_int,t_int,TPtr(PyObject_ptr)],t_int),
    '_exception_cmp' : TFunc([PyObject_ptr,PyObject_ptr],PyObject_ptr),
    '_do_raise' : TFunc([PyObject_ptr,PyObject_ptr],PyObject_ptr),
    'import_all_from' : TFunc([PyObject_ptr,PyObject_ptr],t_int),
    'special_lookup' : TFunc([PyObject_ptr,PyObject_ptr],PyObject_ptr),
    'call_exc_trace' : TFunc([Py_tracefunc,PyObject_ptr,PyThreadState_ptr,PyFrameObject_ptr],t_void),
    '_print_expr' : TFunc([PyObject_ptr],t_int),
    '_load_build_class' : TFunc([PyObject_ptr],PyObject_ptr),
    'c_global_name' : TFunc([PyObject_ptr,PyFrameObject_ptr],PyObject_ptr),
    'c_local_name' : TFunc([PyObject_ptr,PyFrameObject_ptr],PyObject_ptr)
}