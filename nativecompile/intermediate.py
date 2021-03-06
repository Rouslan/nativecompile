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


"""Support for the intermediate instruction representation.

The intermediate representation is basically a series of functions that produce
machine code, and their arguments, except for a few differences: Instead of
registers and addresses pointing to stack space, values are stored in an
unlimited number of variables. Instead of numeric offsets, jump targets are
represented as unique objects (instances of 'Target'), acting like labels in
assembly.

Control-flow is represented via instances of 'IRJump', which must precede
instructions that may not be executed, either because the program counter may
be affected by the previous instruction or because the instructions are
executed conditionally.

"""

__all__ = ['SIZE_B','SIZE_W','SIZE_D','SIZE_Q','StackSection','LocationType',
    'PtrBinomial','SIZE_PTR','Target','Value','MutableValue','Var','Block',
    'VarPart','IndirectVar','Immediate','FixedRegister','ArgStackItem',
    'Symbol','PyConst','ensure_same_size','Instr','AddressType','ParamDir',
    'Overload','RegAllocatorOverloads','OpDescription','CommentDesc',
    'comment_desc','inline_comment_desc','InvalidateRegs','CreateVar','IRJump',
    'IndirectMod','LockRegs','UnlockRegs','IRAnnotation','annotate',
    'IRSymbolLocDescr','Instr2','CmpType','Cmp','BinCmp','AndCmp','OrCmp',
    'OpType','commutative_ops','UnaryOpType','ShiftDir','IROp','IRCode',
    'IROp2','IRCode2','ExtraState','CallConvType','PyFuncInfo','FinallyTarget',
    'OpGen','IROpGen','JumpCondOpGen','IRCompiler','reg_allocate','Param',
    'Function','CompilationUnit','address_of','Tuning','CallingConvention',
    'AbiRegister','Abi','BinaryAbi']

import enum
import collections
import weakref
import binascii
import operator
from functools import reduce
from typing import (Any,Callable,cast,Container,DefaultDict,Dict,Generic,
    Iterable,List,Optional,NamedTuple,NewType,Sequence,Sized,Set,Tuple,
    TYPE_CHECKING,Type,TypeVar,Union)

if __debug__:
    import sys

from . import debug
from . import c_types
from . import pyinternals
from .sorted_list import SortedList
from .dinterval import *
from .compilation_unit import *


EMIT_IR_TEST_CODE = False

SIZE_B = 1
SIZE_W = 2
SIZE_D = 4
SIZE_Q = 8


T = TypeVar('T')
U = TypeVar('U')


CALL_ALIGN_MASK = 0xf


def aligned_for_call(x):
    return (x + CALL_ALIGN_MASK) & ~CALL_ALIGN_MASK

address_of = id


class StackSection(enum.Enum):
    local = 0 # local stack space
    args = 1 # space for arguments passed by stack, for next function
    previous = 2 # space for arguments passed by stack, for current function

StackLocation = NamedTuple('StackLocation',[('sect',StackSection),('index',int)])

class LocationType(enum.Enum):
    register = 1
    stack = 2

class Lifetime:
    def __init__(self) -> None:
        self.intervals = DInterval() # type: DInterval[int]

        if __debug__:
            self.name = None
            self.origin = None

    @property
    def global_start(self):
        return self.intervals.global_start

    @property
    def global_end(self):
        return self.intervals.global_end

    def __contains__(self,x):
        return x in self.intervals

    def itv_at(self,x : int) -> Interval[int]:
        return self.intervals.interval_at(x)

    @staticmethod
    def _itv_shorthand(itv):
        end = itv.end - 1
        assert end >= itv.start
        if itv.start == end: return str(itv.start)
        return '{}-{}'.format(itv.start,end)

    def __repr__(self):
        if __debug__ and self.name:
            return '<{} - {} : [{}]>'.format(
                self.__class__.__name__,
                self.name,
                ','.join(map(self._itv_shorthand,self.intervals)))
        return '<{} : [{}]>'.format(self.__class__.__name__,','.join(map(self._itv_shorthand,self.intervals)))

class VarLifetime(Lifetime):
    def __init__(self,*,dbg_symbol : Optional[str]=None) -> None:
        super().__init__()

        # Every variable will have, at most, one stack location. This is to
        # allow efficient merging of variable locations from converging
        # branches and to allow instructions like x86's LEA to work correctly
        # with variables that don't have a value.
        self.preferred_stack_i = None # type: Optional[int]

        self.aliases = weakref.WeakSet()

        # if not None and not empty and debug.GDB_JIT_SUPPORT is true,
        # instances of IRAnnotation will be added to the produced intermediate
        # representation, that specifies the location of the associated value
        self.dbg_symbol = dbg_symbol

class AliasLifetime(Lifetime):
    def __init__(self,itv : VarLifetime) -> None:
        super().__init__()
        self.itv = itv
        itv.aliases.add(self)


class Filter(Container[T]):
    def __init__(self,include : Optional[Container[T]]=None,exclude : Container[T]=()) -> None:
        self.include = include
        self.exclude = exclude

    def __contains__(self,item):
        return (self.include is None or item in self.include) and item not in self.exclude


class _RegisterMetaType(type):
    def __new__(mcs,name,bases,namespace,*,allowed=None):
        cls = super().__new__(mcs,name,bases,namespace)
        cls.allowed = allowed if isinstance(allowed,Filter) else Filter(allowed)
        return cls

    # noinspection PyUnusedLocal
    def __init__(cls,name,bases,namespace,**kwds):
        super().__init__(name,bases,namespace)

    def __getitem__(cls,allowed):
        if allowed is None: return FixedRegister
        return _RegisterMetaType('DependentFixedRegister',(FixedRegister,),{},allowed=allowed)

    def __instancecheck__(cls,inst):
        return issubclass(inst.__class__,FixedRegister) and inst.reg_index in cls.allowed

    @staticmethod
    def generic_type():
        return FixedRegister

class AddressType:
    @staticmethod
    def generic_type():
        return AddressType

class _ImmediateMetaType(type):
    def __new__(mcs,name,bases,namespace,*,abi=None,allowed_range=None):
        assert (abi is None) == (allowed_range is None)

        cls = super().__new__(mcs,name,bases,namespace)
        cls.abi = abi
        cls.allowed = allowed_range
        return cls

    # noinspection PyUnusedLocal
    def __init__(cls,name,bases,namespace,**kwds):
        super().__init__(name,bases,namespace)

    def __getitem__(cls,args):
        return _ImmediateMetaType('DependentImmediate',(Immediate,),{},abi=args[0],allowed_range=args[1:])

    def __instancecheck__(cls,inst):
        return issubclass(inst.__class__,Immediate) and (cls.abi is None
            or cls.allowed[0] <= inst.val.realize(cls.abi) <= cls.allowed[1])

    @staticmethod
    def generic_type():
        return Immediate

def generic_type(x : object) -> type:
    if not isinstance(x,type):
        x = type(x)

    f = getattr(x,'generic_type',None)
    if f is not None:
        return f()
    return x


class PtrBinomial:
    """A numeric value dependent on the size of a pointer.

    This is a binomial equal to "ptr_factor * ptr_size + val" where ptr_size is
    the size of a pointer on the host machine.

    """
    __slots__ = 'val','ptr_factor'

    @staticmethod
    def __new__(cls,val: Union['PtrBinomial',int],ptr_factor: int = 0) -> 'PtrBinomial':
        if isinstance(val,PtrBinomial):
            if ptr_factor != 0:
                raise TypeError(
                    'ptr_factor cannot be non-zero if val is already an instance of PtrBinomial')
            return val

        r = super().__new__(cls)
        r.val = val
        r.ptr_factor = ptr_factor
        return r

    if TYPE_CHECKING:
        # noinspection PyUnusedLocal
        def __init__(self,val: Union['PtrBinomial',int],ptr_factor: int = 0) -> None:
            self.val = 0
            self.ptr_factor = ptr_factor

    def realize(self,abi) -> int:
        return self.ptr_factor * abi.ptr_size + self.val

    def __add__(self,b):
        if isinstance(b,PtrBinomial):
            return PtrBinomial(self.val + b.val,self.ptr_factor + b.ptr_factor)

        if isinstance(b,int):
            return PtrBinomial(self.val + b,self.ptr_factor)

        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return PtrBinomial(-self.val,-self.ptr_factor)

    def __sub__(self,b):
        if isinstance(b,PtrBinomial):
            return PtrBinomial(self.val - b.val,self.ptr_factor - b.ptr_factor)

        if isinstance(b,int):
            return PtrBinomial(self.val - b,self.ptr_factor)

        return NotImplemented

    def __rsub__(self,b):
        if isinstance(b,int):
            return PtrBinomial(b - self.val,-self.ptr_factor)

        return NotImplemented

    def __mul__(self,b):
        if isinstance(b,int):
            return PtrBinomial(self.val * b,self.ptr_factor * b)

        return NotImplemented

    __rmul__ = __mul__

    def __floordiv__(self,b):
        if isinstance(b,int):
            return PtrBinomial(self.val // b,self.ptr_factor // b)

    def __repr__(self):
        return 'PtrBinomial({},{})'.format(self.val,self.ptr_factor)

    def __str__(self):
        return str(self.val) if self.ptr_factor==0 else self.__repr__()

SIZE_PTR = PtrBinomial(0,1)

class Target:
    if not __debug__:
        __slots__ = 'displacement',

    def __init__(self):
        self.displacement = None # type: Optional[int]
        if __debug__:
            self.origin = _get_origin()

    def __repr__(self):
        return '<Target {:#x}>'.format(id(self))

class Value:
    __slots__ = 'data_type',
    def __init__(self,data_type : c_types.CType=c_types.t_void_ptr) -> None:
        self.data_type = data_type

    def size(self,abi):
        return self.data_type.size(abi)

class MutableValue(Value):
    __slots__ = ()


if __debug__:
    def _get_origin():
        f = sys._getframe(2)
        r = None
        while f:
            r = f.f_code.co_filename,f.f_lineno
            if f.f_code.co_filename != __file__:
                break

            f = f.f_back

        return r

# Var must compare by identity
# The "name" and "origin" attributes only act as helpers for debugging our
# python code
class Var(MutableValue):
    if not __debug__:
        __slots__ = 'lifetime','dbg_symbol'

    def __init__(self,name : Optional[str]=None,data_type : c_types.CType=c_types.t_void_ptr,lifetime : Optional[Lifetime]=None,dbg_symbol : Optional[str]=None) -> None:
        super().__init__(data_type)
        if __debug__:
            self.name = name
            self.origin = _get_origin()

        self.lifetime = lifetime
        self.dbg_symbol = dbg_symbol

    def __repr__(self):
        if __debug__ and self.name is not None:
            return '<{} "{}">'.format(self.__class__.__name__,self.name)

        return '<{} {:#x}>'.format(self.__class__.__name__,id(self))

class Block(MutableValue):
    __slots__ = 'parts','lifetime','__weakref__'

    dbg_symbol = None

    def __init__(self,parts : int,base_type : c_types.CType=c_types.t_void_ptr,*,lifetime : Optional[VarLifetime]=None) -> None:
        assert parts > 0
        super().__init__(c_types.TArray(base_type,parts))
        self.parts = [VarPart(self,i,base_type) for i in range(parts)]
        self.lifetime = lifetime

    def __getitem__(self,i):
        return self.parts[i]

    def __len__(self):
        return len(self.parts)

class VarPart(Var):
    __slots__ = 'block','offset'

    def __init__(self,block : Block,offset : int,data_type : c_types.CType=c_types.t_void_ptr,*,lifetime : Optional[AliasLifetime]=None) -> None:
        super().__init__(None,data_type,lifetime)
        #self._block = weakref.ref(block)
        self.block = block
        self.offset = offset

    #@property
    #def block(self) -> Block:
    #    r = self._block()
    #    assert isinstance(r,Block)
    #    return r

class Immediate(Value,metaclass=_ImmediateMetaType):
    __slots__ = 'val',

    def __init__(self,val : Union[int,PtrBinomial],data_type : c_types.CType=c_types.t_void_ptr) -> None:
        super().__init__(data_type)
        self.val = PtrBinomial(val)

    def __eq__(self,b):
        if isinstance(b,Immediate):
            return self.val == b.val
        if isinstance(b,(int,PtrBinomial)):
            return self.val == b
        return False

    def __ne__(self,b):
        return not self.__eq__(b)

    def __repr__(self):
        type_str = ''
        if self.data_type != c_types.t_void_ptr:
            type_str = ',' + repr(self.data_type)
        return 'Immediate({}{})'.format(self.val,type_str)

class IndirectVar(MutableValue,AddressType):
    __slots__ = 'offset','base','index','scale'

    def __init__(self,offset : Union[int,PtrBinomial]=0,base : Optional[Var]=None,index : Optional[Var]=None,scale : Union[int,PtrBinomial]=1,data_type : c_types.CType=c_types.t_void_ptr) -> None:
        super().__init__(data_type)
        self.offset = PtrBinomial(offset)
        self.base = base
        self.index = index
        self.scale = PtrBinomial(scale)

    def __eq__(self,b):
        if isinstance(b,IndirectVar):
            return (self.offset == b.offset
                and self.base == b.base
                and self.index == b.index
                and self.scale == b.scale
                and self.data_type == b.data_type)
        return False

    def __ne__(self,b):
        return not self.__eq__(b)

    def __repr__(self):
        type_str = ''
        if self.data_type != c_types.t_void_ptr:
            type_str = ',' + repr(self.data_type)
        return 'IndirectVar({},{!r},{!r},{}{})'.format(self.offset,self.base,self.index,self.scale,type_str)

class FixedRegister(MutableValue,metaclass=_RegisterMetaType):
    __slots__ = 'reg_index',

    def __init__(self,reg_index : int,data_type : c_types.CType=c_types.t_void_ptr) -> None:
        super().__init__(data_type)
        self.reg_index = reg_index

    def __repr__(self):
        type_str = ''
        if self.data_type != c_types.t_void_ptr:
            type_str = ',' + repr(self.data_type)
        return 'FixedRegister({}{})'.format(self.reg_index,type_str)

class StackItem(MutableValue,AddressType):
    __slots__ = 'index',

    def __init__(self,index : int,data_type : c_types.CType=c_types.t_void_ptr) -> None:
        super().__init__(data_type)
        self.index = index

    def __repr__(self):
        type_str = ''
        if self.data_type != c_types.t_void_ptr:
            type_str = ','+repr(self.data_type)
        return 'StackItem({}{})'.format(self.index,type_str)

class StackItemPart(MutableValue,AddressType):
    __slots__ = 'block','offset'

    def __init__(self,block : StackItem,offset : int,data_type : c_types.CType=c_types.t_void_ptr) -> None:
        super().__init__(data_type)
        self.block = block
        self.offset = offset

    def __repr__(self):
        type_str = ''
        if self.data_type != c_types.t_void_ptr:
            type_str = ',' + repr(self.data_type)
        return 'StackItemPart({!r},{}{})'.format(self.block,self.offset,type_str)

class ArgStackItem(MutableValue,AddressType):
    """A function argument passed via the stack.

    'index' refers to how many pointer-size increments the argument is from the
    top of the stack.

    If 'prev_frame' is true, the argument is from the previous stack. In other
    words, the argument was passed to the current function, not a function
    about to be called.

    """
    __slots__ = 'index','prev_frame'

    def __init__(self,index : int,prev_frame : bool=False,data_type : c_types.CType=c_types.t_void_ptr) -> None:
        super().__init__(data_type)
        self.index = index
        self.prev_frame = prev_frame

    def __eq__(self,b):
        if isinstance(b,ArgStackItem):
            return self.index == b.index and self.prev_frame == b.prev_frame and self.data_type == b.data_type
        return False

    def __ne__(self,b):
        return not self.__eq__(b)

    def __repr__(self):
        type_str = ''
        if self.data_type != c_types.t_void_ptr:
            type_str = ',' + repr(self.data_type)
        return 'ArgStackItem({},{!r}{})'.format(self.index,self.prev_frame,type_str)

class Symbol(Value):
    """A named entity to be linked."""

    __slots__ = 'name','address'

    def __init__(self,name : str,data_type : c_types.CType=c_types.t_void_ptr,address : Optional[int]=None) -> None:
        super().__init__(data_type)
        self.name = name
        self.address = address

    def __eq__(self,b):
        return isinstance(b,Symbol) and b.name == self.name

    def __ne__(self,b):
        return not self.__eq__(b)

    def __repr__(self):
        type_str = ''
        if self.data_type != c_types.t_void_ptr:
            type_str = ',' + repr(self.data_type)
        return 'Symbol({!r}{})'.format(self.name,type_str)

class PyConst(Value):
    """Represents a name or constant stored in one of the tuples in the
    function body object"""

    __slots__ = 'tuple_name','index','address'

    def __init__(self,tuple_name : str,index : int,address : int) -> None:
        super().__init__(c_types.PyObject_ptr)
        self.tuple_name = tuple_name
        self.index = index
        self.address = address

    def __eq__(self,b):
        return isinstance(b,PyConst) and b.address == self.address

    def __ne__(self,b):
        return not self.__eq__(b)

    def __repr__(self):
        return 'PyConst({!r},{},{})'.format(self.tuple_name,self.index,self.address)

def to_stack_item(location : StackLocation,data_type : c_types.CType=c_types.t_void_ptr) -> Union[StackItem,ArgStackItem]:
    if location.sect == StackSection.local:
        return StackItem(location.index,data_type=data_type)
    if location.sect == StackSection.args:
        return ArgStackItem(location.index,data_type=data_type)

    assert location.sect == StackSection.previous
    return ArgStackItem(location.index,True,data_type=data_type)

def ensure_same_size(abi,a,*args):
    sa = a.size(abi)
    for b in args:
        if b.size(abi) != sa:
            raise ValueError('arguments must have the same data size')

class Instr:
    __slots__ = 'op','args'

    def __init__(self,op : 'RegAllocatorOverloads',*args) -> None:
        self.op = op
        self.args = args

    def __repr__(self):
        return 'Instr({!r},{})'.format(self.op,','.join(map(repr,self.args)))


class ParamDir:
    """Specifies whether a parameter is read-from, written-to, both or neither.

    An example of an instruction that has a parameter that is neither read-from
    nor written-to is x86's 'lea'.

    """
    def __init__(self,reads : bool=False,writes : bool=False) -> None:
        self.reads = reads
        self.writes = writes

    def __or__(self,b):
        if isinstance(b,ParamDir):
            return ParamDir(self.reads or b.reads,self.writes or b.writes)

        return NotImplemented

    def __ior__(self,b):
        if isinstance(b,ParamDir):
            self.reads = self.reads or b.reads
            self.writes = self.writes or b.writes
            return self

        return NotImplemented

    def __repr__(self):
        return 'ParamDir({!r},{!r})'.format(self.reads,self.writes)

class Overload:
    def __init__(self,
            params : Sequence[Union[type,Tuple[type,...]]],
            func : Callable,
            *,
            min_len : Optional[int]=None,
            max_len : Optional[int]=None) -> None:
        self.params = params
        self.func = func
        self.min_len = min_len
        self.max_len = max_len

    def variant(self,params,func):
        return Overload(params,func,min_len=self.min_len,max_len=self.max_len)

    def matches_args(self,args : Sequence) -> bool:
        return len(args) == len(self.params) and all(isinstance(a,p) for a,p in zip(args,self.params))

    def __repr__(self):
        def type_print(x):
            if isinstance(x,tuple):
                return '({})'.format(','.join(map(type_print,x)))
            return x.__name__
        return '<Overload - {}>'.format(','.join(map(type_print,self.params)))

def type_match_score(params,args):
    if len(params) != len(args): return 0
    score = 1
    for p,a in zip(params,args):
        if isinstance(a,(VarPart,Block)):
            # currently, VarPart and Block instances cannot be moved to
            # registers
            if not hassubclass(p,AddressType):
                return 0
        elif hassubclass(p,(FixedRegister,AddressType)):
            score += isinstance(a,p)
        elif isinstance(a,p):
            score += 1
        else:
            # only registers and addresses can be moved between each other
            return 0

    return score

def split_tuples(r,tail,head=()):
    if tail:
        next_tail = tail[1:]
        if isinstance(tail[0],tuple):
            for val in tail[0]: split_tuples(r,next_tail,head + (val,))
        else:
            split_tuples(r,next_tail,head + (tail[0],))
    else:
        r.append(head)

class RegAllocatorOverloads:
    def __init__(self,param_dirs : Sequence[ParamDir]) -> None:
        self.param_dirs = param_dirs

    def best_match(self,args : Sequence) -> Overload:
        return self.exact_match(args)

    def exact_match(self,args : Sequence) -> Overload:
        raise NotImplementedError()

    def to_ir2(self,args : Sequence) -> 'Instr2':
        return Instr2(self,self.exact_match(args),args)

    def __call__(self,*args):
        return self.to_ir2(args)

    def assembly(self,args : Sequence,addr : int,binary : bytes,annot : Optional[str]=None) -> str:
        raise NotImplementedError()

class OpDescription(RegAllocatorOverloads):
    """Describes a machine code instruction for a given set of parameters.

    The parameter types may vary, but the number of parameters and whether a
    parameter is read, write or read-write, must be the same. If an instruction
    can, for example, take 2 and 3 parameters, it must be represented with two
    instances of OpDescription.

    For performance reasons, don't create overloads that differ only by which
    registers or which immediate values are accepted; instead, create one
    function that makes the check manually (otherwise, 'exact_match' and
    consequently 'to_ir2' wont work).

    """
    def __init__(self,name : str,overloads : Sequence[Overload],param_dirs : Sequence[ParamDir]) -> None:
        assert all(len(param_dirs) == len(o.params) for o in overloads)

        super().__init__(param_dirs)

        self.name = name
        self.overloads = overloads

        self._type_lookup = {} # type: Dict[Tuple[type,...],Overload]

    def best_match(self,args : Sequence) -> Overload:
        best_o = self.overloads[0]
        best_score = type_match_score(best_o.params,args)
        for o in self.overloads[1:]:
            s = type_match_score(o.params,args)
            if s > best_score:
                best_score = s
                best_o = o

        if best_score == 0:
            raise TypeError('One or more arguments has an incorrect type ' +
                'and the parameter is not a register or address, for "{}"'
                .format(self.name))
        return best_o

    def exact_match(self,args : Sequence) -> Overload:
        types = tuple(a.__class__ for a in args)
        try:
            r = self._type_lookup[types]
        except KeyError as exc:
            for o in self.overloads:
                if o.matches_args(args):
                    self._type_lookup[types] = o
                    return o

            raise TypeError('no overload takes ({})'.format(','.join(t.__name__ for t in types))) from exc

        # this can still fail because r.params may contain dependent types
        if r.matches_args(args): return r

        raise ValueError('one or more arguments contains a disallowed value')

    def __repr__(self):
        return '<{} "{}">'.format(self.__class__.__name__,self.name)

    def assembly_name(self,args):
        """Get the name of the op-code based on the arguments.

        'args' may be modified, to change the arguments printed."""
        return self.name

    def assembly_arg(self,nextaddr,x):
        return str(x)

    def assembly(self,args : Sequence,addr : int,binary : bytes,annot : Optional[str]=None):
        nextaddr = addr + len(binary)

        args = list(args)
        name = self.assembly_name(args)

        return '{:8x}: {:22}{:8} {}{}'.format(
            addr,
            binascii.hexlify(binary).decode(),
            name,
            ', '.join(self.assembly_arg(nextaddr,arg) for arg in args),
            ('  ; '+annot) if annot else '')

class CommentDesc(RegAllocatorOverloads):
    _sole_overload = Overload([str],lambda x: b'')

    def __init__(self,inline):
        super().__init__([ParamDir(True,False)])
        self.inline = inline

    def exact_match(self,args : Sequence) -> Overload:
        if len(args) != 1:
            raise TypeError('there must be exactly one argument')

        return self._sole_overload

    def assembly(self,args : Sequence,addr : int,binary: bytes,annot : Optional[str] = None):
        return '; {} {}'.format(args[0],annot or '')

comment_desc = CommentDesc(False)
inline_comment_desc = CommentDesc(True)


class InvalidateRegs:
    """Force variables stored in scratch registers to be moved to the stack or
    preserved registers"""
    __slots__ = 'to_free','pres'

    def __init__(self,to_free,pres):
        self.to_free = to_free
        self.pres = pres

    def __repr__(self):
        return 'InvalidateRegs({!r},{!r})'.format(self.to_free,self.pres)

class CreateVar:
    """Indicates that 'var' has obtained a value from 'val'.

    'val' must be an instance of FixedRegister or ArgStackItem. If 'val' is a
    register, it must not be used by any variable, by this point in the
    instruction sequence.

    """
    __slots__ = 'var','val'

    def __init__(self,var : Var,val : Union[FixedRegister,ArgStackItem]) -> None:
        self.var = var
        self.val = val

    def __repr__(self):
        return 'CreateVar({!r},{!r})'.format(self.var,self.val)

class IRJump:
    """Indicates that the next instruction executed may be the one after
    'dest'.

    If 'conditional' is false, the next instruction executed will always be the
    one after 'dest'.

    Sometimes instructions need to be inserted before an instance of IRJump to
    move variables, 'jump_ops' specifies how many elements back, those
    instructions should be inserted (a value of 0 would mean the instructions
    should be inserted directly behind).

    This is used by the register allocator to determine value life-times.

    """
    __slots__ = 'dests','conditional','jump_ops'

    def __init__(self,dests : Union[Target,Iterable[Target]],conditional : bool,jump_ops : int) -> None:
        self.dests = (dests,) if isinstance(dests,Target) else tuple(dests) # type: Tuple[Target]
        self.conditional = conditional
        self.jump_ops = jump_ops

    def __repr__(self):
        return 'IRJump({!r},{!r},{})'.format(self.dests,self.conditional,self.jump_ops)

class IndirectMod:
    """Indicates that a variable was read or updated externally, somehow.

    When written, if the value exists in both a register and the stack, one is
    made invalid depending on which was updated.

    """
    __slots__ = 'var','read','write','loc_type'

    def __init__(self,var: Var,read: object,write: object,loc_type: LocationType) -> None:
        self.var = var
        self.read = bool(read)
        self.write = bool(write)
        self.loc_type = loc_type

    def __repr__(self):
        return 'IndirectMod({!r},{!r},{!r},{!r})'.format(self.var,self.read,self.write,self.loc_type)

class LockRegs:
    """"Lock" one or more registers.

    When a register is "locked" it will not be available for use by variables.
    However, locking a register will cause an existing variable to be unloaded
    from that register.

    """
    __slots__ = 'regs',

    def __init__(self,regs : Iterable[int]) -> None:
        self.regs = tuple(regs)

    def __repr__(self):
        return 'LockRegs({!r})'.format(self.regs)

class UnlockRegs:
    __slots__ = 'regs',

    def __init__(self,regs : Iterable[int]) -> None:
        self.regs = tuple(regs)

    def __repr__(self):
        return 'UnlockRegs({})'.format(self.regs)

IRAnnotation = collections.namedtuple('IRAnnotation','descr')

def annotate(descr=None):
    return [IRAnnotation(descr)] if debug.GDB_JIT_SUPPORT else []

class IRSymbolLocDescr:
    def __init__(self,symbol : str,loc : 'ItvLocation') -> None:
        self.symbol = symbol
        self.loc = loc.copy()

    def __repr__(self):
        return 'IRSymbolDescr({!r},{!r})'.format(self.symbol,self.loc)

def annotate_symbol_loc(itv : VarLifetime,loc : 'ItvLocation'):
    if itv.dbg_symbol and debug.GDB_JIT_SUPPORT:
        return [IRAnnotation(IRSymbolLocDescr(itv.dbg_symbol,loc))]

    return []


class Instr2:
    """Second level intermediate representation op-code.

    This is like Instr, but 'variables' have been replaced with registers
    and stack positions.

    """
    __slots__ = 'op','overload','args'

    def __init__(self,op : RegAllocatorOverloads,overload : Overload,args : Sequence) -> None:
        assert not any(isinstance(a,(Var,Block)) for a in args)
        self.op = op
        self.overload = overload
        self.args = args

class CmpType(enum.Enum):
    eq = 1
    ne = 2
    lt = 3
    le = 4
    gt = 5
    ge = 6

def cmp_complement(t : CmpType):
    return [
        CmpType.ne,
        CmpType.eq,
        CmpType.ge,
        CmpType.gt,
        CmpType.le,
        CmpType.lt
    ][t.value-1]

class Cmp:
    def complement(self) -> 'Cmp':
        raise NotImplementedError()

class BinCmp(Cmp):
    def __init__(self,a : Value,b : Value,op : CmpType,signed : bool=True) -> None:
        self.a = a
        self.b = b
        self.op = op
        self.signed = signed

    def complement(self) -> 'BinCmp':
        return BinCmp(self.a,self.b,cmp_complement(self.op),self.signed)

class AndCmp(Cmp):
    def __init__(self,a : Cmp,b : Cmp) -> None:
        self.a = a
        self.b = b

    def complement(self) -> 'OrCmp':
        return OrCmp(self.a.complement(),self.b.complement())

class OrCmp(Cmp):
    def __init__(self,a: Cmp,b: Cmp) -> None:
        self.a = a
        self.b = b

    def complement(self) -> AndCmp:
        return AndCmp(self.a.complement(),self.b.complement())

class OpType(enum.Enum):
    add = 1
    sub = 2
    mul = 3
    div = 4
    and_ = 5
    or_ = 6
    xor = 7

commutative_ops = {OpType.add,OpType.mul,OpType.and_,OpType.or_,OpType.xor}

class UnaryOpType(enum.Enum):
    neg = 1

class ShiftDir(enum.Enum):
    left = -1
    right = 1

IROp = Union[Instr,InvalidateRegs,CreateVar,Target,IRJump,IndirectMod,LockRegs,UnlockRegs,IRAnnotation]
IRCode = List[IROp]
IROp2 = Union[Instr2,Target,IRAnnotation]
IRCode2 = List[IROp2]
MCode = NewType('MCode',Sized)
AnnotatedOp = Union[MCode,IRAnnotation]
AnnotatedCode = List[AnnotatedOp]

class DelayedCompileLate:
    def __init__(self,op,overload,args,displacement):
        self.op = op
        self.overload = overload
        self.args = args
        self.displacement = displacement

    def __len__(self):
        assert self.overload.max_len is not None
        return self.overload.max_len

class IRCompiler:
    def __init__(self,abi):
        self.abi = abi

    def prolog(self) -> IRCode2:
        raise NotImplementedError()

    def epilog(self) -> IRCode2:
        raise NotImplementedError()

    def get_reg(self,index: int,size : int) -> Any:
        raise NotImplementedError()

    def get_stack_addr(self,index : int,offset : int,size : int,block_size : int,sect : StackSection) -> Any:
        raise NotImplementedError()

    def get_displacement(self,amount : int,force_wide : bool) -> Any:
        raise NotImplementedError()

    def get_immediate(self,val : int,size : int) -> Any:
        raise NotImplementedError()

    def get_machine_arg(self,arg,displacement):
        if isinstance(arg,FixedRegister):
            return self.get_reg(arg.reg_index,arg.size(self.abi))
        if isinstance(arg,StackItem):
            size = arg.size(self.abi)
            return self.get_stack_addr(arg.index,0,size,size,StackSection.local)
        if isinstance(arg,StackItemPart):
            return self.get_stack_addr(arg.block.index,arg.offset,arg.size(self.abi),arg.block.size(self.abi),StackSection.local)
        if isinstance(arg,ArgStackItem):
            size = arg.size(self.abi)
            return self.get_stack_addr(
                arg.index,
                0,
                size,
                size,
                StackSection.previous if arg.prev_frame else StackSection.args)
        if isinstance(arg,Target):
            assert arg.displacement is not None
            return self.get_displacement(arg.displacement - displacement,False)
        if isinstance(arg,Immediate):
            return self.get_immediate(arg.val.realize(self.abi),arg.size(self.abi))

        return arg

    def compile_early(self,item : Instr2,displacement : int) -> Sized:
        ready = True
        new_args = [None] * len(item.args) # type: List[Any]
        for i,arg in enumerate(item.args):
            if isinstance(arg,Target) and arg.displacement is None:
                ready = False
                new_args[i] = arg
            else:
                new_args[i] = self.get_machine_arg(arg,displacement)

        if ready:
            # noinspection PyCallingNonCallable
            return item.overload.func(*new_args)

        return DelayedCompileLate(item.op,item.overload,new_args,displacement)

    def compile_late(self,item : Sized) -> MCode:
        if isinstance(item,DelayedCompileLate):
            new_args = [None] * len(item.args)
            for i,arg in enumerate(item.args):
                new_args[i] = arg
                if isinstance(arg,Target):
                    assert arg.displacement is not None
                    new_args[i] = self.get_displacement(arg.displacement-item.displacement,True)

            return item.overload.func(*new_args)
        else:
            assert hasattr(item,'__len__')
            return cast(MCode,item)

    def nops(self,size : int) -> List[Instr2]:
        raise NotImplementedError()


EST = TypeVar('EST',bound='ExtraState')

class ExtraState:
    """An abstract class for modifying instructions in the register allocation
    pass."""
    def copy(self : EST) -> EST:
        raise NotImplementedError()

    def process_instr(self,i : int,instr : Instr) -> Optional[Instr]:
        raise NotImplementedError()

    def conform_to(self : EST,other : EST) -> None:
        raise NotImplementedError()

class NullState(ExtraState):
    """A do-nothing implementation of ExtraState"""
    def copy(self):
        return self

    def process_instr(self,i,instr):
        return instr

    def conform_to(self,other):
        pass

class PyFuncInfo:
    def __init__(self,frame_var,func_var,args_var,kwds_var):
        self.frame_var = frame_var
        self.func_var = func_var
        self.args_var = args_var
        self.kwds_var = kwds_var

class FinallyTarget:
    def __init__(self) -> None:
        self.start = Target()
        self.next_var = Var()

        # A list of targets that the code jumps to, after the end of the
        # finally block. This is needed for variable lifetime determination
        self.next_targets = []  # type: List[Target]

        if __debug__:
            self._used = False

    def add_next_target(self,t: Target):
        assert not self._used
        self.next_targets.append(t)

class OpGen(Generic[T]):
    def __init__(self,abi: 'Abi',func_args: Iterable[Var]=(),callconv: CallConvType=CallConvType.default,pyinfo: Optional[PyFuncInfo]=None) -> None:
        self.abi = abi
        self.func_arg_vars = list(func_args)
        self.callconv = callconv
        self.pyinfo = pyinfo

    def callconv_for(self,callconv: Optional[CallConvType],prev_frame: bool) -> CallConvType:
        if callconv is not None: return callconv
        if prev_frame: return self.callconv
        return CallConvType.default

    def bin_op(self,a : Value,b : Value,dest : MutableValue,op_type : OpType) -> T:
        raise NotImplementedError()

    def unary_op(self,a : Value,dest : MutableValue,op_type : UnaryOpType) -> T:
        raise NotImplementedError()

    def load_addr(self,addr : MutableValue,dest : MutableValue) -> T:
        raise NotImplementedError()

    def call(self,func,args : Sequence[Value]=(),store_ret : Optional[Var]=None,callconv: CallConvType=CallConvType.default) -> T:
        raise NotImplementedError()

    def call_preloaded(self,func,args : int,store_ret : Optional[Var]=None,callconv: CallConvType=CallConvType.default) -> T:
        raise NotImplementedError()

    def get_func_arg(self,i : int,prev_frame : bool=False,callconv : Optional[CallConvType]=None) -> MutableValue:
        raise NotImplementedError()

    def jump(self,dest : Union[Target,Value],targets : Union[Iterable[Target],Target,None]=None) -> T:
        raise NotImplementedError()

    def jump_if(self,dest : Target,cond : Cmp) -> T:
        raise NotImplementedError()

    def if_(self,cond : Cmp,on_true : T,on_false : Optional[T]) -> T:
        raise NotImplementedError()

    def do_while(self,action : T,cond : Cmp) -> T:
        raise NotImplementedError()

    def jump_table(self,val : Value,targets : Sequence[Target]) -> T:
        raise NotImplementedError()

    def move(self,src : Value,dest : MutableValue) -> T:
        raise NotImplementedError()

    def shift(self,src : Value,shift_dir : ShiftDir,amount : Value,dest : MutableValue) -> T:
        raise NotImplementedError()

    def enter_finally(self,f: FinallyTarget,next_t: Optional[Target]=None) -> T:
        raise NotImplementedError()

    def finally_body(self,f: FinallyTarget,body: T) -> T:
        """Create code for a "finally" body.
        
        This works like a local function, except it can only be called from the
        immediate outer scope, and can be jumped out of, to anywhere in the
        outer scope.
        
        Finally body code can only be entered via "enter_finally".
        
        """
        raise NotImplementedError()

    def compile(self,func_name : str,code: T,ret_var: Optional[Var]=None,end_targets=()) -> Function:
        raise NotImplementedError()

    def new_func_body(self):
        raise NotImplementedError()

# noinspection PyAbstractClass
class IROpGen(OpGen[IRCode]):
    max_args_used = 0

    if TYPE_CHECKING:
        def __init__(self,abi: 'BinaryAbi',func_args: Iterable[Var]=(),callconv: CallConvType=CallConvType.default,pyinfo: Optional[PyFuncInfo]=None) -> None:
            super().__init__(abi,func_args,callconv,pyinfo)
        abi = cast(BinaryAbi,object()) # type: BinaryAbi

    def get_func_arg(self,i : int,prev_frame : bool=False,callconv : Optional[CallConvType]=None) -> MutableValue:
        return self.abi.callconvs[
            self.callconv_for(callconv,prev_frame).value].get_arg(self.abi,i,prev_frame)

    def get_return(self,prev_frame : bool=False,callconv : Optional[CallConvType]=None) -> MutableValue:
        return self.abi.callconvs[
            self.callconv_for(callconv,prev_frame).value].get_return(self.abi,prev_frame)

    def call(self,func,args : Sequence[Value]=(),store_ret : Optional[Var]=None,callconv : CallConvType=CallConvType.default) -> IRCode:
        self.max_args_used = max(self.max_args_used,len(args))
        return self._call_impl(func,args,store_ret,callconv)

    def call_preloaded(self,func,args : int,store_ret : Optional[Var]=None,callconv : CallConvType=CallConvType.default) -> IRCode:
        self.max_args_used = max(self.max_args_used,args)
        return self._call_preloaded_impl(func,args,store_ret,callconv)

    def _call_impl(self,func,args : Sequence[Value],store_ret : Optional[Var],callconv : CallConvType) -> IRCode:
        arg_dests = [self.get_func_arg(i,callconv=callconv) for i in range(len(args))]
        arg_r_indices = [arg.reg_index for arg in arg_dests if isinstance(arg,FixedRegister)]

        r = []  # type: IRCode
        if arg_r_indices: r.append(LockRegs(arg_r_indices))

        for arg,dest in zip(args,arg_dests):
            r += self.move(arg,dest)

        r += self._call_preloaded_impl(func,len(args),store_ret,callconv)

        if arg_r_indices: r.append(UnlockRegs(arg_r_indices))

        return r

    def _call_preloaded_impl(self,func,args : int,store_ret : Optional[Var],callconv : CallConvType) -> IRCode:
        raise NotImplementedError()

    def get_compiler(self,regs_used: Set[int],stack_used: int,args_used: int) -> IRCompiler:
        raise NotImplementedError()

    def allocater_extra_state(self) -> ExtraState:
        """Return a new instance of ExtraState.

        During register allocation, implicit state is tacked in a way that
        takes into account branching, as the instructions are analyzed, one by
        one. This method returns an instance of ExtraState that stores nothing
        and does nothing, but this method can be overridden to store extra
        data per branch, and substitute or even omit certain instructions as
        each branch is analyzed.

        """
        return NullState()

    def process_indirection(self,instr: Instr,ov: Overload,inds: Sequence[int]) -> Tuple[Instr,Overload]:
        """Remove instances of IndirectVar.

        The return value is equivalent to the input parameters, except each
        argument indexed by "inds" (all of which will be instances of
        IndirectVar) is replaced with one or more of its component instances of
        Var, and "instr.op" and "ov" are replaced by equivalent instances of
        RegAllocatorOverloads and Overload that accept the new arguments. The
        intent is to free the register allocator from the burden of supporting
        different address formats.

        """
        raise NotImplementedError()

    def compile(self,func_name: str,code1: IRCode,ret_var: Optional[Var]=None,end_targets=()) -> Function:
        if EMIT_IR_TEST_CODE:
            print(func_name+':')
            print(debug_gen_code(code1))
            print()

        code2 = [] # type: IRCode
        for i,av in enumerate(self.func_arg_vars):
            loc = self.get_func_arg(i,True)
            if isinstance(loc,(FixedRegister,ArgStackItem)):
                code2.append(CreateVar(av,loc))
            else:
                # this case probably won't even come up
                code2.extend(self.move(loc,av))
        code2.extend(code1)
        if ret_var is not None:
            code2.extend(self.move(ret_var,self.get_return(True)))

        code,r_used,s_used = reg_allocate(self,code2,self.abi.gen_regs)
        irc = self.get_compiler(r_used,s_used,self.max_args_used)

        displacement = 0
        pad_size = 0
        annot_size = 0
        annots = []  # type: List[debug.Annotation]

        # this item will be replaced with padding if needed
        late_chunks = [None]  # type: List[Optional[Sized]]

        code = irc.prolog() + code + irc.epilog()

        for instr in reversed(code):
            if isinstance(instr,IRAnnotation):
                descr = instr.descr
                if isinstance(descr,IRSymbolLocDescr):
                    descr = debug.VariableLoc(
                        descr.symbol,
                        irc.get_machine_arg(descr.loc.to_ir(),displacement) if descr.loc else None)

                annot = debug.Annotation(descr,annot_size)

                if self.abi.assembly:
                    late_chunks.append(AsmSequence([AsmOp(comment_desc,
                        ('annotation: {!r}'.format(annot.descr),),b'')]))

                # since appending to a list is O(1) while prepending is O(n), we
                # add the items backwards and reverse the list afterwards
                annots.append(annot)

                annot_size = 0
            elif isinstance(instr,Target):
                instr.displacement = displacement
            else:
                assert isinstance(instr,Instr2)
                chunk = irc.compile_early(make_asm_if_needed(instr,self.abi.assembly),displacement)

                # items are added backwards for the same reason as above
                late_chunks.append(chunk)
                displacement -= len(chunk)
                annot_size += len(chunk)

        assert annot_size == 0 or not annots,"if there are any annotations, there should be one at the start"

        annots.reverse()

        # add padding for alignment
        if CALL_ALIGN_MASK:
            unpadded = displacement
            displacement = aligned_for_call(displacement)
            pad_size = displacement - unpadded
            if pad_size:
                late_chunks[0] = code_join(
                    [irc.compile_early(make_asm_if_needed(c,self.abi.assembly),0) for c
                        in irc.nops(pad_size)])

        for et in end_targets:
            et.displacement -= displacement

        return Function(
            code_join([irc.compile_late(c) for c in reversed(late_chunks) if c is not None]),
            pad_size,
            name=func_name,
            annotation=annots,
            returns=c_types.t_void if ret_var is None else ret_var.data_type,
            params=[Param(av.dbg_symbol or '__a'+str(i),av.data_type) for i,av in enumerate(self.func_arg_vars)],
            callconv=self.callconv)

    def new_func_body(self):
        return pyinternals.FunctionBody.__new__(pyinternals.FunctionBody)

# noinspection PyAbstractClass
class JumpCondOpGen(IROpGen):
    """An implementation of IROpGen that uses jumps for control flow"""

    def jump_if(self,dest : Target,cond : Cmp) -> IRCode:
        if isinstance(cond,OrCmp):
            return self.jump_if(dest,cond.a) + self.jump_if(dest,cond.b)

        if isinstance(cond,AndCmp):
            return self.if_(cond.a,self.jump_if(dest,cond.b),None)

        assert isinstance(cond,BinCmp)
        return self._basic_jump_if(dest,cond)

    def _basic_jump_if(self,dest : Target,cond : BinCmp) -> IRCode:
        raise NotImplementedError()

    def if_(self,cond : Cmp,on_true : IRCode,on_false : Optional[IRCode]) -> IRCode:
        endif = Target()
        if on_false:
            else_ = Target()
            return self.jump_if(else_,cond.complement()) + on_true + self.jump(endif) + [else_] + on_false + [endif]

        return self.jump_if(endif,cond.complement()) + on_true + [endif]

    def do_while(self,action : IRCode,cond : Cmp) -> IRCode:
        t = Target()
        return cast(IRCode,[t]) + action + self.jump_if(t,cond.complement())

class FollowNonlinear(Generic[T,U]):
    """Follow code non-linearly by splitting and merging state with the
    branches."""
    def __init__(self,state : T) -> None:
        self.state = state # type: Optional[T]
        self.prior_states = {} # type: Dict[Target,T]
        self.pending_states = collections.defaultdict(list) # type: DefaultDict[Target,List[U]]

        self.elided_targets = {} # type: Dict[Target,int]

        self.farthest_i = -1

    def backtracking(self,op_i):
        return self.farthest_i is not None and op_i <= self.farthest_i

    def handle_instr(self,op_i : int,op : Instr) -> None:
        pass

    def handle_invalidate_regs(self,op_i : int,op : InvalidateRegs) -> None:
        pass

    def handle_create_var(self,op_i : int,op : CreateVar) -> None:
        pass

    def handle_indirect_mod(self,op_i: int,op: IndirectMod) -> None:
        pass

    def handle_target(self,op_i: int,op: Target) -> None:
        pass

    def handle_elided_target(self,op_i: int,op: Target) -> None:
        pass

    def _redo_section(self,prev_i: int,code: IRCode):
        old_s = self.state
        self.state = self.make_prior_state(prev_i)
        self.run(code,prev_i - 1)
        self.state = old_s

    def handle_irjump(self,op_i: int,op: IRJump,code: IRCode) -> None:
        assert self.state is not None and op.dests
        assert all((d in self.prior_states) == (op.dests[0] in self.prior_states) for d in op.dests[1:]),(
            'a jump is not allowed to have both forward and backward targets')

        unhandled_dests = [] # type: List[Target]
        for d in op.dests:
            prev_i = self.elided_targets.get(d)
            if prev_i is not None:
                # if there is a jump backwards, to a part that had a None
                # state, go back and handle it again
                self._redo_section(prev_i,code)

                # _redo_section terminates as soon as it encounters an already
                # handled Target, but there could have been a jump forward to
                # another elided Target between prev_i and here
                for t in self.elided_targets.keys() & self.pending_states.keys():
                    self._redo_section(self.elided_targets[t],code)
            else:
                unhandled_dests.append(d)

        if unhandled_dests:
            if unhandled_dests[0] in self.prior_states:
                self.conform_prior_states(op_i,op.jump_ops,[self.prior_states[d] for d in unhandled_dests])
            else:
                pending = self.make_pending_state(op_i,op)
                for d in unhandled_dests:
                    self.pending_states[d].append(pending)

        if not op.conditional:
            old = self.state
            self.state = None
            self.after_state_change(old)

    def handle_lock_regs(self,op_i : int,op : LockRegs):
        pass

    def handle_unlock_regs(self,op_i : int,op : UnlockRegs):
        pass

    # occurs when merging states at a target we just reached
    def conform_states_to(self,op_i : int,state : Sequence[U]) -> None:
        pass

    # occurs when merging states at a target we already went over
    def conform_prior_states(self,op_i : int,jump_ops : int,states : Sequence[T]) -> None:
        pass

    def after_state_change(self,old_state : Optional[T]) -> None:
        pass

    def make_pending_state(self,op_i : int,op : IRJump) -> U:
        raise NotImplementedError()

    def make_prior_state(self,op_i : int) -> T:
        raise NotImplementedError()

    def recall_pending_state(self,op_i : int,state : U) -> T:
        raise NotImplementedError()

    def handle_op(self,op_i : int,code : IRCode) -> bool:
        op = code[op_i]

        if isinstance(op,Instr):
            if self.state is not None:
                self.handle_instr(op_i,op)
        elif isinstance(op,InvalidateRegs):
            if self.state is not None:
                self.handle_invalidate_regs(op_i,op)
        elif isinstance(op,CreateVar):
            if self.state is not None:
                self.handle_create_var(op_i,op)
        elif isinstance(op,IndirectMod):
            if self.state is not None:
                self.handle_indirect_mod(op_i,op)
        elif isinstance(op,Target):
            ps = self.prior_states.get(op)
            if ps is not None:
                if self.state is not None:
                    self.conform_prior_states(op_i,0,[ps])
                return False

            states = self.pending_states.get(op)

            if states:
                old = self.state

                del self.pending_states[op]
                if self.state is None:
                    self.state = self.recall_pending_state(op_i,states[0])
                    states = states[1:]

                if states:
                    self.conform_states_to(op_i,states)

                self.after_state_change(old)

            if self.state is not None:
                self.prior_states[op] = self.make_prior_state(op_i)
                try:
                    del self.elided_targets[op]
                except KeyError:
                    pass
                self.handle_target(op_i,op)
            else:
                self.elided_targets[op] = op_i
                self.handle_elided_target(op_i,op)
        elif isinstance(op,IRJump):
            if self.state is not None:
                self.handle_irjump(op_i,op,code)
        elif isinstance(op,LockRegs):
            if self.state is not None:
                self.handle_lock_regs(op_i,op)
        else:
            assert isinstance(op,UnlockRegs)
            if self.state is not None:
                self.handle_unlock_regs(op_i,op)

        return True

    def next_i(self,i : int,code : Sequence[IROp]) -> Optional[int]:
        return i + 1 if (i + 1) < len(code) else None

    def run(self,code : IRCode,start : int=-1) -> None:
        op_i = start
        while True:
            next_i = self.next_i(op_i,code)
            if next_i is None: break
            op_i = next_i

            if not self.handle_op(op_i,code): break
            self.farthest_i = max(op_i,self.farthest_i)


class VarState:
    def __init__(self,read_starts : Optional[Dict[Lifetime,int]]=None) -> None:
        self.read_starts = {} if read_starts is None else read_starts # type: Dict[Lifetime,int]

    def apply(self,life : Lifetime,i : int,only_tracked=False) -> None:
        rs = self.read_starts.get(life)
        if rs is not None:
            itv = Interval(i,rs)
            del self.read_starts[life]
        else:
            if only_tracked: return
            itv = Interval(i,i + 1)

        life.intervals |= itv
        #print('{} |= {}'.format(life.intervals,itv))
        #print('{}: {}\n'.format(life.name,life.intervals))

    def apply_all(self,i : int) -> None:
        while self.read_starts:
            life,rs = self.read_starts.popitem()
            life.intervals |= Interval(i,rs)

    def __repr__(self):
        return 'VarState({!r})'.format(self.read_starts)

class _CalcVarIntervalsCommon(FollowNonlinear[VarState,Tuple[VarState,int]]):
    def __init__(self) -> None:
        super().__init__(VarState())

    # Here, "write" means completely overwrite. In the case of partial write,
    # both "read" and "write" will be false.
    def create_var_life(self,var,_i,read,write):
        raise NotImplementedError()

    def handle_instr(self,op_i : int,op : Instr) -> None:
        var_dirs = collections.defaultdict(ParamDir) # type: DefaultDict[Union[Var,Block],ParamDir]
        for var,pd in zip(op.args,op.op.param_dirs):
            if isinstance(var,(Var,Block)):
                var_dirs[var] |= pd
            elif isinstance(var,IndirectVar):
                assert not (isinstance(var.base,Block) or isinstance(var.index,Block))
                if isinstance(var.base,Var):
                    var_dirs[var.base].reads = True
                if isinstance(var.index,Var):
                    var_dirs[var.index].reads = True

        for var,pd in var_dirs.items():
            self.create_var_life(var,op_i,pd.reads,pd.writes)

    def handle_create_var(self,op_i : int,op : CreateVar) -> None:
        self.create_var_life(op.var,op_i,False,True)

    def handle_indirect_mod(self,op_i : int,op : IndirectMod) -> None:
        self.create_var_life(op.var,op_i,op.read,op.write)

    def handle_irjump(self,op_i : int,op : IRJump,code : IRCode) -> None:
        assert self.state is not None
        state = self.state

        super().handle_irjump(op_i,op,code)

        if self.state is None:
            state.apply_all(op_i+1)

            prior = [self.prior_states.get(d) for d in op.dests]
            if prior[0]:
                assert all(prior)
                lives = set()
                for s in prior:
                    lives.update(s.read_starts)
                self.state = VarState({life: op_i + 1 for life in lives})
            else:
                assert not any(prior)
                self.state = state

    def make_prior_state(self,op_i : int) -> VarState:
        assert self.state is not None
        return VarState(self.state.read_starts.copy())

    def make_pending_state(self,op_i : int,op : IRJump) -> Tuple[VarState,int]:
        assert self.state is not None
        return VarState(self.state.read_starts.copy()),op_i

    def recall_pending_state(self,op_i : int,state : Tuple[VarState,int]) -> VarState:
        return state[0]

    def conform_prior_states(self,op_i : int,jump_ops : int,states : Sequence[VarState]) -> None:
        assert self.state is not None

        for state in states:
            for life in (state.read_starts.keys() | self.state.read_starts.keys()):
                m_rs = self.state.read_starts.get(life)
                if m_rs is None:
                    assert life in state.read_starts
                    m_rs = op_i + 1

                self.state.read_starts[life] = m_rs

    def next_i(self,i : int,code : Sequence[IROp]) -> Optional[int]:
        return i - 1 if i > 0 else None

class _CalcVarIntervals(_CalcVarIntervalsCommon):
    def __init__(self):
        super().__init__()

        # maps a lifetime to a set of code positions that we will need to
        # back-track to
        self.back_starts = collections.defaultdict(set) # type: DefaultDict[int,Set[Lifetime]]

        self.block_vars = [] # type: List[Block]

    # Here, "write" means completely overwrite. In the case of partial write,
    # both "read" and "write" will be false.
    def create_var_life(self,var,_i,read,write):
        assert self.state is not None

        i = _i

        # the variable doesn't officially exist until after the instruction
        if write and not read: i += 1

        if var.lifetime is None:
            if isinstance(var,VarPart):
                if var.block.lifetime is None:
                    var.block.lifetime = VarLifetime()
                    self.block_vars.append(var.block)

                var.lifetime = AliasLifetime(
                    cast(VarLifetime,var.block.lifetime))
            else:
                var.lifetime = VarLifetime(dbg_symbol=var.dbg_symbol)

            if __debug__:
                var.lifetime.name = getattr(var,'name',None)
                var.lifetime.origin = getattr(var,'origin',None)

        if write:
            self.state.apply(var.lifetime,i)
            if isinstance(var,Block):
                for part in var:
                    self.create_var_life(part,_i,False,True)
        elif not read:
            # even if a variable is neither read-from nor written-to, it still
            # has to exist
            var.lifetime.intervals |= Interval(i,i+1)

        if read:
            self.state.read_starts.setdefault(var.lifetime,i + 1)

    def conform_states_to(self,op_i : int,states : Sequence[Tuple[VarState,int]]) -> None:
        assert self.state is not None

        for state in states:
            for life in self.state.read_starts:
                # This variable's lifetime needs to be propagated to a later
                # location. Since we are scanning the code backwards, we have
                # to do this in another pass
                self.back_starts[state[1]].add(life)

class _CalcBackVarIntervals(_CalcVarIntervalsCommon):
    def __init__(self,back_starts):
        super().__init__()
        self.back_starts = back_starts

    def create_var_life(self,var,i,read,write):
        assert self.state

        if isinstance(var,VarPart):
            self.create_var_life(var.block,i,read,False)

        if not read: i += 1

        # the previous pass should have handled it by this point, so don't
        # track it any further
        assert var.lifetime is not None
        self.state.apply(var.lifetime,i,True)

    def handle_op(self,op_i : int,code : IRCode):
        r = super().handle_op(op_i,code)

        lives = self.back_starts.get(op_i)
        if lives:
            assert self.state is not None
            for life in lives:
                self.state.read_starts.setdefault(life,op_i + 1)

        return r

def calc_var_intervals(code : IRCode) -> None:
    c = _CalcVarIntervals()
    c.run(code,len(code))

    # This only checks for read-violations from code that is reachable from the
    # start, which is fine, since code that isn't reachable from the start,
    # isn't reachable at all, and will be stripped in a later pass.
    assert c.state
    if c.state.read_starts:
        raise ValueError('one or more variables may be read-from before being written-to')

    for b in c.block_vars:
        assert b.lifetime
        for a in b.lifetime.aliases:
            b.lifetime.intervals |= a.intervals

    if c.back_starts:
        cb = _CalcBackVarIntervals(c.back_starts)
        cb.run(code,len(code))


class ItvLocation:
    def __init__(self,reg : Optional[int]=None,stack_loc : Optional[StackLocation]=None) -> None:
        self.reg = reg
        self.stack_loc = stack_loc

    def copy(self) -> 'ItvLocation':
        return ItvLocation(self.reg,self.stack_loc)

    def __bool__(self):
        return self.reg is not None or self.stack_loc is not None

    def __repr__(self):
        return 'ItvLocation({!r},{!r})'.format(self.reg,self.stack_loc)

    def __eq__(self,b):
        if isinstance(b,ItvLocation):
            return self.reg == b.reg and self.stack_loc == b.stack_loc

        return NotImplemented

    def __ne__(self,b):
        r = self.__eq__(b)
        return r if r is NotImplemented else not r

    def to_opt_ir(self,data_type : c_types.CType=c_types.t_void_ptr) -> Union[FixedRegister,StackItem,ArgStackItem,None]:
        """Return an IR value representing a value in this location.

        If this location is both a register and stack item, the return value
        will represent a register. If this location is neither, None is
        returned.

        """
        if self.reg is not None:
            return FixedRegister(self.reg,data_type)

        if self.stack_loc is not None:
            return to_stack_item(self.stack_loc,data_type)

        return None

    def to_ir(self,data_type : c_types.CType=c_types.t_void_ptr) -> Union[FixedRegister,StackItem,ArgStackItem]:
        """Return an IR value representing a value in this location.

        If this location is both a register and stack item, the return value
        will represent a register. If this location is neither, a ValueError
        will be raised.

        """
        r = self.to_opt_ir(data_type)
        if r is None:
            raise ValueError('this location is blank')
        return r

LINEAR_SCAN_EXTRA_CHECKS = False

if __debug__ and LINEAR_SCAN_EXTRA_CHECKS:
    def do_consistency_check(self):
        for life,loc in self.itv_locs.items():
            if loc.reg is not None:
                assert self.reg_pool[loc.reg] is life
            if loc.stack_loc is not None and loc.stack_loc.sect == StackSection.local:
                assert self.stack_pool[loc.stack_loc.index] is life
            if loc.reg is None and loc.stack_loc is None:
                assert self.cur_pos not in life.intervals or self.cur_pos == life.intervals.interval_at(self.cur_pos).start
        for itv,life in self.active_r:
            assert self.itv_locs[life].reg is not None
        for itv,life in self.active_s:
            sloc = self.itv_locs[life].stack_loc
            assert sloc is not None and sloc.sect == StackSection.local

    def consistency_check(f):
        def inner(self,*args,**kwds):
            r = f(self,*args,**kwds)
            if not consistency_check.suspend:
                do_consistency_check(self)
            return r
        return inner
    consistency_check.suspend = False
else:
    def consistency_check(f):
        return f

class LocationScan:
    def __init__(self,regs : int,extra : ExtraState) -> None:
        assert regs > 0

        key = lambda x: x[0].end
        self.active_r = SortedList(key=key) # type: SortedList[Tuple[Interval[int],VarLifetime]]
        self.active_s = SortedList(key=key) # type: SortedList[Tuple[Interval[int],VarLifetime]]

        self.reg_pool = [None] * regs # type: List[Optional[VarLifetime]]

        # The stack pool grows as needed. Unlike the register pool, the stack
        # pool is shared between branches and members are not removed until
        # all of their intervals have passed.
        # The stack pool only tracks local stack items. Variables passed by
        # stack will reside in a separate un-tracked area.
        # TODO: are functions allowed to write to the stack space used by
        #       arguments? If so, we should do that.
        self.stack_pool = [] # type: List[Optional[VarLifetime]]

        self.itv_locs = collections.defaultdict(ItvLocation) # type: DefaultDict[VarLifetime,ItvLocation]
        self.cur_pos = 0
        self.extra = extra
        self.locked_regs = set()  # type: Set[int]

    def branch(self) -> 'LocationScan':
        r = LocationScan.__new__(LocationScan)
        r.active_r = self.active_r.copy()
        r.active_s = self.active_s.copy()
        r.reg_pool = self.reg_pool[:]
        r.stack_pool = self.stack_pool # there is only ever one stack pool
        # noinspection PyArgumentList
        r.itv_locs = collections.defaultdict(ItvLocation,((life,loc.copy()) for life,loc in self.itv_locs.items()))
        r.cur_pos = self.cur_pos
        r.extra = self.extra.copy()
        r.locked_regs = self.locked_regs.copy()
        return r

    def _alloc_r(self,life,opts):
        assert self.itv_locs[life].reg is None

        for i in range(len(self.reg_pool)):
            if self.reg_pool[i] is None and i in opts and i not in self.locked_regs:
                self.reg_pool[i] = life
                self.itv_locs[life].reg = i
                self.active_r.add_item((life.itv_at(self.cur_pos),life))
                return True
        return False

    def _move_r(self,life,opts):
        itv_l = self.itv_locs[life]
        assert itv_l.reg is not None

        for i in range(len(self.reg_pool)):
            if self.reg_pool[i] is None and i in opts and i not in self.locked_regs:
                self.reg_pool[itv_l.reg] = None
                self.reg_pool[i] = life
                self.itv_locs[life].reg = i
                return True
        return False

    def _alloc_s(self,life : VarLifetime,itv : Optional[Interval[int]]=None,reserve_only=False) -> bool:
        itvl = self.itv_locs[life]
        if itvl.stack_loc is not None: return False

        if life.preferred_stack_i is not None:
            assert self.stack_pool[life.preferred_stack_i] is life
            if not reserve_only:
                itvl.stack_loc = StackLocation(StackSection.local,life.preferred_stack_i)
        else:
            for i in range(len(self.stack_pool)):
                if self.stack_pool[i] is None:
                    self.stack_pool[i] = life
                    life.preferred_stack_i = i
                    if not reserve_only:
                        itvl.stack_loc = StackLocation(StackSection.local,i)
                    break
            else:
                life.preferred_stack_i = len(self.stack_pool)
                if not reserve_only:
                    itvl.stack_loc = StackLocation(StackSection.local,len(self.stack_pool))
                self.stack_pool.append(life)

        if not reserve_only:
            if itv is None: itv = life.itv_at(self.cur_pos)
            self.active_s.add_item((cast(Interval[int],itv),life))

        return True

    def is_reg_free(self,r):
        return self.reg_pool[r] is None

    def _update_r(self,life,on_loc_expire):
        try:
            new_itv = life.itv_at(self.cur_pos)
        except ValueError:
            loc = self.itv_locs[life]
            assert loc.reg is not None
            self.reg_pool[loc.reg] = None
            loc.reg = None

            # if loc.stack_loc is not None, the next loop will call
            # on_loc_expire for this instance of VarLifetime
            if on_loc_expire and loc.stack_loc is None: on_loc_expire(life)
        else:
            self.active_r.add_item((new_itv,life))

    def _update_s(self,life,on_loc_expire):
        try:
            new_itv = life.itv_at(self.cur_pos)
        except ValueError:
            loc = self.itv_locs[life]
            assert loc is not None
            assert loc.stack_loc is not None
            assert loc.stack_loc.sect == StackSection.local

            loc.stack_loc = None
            if on_loc_expire: on_loc_expire(life)
        else:
            self.active_s.add_item((new_itv,life))

    def _advance(self,pos,on_loc_expire):
        while self.active_r:
            itv,life = self.active_r[0]
            if itv.end > pos: break
            del self.active_r[0]
            self._update_r(life,on_loc_expire)


        while self.active_s:
            itv,life = self.active_s[0]
            if itv.end > pos: break
            del self.active_s[0]
            self._update_s(life,on_loc_expire)

        for i,life in enumerate(self.stack_pool):
            if life is not None and pos >= life.intervals.global_end:
                self.stack_pool[i] = None

    def _reverse(self,pos,on_loc_expire):
        self.cur_pos = pos

        old_a = list(self.active_r)
        del self.active_r[:]
        for itv,life in old_a:
            self._update_r(life,on_loc_expire)

        old_a = list(self.active_s)
        del self.active_s[:]
        for itv,life in old_a:
            self._update_s(life,on_loc_expire)

    @consistency_check
    def advance(self,pos: int,on_loc_expire: Optional[Callable[[VarLifetime],None]]) -> None:
        old_pos = self.cur_pos
        self.cur_pos = pos

        if pos > old_pos:
            self._advance(pos,on_loc_expire)
        elif pos < old_pos:
            self._reverse(pos,on_loc_expire)

    @staticmethod
    def _remove_active(life,active):
        for i,item in enumerate(active):
            if item[1] == life:
                del active[i]
                break

    @consistency_check
    def free_reg(self,reg : int,alt_regs : Sequence[int]=()) -> Optional[VarLifetime]:
        """Free the given register by either moving the value to the stack or
        one of the registers in 'alt_regs' if any are available.

        If a value had to be moved, the return value is an instance of
        VarLifetime, indicating where the value was moved; Otherwise, the
        return value is None.

        """
        assert reg not in alt_regs

        life = self.reg_pool[reg]
        if life is None:
            return None
        life_l = self.itv_locs[life]
        self.reg_pool[reg] = None

        for ar in alt_regs:
            if self.reg_pool[ar] is None and ar not in self.locked_regs:
                life_l.reg = ar
                self.reg_pool[ar] = life
                break
        else:
            self._remove_active(life,self.active_r)
            life_l.reg = None
            if not self._alloc_s(life):
                # the value already had a copy in the stack
                life = None

        return life

    @consistency_check
    def load_reg(self,life : VarLifetime,opts : Container[int]=Filter()) -> Optional[VarLifetime]:
        """Load the given value into a register.

        If another value had to be saved to the stack, to free up a register,
        the return value will be an instance of VarLifetime, indicating where
        the value was saved. Otherwise the return value is None.

        The register will be one that is in 'opts'.

        """
        itv_l = self.itv_locs[life]

        if len(self.active_r) < len(self.reg_pool) and (
            self._alloc_r if itv_l.reg is None else self._move_r)(life,opts):
            return None

        for i in range(len(self.active_r)-1,-1,-1):
            spill = self.active_r[i][1]
            spill_l = self.itv_locs[spill]
            assert spill_l.reg is not None
            if spill_l.reg in opts:
                if itv_l.reg is None:
                    self.active_r.add_item((life.itv_at(self.cur_pos),life))
                else:
                    self.reg_pool[itv_l.reg] = None

                self.reg_pool[spill_l.reg] = life
                itv_l.reg = spill_l.reg
                self._alloc_s(spill)
                spill_l.reg = None
                del self.active_r[i]
                return spill
        assert None

    @consistency_check
    def load_stack(self,life : VarLifetime,loc : StackLocation) -> None:
        """Load the given value into the stack.

        This is only allowed for non-local stack locations, since shuffling
        stack items is not supported.

        Non-local stack items are not tracked. Providing a stack location like
        this will simply prevent creating an extra copy in the local stack if
        the value gets moved to a register and then spilled.

        """
        assert self.itv_locs[life].stack_loc is None and loc.sect != StackSection.local
        self.itv_locs[life].stack_loc = loc

    @consistency_check
    def to_stack(self,life: VarLifetime,reserve_only: bool=False) -> bool:
        """Copy the value to the stack.

        This will place a value onto the stack, if it's not already there. This
        will not free a register.

        """
        if self.itv_locs[life].stack_loc is not None: return False
        self._alloc_s(life,reserve_only=reserve_only)
        return True

    # @consistency_check
    # def create_loc(self,life : VarLifetime) -> Optional[VarLifetime]:
    #     """Assign the value a location"""
    #
    #     assert self.itv_locs[life].reg is None and self.itv_locs[life].stack_loc is None
    #
    #     if len(self.active_r) == len(self.reg_pool):
    #         spill_itv,spill = self.active_r[-1]
    #         spill_l = self.itv_locs[spill]
    #         itv = life.itv_at(self.cur_pos)
    #         if spill_itv.end > itv.end:
    #             self.itv_locs[life].reg = spill_l.reg
    #             self._alloc_s(spill,spill_itv)
    #             spill_l.reg = None
    #             del self.active_r[-1]
    #             self.active_r.add_item((itv,life))
    #             return spill
    #
    #         self._alloc_s(life,itv)
    #     else:
    #         self._alloc_r(life,Filter())
    #
    #     return None

    @consistency_check
    def value_updated(self,life : VarLifetime,where : LocationType) -> None:
        """Indicate that a value was updated.

        This indicates a value was updated either in a register or the stack,
        and that if a value was stored in both, the other location is no longer
        valid.

        """
        itvl = self.itv_locs[life]
        if where == LocationType.register:
            if itvl.reg is None:
                raise ValueError('cannot update the register content of a variable that is not currently in a register')

            if itvl.stack_loc is not None:
                if itvl.stack_loc.sect == StackSection.local:
                    #self._stack_pool_remove(life,itvl.stack_loc.index)
                    self._remove_active(life,self.active_s)
                itvl.stack_loc = None
        else:
            assert where == LocationType.stack
            if itvl.stack_loc is None:
                raise ValueError('cannot update the address content of a variable that is not currently in memory')

            if itvl.reg is not None:
                self.reg_pool[itvl.reg] = None
                self._remove_active(life,self.active_r)
                itvl.reg = None

    @consistency_check
    def alloc_block(self,life: VarLifetime,size: int,reserve_only: bool=False) -> bool:
        itvl = self.itv_locs[life]
        if itvl.stack_loc is not None: return False

        if life.preferred_stack_i is not None:
            assert all(self.stack_pool[life.preferred_stack_i + i] is life for i in range(size))
            if not reserve_only:
                itvl.stack_loc = StackLocation(StackSection.local,life.preferred_stack_i)
        else:
            i = 0
            n = len(self.stack_pool)
            while i < n:
                for j in range(i,min(n,i+size)):
                    if self.stack_pool[j] is not None:
                        i = j + 1
                        break
                else:
                    break

            for j in range(i,i + size):
                if j == len(self.stack_pool):
                    self.stack_pool.append(life)
                else:
                    self.stack_pool[j] = life

            if not reserve_only:
                itvl.stack_loc = StackLocation(StackSection.local,i)
            life.preferred_stack_i = i

        if not reserve_only:
            self.active_s.add_item((life.itv_at(self.cur_pos),life))

        return True

    def interval_loc(self,itv : Lifetime):
        if isinstance(itv,AliasLifetime):
            itv = itv.itv

        assert isinstance(itv,VarLifetime)
        return self.itv_locs[itv]

    def to_ir(self,itv : Lifetime,data_type : c_types.CType=c_types.t_void_ptr) -> Union[FixedRegister,StackItem,ArgStackItem]:
        return self.interval_loc(itv).to_ir(data_type)

def load_to_reg(alloc : LocationScan,itv : VarLifetime,allowed_reg : Container[int],cgen : IROpGen,code : IRCode2,val : Optional[Value]=None) -> FixedRegister:
    displaced = alloc.load_reg(itv,Filter(allowed_reg))
    dest = alloc.interval_loc(itv)
    ir_dest = dest.to_ir()
    if displaced is not None:
        d_loc = alloc.interval_loc(displaced)
        code.extend(ir_preallocated_to_ir2(cgen.move(ir_dest,d_loc.to_ir())))
        code.extend(annotate_symbol_loc(displaced,d_loc))
    if val is not None:
        code.extend(ir_preallocated_to_ir2(cgen.move(val,ir_dest)))
    code.extend(annotate_symbol_loc(itv,dest))

    return ir_dest

def alloc_stack(alloc,var,reserve_only=False):
    if isinstance(var,Block):
        alloc.alloc_block(var.lifetime,len(var),reserve_only)
    elif isinstance(var,VarPart):
        alloc.alloc_block(var.block.lifetime,len(var.block),reserve_only)
    else:
        alloc.to_stack(var.lifetime,reserve_only=reserve_only)

class JumpState:
    def __init__(self,alloc : LocationScan,code : IRCode2,jump_ops : int,branches : bool=False) -> None:
        assert jump_ops <= len(code)
        self.alloc = alloc
        self.code = code
        self.jump_ops = jump_ops
        self.branches = branches

    def extend_code(self,ops : IRCode2):
        if self.jump_ops:
            self.code[-self.jump_ops:-self.jump_ops] = ops
        else:
            self.code.extend(ops)

    def conform_to(self,cgen : IROpGen,other : LocationScan) -> None:
        """Move values so that they have the same locations in self.alloc as
        'other'."""

        # noinspection PyUnresolvedReferences
        for itv in self.alloc.itv_locs.keys() & other.itv_locs.keys():
            loc_self = self.alloc.itv_locs[itv]
            loc_o = other.itv_locs[itv]
            if not (loc_self and loc_o): continue

            if loc_self.reg != loc_o.reg and loc_o.reg is not None:
                displaced = self.alloc.free_reg(loc_o.reg)
                if displaced is not None:
                    d_loc = self.alloc.interval_loc(displaced)
                    self.extend_code(ir_preallocated_to_ir2(
                        cgen.move(loc_o.to_ir(),d_loc.to_ir())))
                    self.extend_code(annotate_symbol_loc(itv,d_loc))
                tmp = [] # type: IRCode2
                load_to_reg(self.alloc,itv,(loc_o.reg,),cgen,tmp,loc_self.to_ir())
                self.extend_code(tmp)

            if loc_self.stack_loc != loc_o.stack_loc:
                if loc_o.stack_loc is None:
                    self.alloc.value_updated(itv,LocationType.register)
                else:
                    # it should never be the case where an interval is placed
                    # in more than one stack location
                    assert loc_self.stack_loc is None
                    self.alloc.to_stack(itv)
                    self.extend_code(ir_preallocated_to_ir2(
                        cgen.move(loc_self.to_ir(),to_stack_item(loc_o.stack_loc))))

            if loc_self.reg != loc_o.reg:
                assert loc_o.reg is None
                assert loc_self.stack_loc is not None
                self.alloc.free_reg(loc_self.reg)

def hassubclass(cls : Union[type,Tuple[type,...]],classinfo : Union[type,Tuple[type,...]]) -> Optional[type]:
    if isinstance(cls,tuple):
        for c in cls:
            if issubclass(c,classinfo): return c
    elif issubclass(cls,classinfo):
        return cls

    return None

def process_indirection(cgen : IROpGen,instr : Instr,overload : Overload) -> Tuple[Instr,Overload]:
    inds = []
    for i,ta in enumerate(zip(overload.params,instr.args)):
        if hassubclass(ta[0],AddressType) and isinstance(ta[1],IndirectVar):
            inds.append(i)
    if inds:
        instr,overload = cgen.process_indirection(instr,overload,inds)

    return instr,overload

def ir_preallocated_to_ir2(code):
    r = []
    for instr in code:
        if isinstance(instr,Instr):
            r.append(instr.op.to_ir2(instr.args))
        else:
            assert isinstance(instr,Target)
            r.append(instr)

    return r

class ListChainLink(List[T]):
    """A basic singly-linked list of arrays"""
    def __init__(self):
        super().__init__()
        self.next = None # type: Optional[ListChainLink[T]]

    def new_link(self):
        r = ListChainLink()
        r.next = self.next
        self.next = r
        return r

class _RegAllocate(FollowNonlinear[LocationScan,JumpState]):
    def __init__(self,cgen : IROpGen,alloc : LocationScan) -> None:
        super().__init__(alloc)
        self.cgen = cgen
        self.stack_pool = alloc.stack_pool

        self.new_code = ListChainLink() # type: ListChainLink[IROp2]
        self.new_code_head = self.new_code # type: ListChainLink[IROp2]
        self.elided_links = {} # type: Dict[int,ListChainLink[IROp2]]

        self.active_symbols = set() # type: Set[str]

    @property
    def _ptr_size(self):
        return self.cgen.abi.ptr_size

    def _load_to_block(self,block):
        assert self.state is not None
        assert block.lifetime is not None
        self.state.alloc_block(block.lifetime,len(block))
        return self.state.to_ir(block.lifetime,block.data_type)

    def _copy_val_to_stack(self,state,val,itv,code):
        moved = state.to_stack(itv)
        d_loc = state.interval_loc(itv)
        dest = to_stack_item(d_loc.stack_loc)
        if moved and code is not None:
            if isinstance(val,Var):
                assert d_loc.reg is not None
                val = FixedRegister(d_loc.reg)
            code.extend(ir_preallocated_to_ir2(self.cgen.move(val,dest)))
            code.extend(annotate_symbol_loc(itv,d_loc))

        return dest

    def to_addr(self,val,life,reads):
        if isinstance(val,Block):
            assert not reads
            return self._load_to_block(val)

        if isinstance(val,VarPart):
            assert not reads
            dest = self._load_to_block(val.block)
            assert isinstance(dest,StackItem)
            return StackItemPart(dest,val.offset * self._ptr_size,val.data_type)

        assert isinstance(life,VarLifetime)
        return self._copy_val_to_stack(self.state,val,life,self.new_code_head if reads else None)

    def to_reg(self,val,life,reads,allowed_regs=Filter()):
        assert not isinstance(val,(Block,VarPart)),(
            "'Block' and 'VarPart' instances cannot be put in registers")
        assert isinstance(life,VarLifetime)
        return load_to_reg(
            self.state,
            life,
            allowed_regs,
            self.cgen,
            self.new_code_head,
            val if reads else None)

    # if non_var_moves is not None, the value is to be written-to
    def move_for_param(self,i,p,new_args,reads,non_var_moves=None):
        assert self.state is not None

        arg = new_args[i]

        if not isinstance(arg,p):
            if isinstance(arg,Var):
                itv = arg.lifetime
            else:
                itv = VarLifetime()
                itv.intervals |= Interval(self.state.cur_pos,self.state.cur_pos+1)

                if __debug__:
                    itv.origin = 'temporary lifetime'

            fr_type = hassubclass(p,FixedRegister)
            if fr_type:
                dest = self.to_reg(arg,itv,reads,cast(_RegisterMetaType,fr_type).allowed)
            else:
                assert hassubclass(p,AddressType)
                dest = self.to_addr(arg,itv,reads)

            if non_var_moves is not None and not isinstance(arg,Var):
                if not isinstance(arg,MutableValue):
                    raise ValueError('cannot write to a read-only value')
                non_var_moves.append((arg,dest))
            new_args[i] = dest

    def get_var_loc(self,var):
        assert var.lifetime is not None
        itvl = self.state.interval_loc(var.lifetime)

        if itvl.reg is not None:
            return FixedRegister(itvl.reg)

        assert itvl.stack_loc is not None
        if isinstance(var,VarPart):
            assert itvl.stack_loc.sect == StackSection.local
            return StackItemPart(StackItem(itvl.stack_loc.index,var.block.data_type),var.offset*self._ptr_size,var.data_type)
        if isinstance(var,Block):
            return StackItem(itvl.stack_loc.index,var.data_type)

        return to_stack_item(itvl.stack_loc)

    def _on_loc_expire(self,life):
        self.new_code_head.extend(annotate_symbol_loc(life,ItvLocation()))

    def _supporting_instr(self,code : IRCode):
        for op in code:
            if isinstance(op,Instr):
                op,overload = process_indirection(self.cgen,op,op.op.exact_match(op.args))
                new_args = [] # type: List[Any]
                for i,a in enumerate(op.args):
                    new_args.append(a)
                    if isinstance(a,Block):
                        new_args[i] = self._load_to_block(a)
                    elif isinstance(a,Var):
                        new_args[i] = self.get_var_loc(a)
                processed = Instr2(op.op,overload,new_args) # type: IROp2
            else:
                assert isinstance(op,Target)
                processed = op

            self.new_code_head.append(processed)

    def handle_instr(self,op_i : int,op : Instr):
        assert self.state is not None

        tmp = self.state.extra.process_instr(op_i,op)
        if tmp is None: return
        op = tmp

        new_args = []  # type: List[Any]

        # First pass: don't actually do anything, just get the locations of the
        # arguments to pick the best overload.
        for i,a in enumerate(op.args):
            assert not isinstance(a,StackItem)
            pd = op.op.param_dirs[i]

            new_args.append(a)

            if not pd.reads:
                continue

            if isinstance(a,(Var,Block)):
                new_args[i] = self.get_var_loc(a)

        op,overload = process_indirection(self.cgen,op,op.op.best_match(new_args))

        new_args = []
        non_var_moves = [] # type: List[Any]

        # Second pass: every argument we read-from should already have a
        # location. Some arguments will need to be moved. Arguments that are
        # neither read-from nor written-to (as in the case of the first
        # argument of x86's 'lea') will still need a location.
        for i,a in enumerate(op.args):
            pd = op.op.param_dirs[i]

            new_args.append(a)

            if isinstance(a,FixedRegister):
                assert pd.writes,"trying to read from a specific register without using CreateVar is probably a mistake"
                displaced = self.state.free_reg(a.reg_index)
                if displaced is not None:
                    d_loc = self.state.interval_loc(displaced)
                    self.new_code_head.extend(ir_preallocated_to_ir2(
                        self.cgen.move(a,d_loc.to_ir())))
                    self.new_code_head.extend(annotate_symbol_loc(displaced,d_loc))
                continue
            if not pd.reads:
                if not pd.writes:
                    assert hassubclass(overload.params[i],(AddressType,Target)),(
                        'only addresses can be reserved')

                    if isinstance(a,Block):
                        self.state.alloc_block(a.lifetime,len(a),True)
                        new_args[i] = StackItem(a.lifetime.preferred_stack_i,a.data_type)
                    elif isinstance(a,VarPart):
                        self.state.alloc_block(a.block.lifetime,len(a.block),True)
                        new_args[i] = StackItemPart(
                            StackItem(a.block.lifetime.preferred_stack_i,a.block.data_type),
                            a.offset * self._ptr_size,
                            a.data_type)
                    elif isinstance(a,Var):
                        assert isinstance(a.lifetime,VarLifetime)
                        self.state.to_stack(a.lifetime,True)
                        new_args[i] = StackItem(a.lifetime.preferred_stack_i,a.data_type)
                    elif not isinstance(a,(AddressType,Target)):
                        raise TypeError('can only reserve address for variables and address values')
                continue

            if isinstance(a,Block):
                new_args[i] = self._load_to_block(a)
            elif isinstance(a,Var):
                new_args[i] = self.get_var_loc(a)

            self.move_for_param(i,overload.params[i],new_args,True,non_var_moves if pd.writes else None)


        self.state.advance(op_i + 1,self._on_loc_expire)

        # Third pass: for the rest of the arguments, create a physical location
        for i,p in enumerate(overload.params):
            pd = op.op.param_dirs[i]

            if pd.writes:
                if not pd.reads:
                    self.move_for_param(i,p,new_args,False,non_var_moves)

                arg = op.args[i]

                if isinstance(arg,Var) and not isinstance(arg,VarPart):
                    assert isinstance(arg.lifetime,VarLifetime)

                    if hassubclass(p,FixedRegister):
                        self.state.value_updated(arg.lifetime,LocationType.register)
                    elif hassubclass(p,AddressType):
                        self.state.value_updated(arg.lifetime,LocationType.stack)

        self.new_code_head.append(Instr2(op.op,overload,new_args))

        # arguments written-to that represent specific locations need to be
        # actually written to those locations
        for src,dest in non_var_moves:
            self._supporting_instr(self.cgen.move(dest,src))

    def handle_invalidate_regs(self,op_i : int,op : InvalidateRegs):
        assert self.state is not None

        for r in op.to_free:
            displaced = self.state.free_reg(r,op.pres)
            if displaced is not None:
                disp_l = self.state.interval_loc(displaced)
                self.new_code_head.extend(
                    ir_preallocated_to_ir2(
                        self.cgen.move(FixedRegister(r),disp_l.to_ir())))
                self.new_code_head.extend(annotate_symbol_loc(displaced,disp_l))

    def handle_create_var(self,op_i : int,op : CreateVar):
        assert self.state is not None
        assert isinstance(op.var.lifetime,VarLifetime)

        if __debug__ and LINEAR_SCAN_EXTRA_CHECKS:
            consistency_check.suspend = True

        # variables' lifetimes always start after the instruction
        self.state.advance(op_i + 1,self._on_loc_expire)

        if isinstance(op.val,FixedRegister):
            assert self.state.is_reg_free(op.val.reg_index)
            self.state.load_reg(op.var.lifetime,Filter((op.val.reg_index,)))
        else:
            assert isinstance(op.val,ArgStackItem)
            self.state.load_stack(
                op.var.lifetime,
                StackLocation(
                    StackSection.previous if op.val.prev_frame else StackSection.args,
                    op.val.index))
        self.new_code_head.extend(annotate_symbol_loc(op.var.lifetime,self.state.interval_loc(op.var.lifetime)))

        if __debug__ and LINEAR_SCAN_EXTRA_CHECKS:
            consistency_check.suspend = False
            do_consistency_check(self.state)

    def handle_indirect_mod(self,op_i : int,op : IndirectMod):
        assert self.state is not None and (op.read or op.write)
        assert isinstance(op.var.lifetime,VarLifetime)

        if op.write:
            # variables' lifetimes always start after the instruction
            self.state.advance(op_i + 1,self._on_loc_expire)

        loc = self.state.interval_loc(op.var.lifetime)
        if op.loc_type == LocationType.stack:
            assert loc.reg is not None or not op.read
            if loc.stack_loc is None:
                self.to_addr(op.var,op.var.lifetime,op.read)
        else:
            assert op.loc_type == LocationType.register
            assert loc.stack_loc is not None or not op.read
            if loc.reg is None:
                self.to_reg(op.var,op.var.lifetime,op.read)

        if op.write:
            self.state.value_updated(op.var.lifetime,op.loc_type)

    def handle_target(self,op_i : int,op : Target):
        prev = self.elided_links.get(op_i)
        if prev is not None:
            assert self.backtracking(op_i)
            self.new_code_head = prev

        self.new_code_head.append(op)

    def handle_elided_target(self,op_i : int,op : Target):
        if not self.backtracking(op_i):
            self.elided_links[op_i] = self.new_code_head
            self.new_code_head = self.new_code_head.new_link()

    def handle_lock_regs(self,op_i : int,op : LockRegs):
        assert self.state is not None
        self.state.locked_regs.update(op.regs)

    def handle_unlock_regs(self,op_i : int,op : UnlockRegs):
        assert self.state is not None
        self.state.locked_regs.difference_update(op.regs)

    def conform_states_to(self,op_i : int,states : Sequence[JumpState]):
        assert self.state is not None and len(states)

        cond_states = [s for s in states if s.branches]

        if len(cond_states) == 1:
            JumpState(self.state,self.new_code_head,0).conform_to(self.cgen,states[0].alloc)
        elif cond_states:
            # for conditional jumps (including jumps with more than one
            # possible target), we can't move variables to different registers,
            # but because every variable gets only one stack position, we can
            # copy variables to the stack, to resolve variables being in
            # different places

            bad_vars = set() # type: Set[VarLifetime]
            for state in cond_states:
                for life,loc in self.state.itv_locs.items():
                    if loc:
                        loc_b = state.alloc.itv_locs.get(life)
                        if loc_b and loc != loc_b:
                            bad_vars.add(life)
                            self._copy_val_to_stack(self.state,loc.to_ir(),life,self.new_code_head)

            for state in cond_states:
                for life in bad_vars:
                    self._copy_val_to_stack(state.alloc,state.alloc.to_ir(life),life,state.code)

        # unconditional jumps are safe to modify
        for state in states:
            if not state.branches:
                state.alloc.advance(self.state.cur_pos,self._on_loc_expire)
                state.conform_to(self.cgen,self.state)

    def conform_prior_states(self,op_i : int,jump_ops : int,states : Sequence[LocationScan]):
        assert self.state is not None
        assert len(states)
        if len(states) > 1:
            raise ValueError('a jump cannot have multiple backward targets unless the jump is the only way to reach the targets')
        JumpState(self.state,self.new_code_head,jump_ops).conform_to(self.cgen,states[0])

    def make_prior_state(self,op_i : int) -> LocationScan:
        assert self.state is not None
        return self.state.branch()

    def make_pending_state(self,op_i : int,op : IRJump):
        assert self.state is not None

        r = JumpState(self.state.branch(),self.new_code_head,op.jump_ops,op.conditional or len(op.dests) > 1)
        self.new_code_head = self.new_code_head.new_link()

        return r

    def recall_pending_state(self,op_i : int,state : JumpState):
        r = state.alloc.branch()
        r.advance(op_i,self._on_loc_expire)
        return r

    def after_state_change(self,old_state):
        if not debug.GDB_JIT_SUPPORT: return

        if old_state:
            for itv,loc in old_state.itv_locs.items():
                if loc:
                    self.new_code_head.extend(annotate_symbol_loc(itv,ItvLocation()))

        if self.state:
            for itv,loc in self.state.itv_locs.items():
                if loc:
                    self.new_code_head.extend(annotate_symbol_loc(itv,loc))

    def handle_op(self,op_i : int,code : IRCode):
        if self.state is not None:
            self.state.advance(op_i,self._on_loc_expire)

        return super().handle_op(op_i,code)

    def run(self,code : IRCode,start : int=-1):
        prev_head = self.new_code_head
        try:
            super().run(code,start)
        finally:
            self.new_code_head = prev_head

def reg_allocate(cgen : IROpGen,code : IRCode,regs : int) -> Tuple[IRCode2,Set[int],int]:
    """Convert IRCode into IRCode2.

    This converts all instances of Var into FixedRegister or StackItem, and
    adds extra instructions to shuffle values between registers and the stack,
    as needed.

    As a side-effect, this will also remove unreachable code.

    """
    calc_var_intervals(code)
    ls = LocationScan(regs,cgen.allocater_extra_state())
    stack_pool = ls.stack_pool
    allocater = _RegAllocate(cgen,ls)
    allocater.run(code)

    # TODO: return actual registers used
    code2 = [] # type: IRCode2
    nc = allocater.new_code # type: Optional[ListChainLink[IROp2]]
    while nc is not None:
        code2.extend(nc)
        nc = nc.next
    return code2,set(range(regs)),len(stack_pool)


def code_join(x):
    if isinstance(x[0],bytes):
        return b''.join(x)
    return reduce(operator.add,x)

class AsmOp:
    __slots__ = 'op','args','binary','annot'

    def __init__(self,op,args,binary,annot=''):
        self.op = op
        self.args = args
        self.binary = binary
        self.annot = annot

    def __len__(self):
        return len(self.binary)

    def emit(self,addr):
        return self.op.assembly(self.args,addr,self.binary,self.annot)

    @property
    def inline_comment(self):
        return isinstance(self.op,CommentDesc) and self.op.inline

class AsmSequence(Sized):
    def __init__(self,ops=None):
        self.ops = ops or []

    def __len__(self):
        return sum((len(op.binary) for op in self.ops),0)

    def __add__(self,b):
        if isinstance(b,AsmSequence):
            return AsmSequence(self.ops+b.ops)

        return NotImplemented

    def __iadd__(self,b):
        if isinstance(b,AsmSequence):
            self.ops += b.ops
            return self

        return NotImplemented

    def __mul__(self,b):
        if isinstance(b,int):
            return AsmSequence(self.ops*b)

        return NotImplemented

    def __imul__(self,b):
        if isinstance(b,int):
            self.ops *= b
            return self

        return NotImplemented

    def emit(self,base=0):
        lines = []
        addr = base
        for op in self.ops:
            line = op.emit(addr)
            if op.inline_comment:
                assert lines
                lines[-1] = ' '.join((lines[-1],line))
            else:
                lines.append(line)
            addr += len(op)

        return '\n'.join(lines)

def asm_converter(op,f):
    return lambda *args: AsmSequence([AsmOp(op,args,f(*args))])

def make_asm_if_needed(instr,assembly):
    if assembly:
        return Instr2(instr.op,instr.overload.variant(
            instr.overload.params,
            asm_converter(instr.op,instr.overload.func)),instr.args)

    return instr


class Tuning:
    prefer_addsub_over_incdec = True
    build_seq_loop_threshhold = 5
    unpack_seq_loop_threshhold = 5
    build_set_loop_threshhold = 5
    mem_copy_loop_threshhold = 9


class AbiRegister:
    pass

cc_T = TypeVar('cc_T',bound=AbiRegister)
class CallingConvention(Generic[cc_T]):
    def __init__(self,r_ret: cc_T,r_pres: List[cc_T],r_scratch: List[cc_T],r_arg: List[cc_T],shadow: bool=False) -> None:
        self.r_ret = r_ret
        self.r_pres = r_pres
        self.r_scratch = r_scratch
        self.r_arg = r_arg

        # If True, stack space is reserved for function calls for all
        # arguments, even those that are passed in registers, such that the
        # called function could move those arguments to where they would be if
        # all arguments were passed by stack in the first place.
        self.shadow = shadow

    def get_arg(self,abi : 'BinaryAbi',i : int,prev_frame=False) -> MutableValue:
        if i < len(self.r_arg):
            return abi.reg_to_ir(self.r_arg[i])

        if not self.shadow:
            i -= len(self.r_arg)
        return ArgStackItem(i,prev_frame)

    def get_return(self,abi : 'BinaryAbi',prev_frame=False) -> MutableValue:
        return abi.reg_to_ir(self.r_ret)

class Abi:
    code_gen = None # type: Type[OpGen]

    def __init__(self,*,assembly=False):
        self.assembly = assembly
        self.tuning = Tuning()

class BinAbiMeta(type):
    def __new__(mcs,name,bases,namespace):
        r = type.__new__(mcs,name,bases,namespace) # type: Type[BinaryAbi]

        r.reg_indices = {r:i for i,r in enumerate(r.all_regs)}

        return r

class BinaryAbi(Abi,metaclass=BinAbiMeta):
    code_gen = None # type: Type[IROpGen]
    has_cmovecc = False

    callconvs = None # type: Tuple[CallingConvention,CallingConvention]

    # registers should be ordered by preferred usage, in decreasing order
    all_regs = [] # type: List[AbiRegister]

    # The number of general-purpose registers. These must be located at the
    # front of "all_regs".
    gen_regs = 0

    # this is filled automatically by the metaclass
    reg_indices = {} # type: Dict[AbiRegister,int]

    ptr_size = 0
    char_size = 0
    short_size = 0
    int_size = 0
    long_size = 0

    @classmethod
    def reg_to_ir(cls,reg):
        return FixedRegister(cls.reg_indices[reg])


def debug_gen_code(code):
    """Convert code into test source code for tests/test_intermediate.py"""

    class Namer:
        def __init__(self,pre):
            self.pre = pre
            self.count = 0

        def __call__(self):
            r = self.pre + str(self.count)
            self.count += 1
            return r

    def convert_val(x):
        if isinstance(x,VarPart):
            return '{}[{}]'.format(blocks[x.block],x.offset)
        if isinstance(x,Var):
            return vars_[x]
        if isinstance(x,Target):
            return targets[x]
        if isinstance(x,Block):
            return blocks[x]
        if isinstance(x,IndirectVar):
            return 'ir.IndirectVar({},{},{},{})'.format(
                x.offset,
                convert_val(x.base),
                convert_val(x.index),
                x.scale)
        if isinstance(x,ArgStackItem):
            return 'ir.'+repr(x)

        return None

    targets = collections.defaultdict(Namer('target_'))
    vars_ = collections.defaultdict(Namer('var_'))
    blocks = collections.defaultdict(Namer('block_'))
    code_str = []

    for instr in code:
        if isinstance(instr,Instr):
            for pd,a in zip(instr.op.param_dirs,instr.args):
                if isinstance(a,Target): continue

                arg = convert_val(a)
                if arg is None: continue

                if pd.writes:
                    if pd.reads:
                        op = 'readwrite_op'
                    else:
                        op = 'write_op'
                elif pd.writes:
                    op = 'read_op'
                else:
                    op = 'lea_op'

                line = 'ir.Instr({},{})'.format(op,arg)
                if code_str and line != code_str[-1]:
                    code_str.append(line)
        elif isinstance(instr,Target):
            code_str.append(targets[instr])
        elif isinstance(instr,IRJump):
            code_str.append('ir.IRJump([{}],{!r},0)'.format(','.join(targets[d] for d in instr.dests),instr.conditional))
        elif isinstance(instr,IndirectMod):
            code_str.append('ir.IndirectMod({},{!r},{!r},ir.LocationType.{})'.format(
                convert_val(instr.var),
                instr.read,
                instr.write,
                'register' if instr.loc_type == LocationType.register else 'stack'))
        elif isinstance(instr,CreateVar):
            code_str.append('ir.CreateVar({},ir.{!r})'.format(convert_val(instr.var),instr.val))
        elif isinstance(instr,(LockRegs,UnlockRegs,InvalidateRegs)):
            code_str.append('ir.{!r}'.format(instr))

    r = []
    for name in targets.values():
        r.append('        {} = ir.Target()'.format(name))

    for name in vars_.values():
        r.append("        {0} = ir.Var('{0}')".format(name))

    for b,name in blocks.items():
        r.append('        {} = ir.Block({})'.format(name,len(b.parts)))

    r.append('        code = [')
    for i,c in enumerate(code_str):
        comma = ','
        if i == len(code_str) - 1: comma = ']'
        r.append('            {}{}'.format(c,comma))

    return '\n'.join(r)
