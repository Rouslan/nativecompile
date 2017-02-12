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
    'Target','Value','MutableValue','Var','Block','VarPart','IndirectVar',
    'Immediate','FixedRegister','ArgStackItem','ensure_same_size','Instr',
    'AddressType','ParamDir','Overload','RegAllocatorOverloads',
    'OpDescription','CommentDesc','comment_desc','inline_comment_desc',
    'InvalidateRegs','CreateVar','IRJump','IndirectMod','LockRegs',
    'UnlockRegs','IRAnnotation','annotate','IRSymbolLocDescr','Instr2',
    'CmpType','Cmp','BinCmp','AndCmp','OrCmp','OpType','commutative_ops',
    'UnaryOpType','ShiftDir','IROp','IRCode','IROp2','IRCode2','ExtraState',
    'OpGen','JumpCondOpGen','IRCompiler','reg_allocate']

import enum
import collections
import functools
import operator
import weakref
import binascii
import copy
from typing import (Any,Callable,cast,Container,DefaultDict,Dict,Generic,
    Iterable,List,Optional,NamedTuple,NewType,Sequence,Sized,Set,Tuple,
    TypeVar,Union)

if __debug__:
    import sys

from . import abi
from . import debug
from .sorted_list import SortedList
from .dinterval import *


SIZE_DEFAULT = 0
SIZE_B = 1
SIZE_W = 2
SIZE_D = 4
SIZE_Q = 8


T = TypeVar('T')
U = TypeVar('U')

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


class _RegisterMetaType(type):
    def __new__(mcs,name,bases,namespace,*,allowed=None):
        cls = super().__new__(mcs,name,bases,namespace)
        cls.allowed = frozenset(allowed) if allowed is not None else None
        return cls

    # noinspection PyUnusedLocal
    def __init__(cls,name,bases,namespace,**kwds):
        super().__init__(name,bases,namespace)

    def __getitem__(cls,allowed):
        if allowed is None: return FixedRegister
        return _RegisterMetaType('DependentFixedRegister',(FixedRegister,),{},allowed=allowed)

    def __instancecheck__(cls,inst):
        return issubclass(inst.__class__,FixedRegister) and (
            cls.allowed is None or inst.reg_index in cls.allowed)

    @staticmethod
    def generic_type():
        return FixedRegister

class AddressType:
    @staticmethod
    def generic_type():
        return AddressType

class _ImmediateMetaType(type):
    def __new__(mcs,name,bases,namespace,*,allowed_range=None):
        cls = super().__new__(mcs,name,bases,namespace)
        cls.allowed = allowed_range
        return cls

    # noinspection PyUnusedLocal
    def __init__(cls,name,bases,namespace,**kwds):
        super().__init__(name,bases,namespace)

    def __getitem__(cls,allowed_range):
        return _ImmediateMetaType('DependentImmediate',(Immediate,),{},allowed_range=allowed_range)

    def __instancecheck__(cls,inst):
        return issubclass(inst.__class__,Immediate) and (cls.allowed is None
            or cls.allowed[0] <= inst.val <= cls.allowed[1])

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


class Target:
    __slots__ = 'displacement',

    def __init__(self):
        self.displacement = None # type: Optional[int]

    def __repr__(self):
        return '<Target {:#x}>'.format(id(self))

class Value:
    __slots__ = 'size',
    def __init__(self,size : int=SIZE_DEFAULT) -> None:
        self.size = size

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

    def __init__(self,name : Optional[str]=None,size : int = SIZE_DEFAULT,lifetime : Optional[Lifetime]=None,dbg_symbol : Optional[str]=None) -> None:
        super().__init__(size)
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

    def __init__(self,parts : int,*,lifetime : Optional[VarLifetime]=None) -> None:
        assert parts > 0
        super().__init__(0)
        self.parts = [VarPart(self,i) for i in range(parts)]
        self.lifetime = lifetime

    def __getitem__(self,i):
        return self.parts[i]

    def __len__(self):
        return len(self.parts)

class VarPart(Var):
    __slots__ = 'block','offset'

    def __init__(self,block : Block,offset : int,size : int=SIZE_DEFAULT,*,lifetime : Optional[AliasLifetime]=None) -> None:
        super().__init__(None,size,lifetime)
        #self._block = weakref.ref(block)
        self.block = block
        self.offset = offset

    #@property
    #def block(self) -> Block:
    #    r = self._block()
    #    assert isinstance(r,Block)
    #    return r

class Immediate(Value,metaclass=_ImmediateMetaType):
    __slots__ = 'val','size'

    def __init__(self,val : int,size : int=SIZE_DEFAULT) -> None:
        super().__init__(size)
        self.val = val

    def __eq__(self,b):
        if isinstance(b,Immediate):
            return self.val == b.val and self.size == b.size
        if isinstance(b,int):
            return self.size == SIZE_DEFAULT and self.val == b
        return False

    def __ne__(self,b):
        return not self.__eq__(b)

    def __repr__(self):
        return 'Immediate({},{})'.format(self.val,self.size)

class IndirectVar(MutableValue,AddressType):
    __slots__ = 'offset','base','index','scale'

    def __init__(self,offset : int=0,base : Optional[Var]=None,index : Optional[Var]=None,scale : int=1,size : int=SIZE_DEFAULT) -> None:
        super().__init__(size)
        self.offset = offset
        self.base = base
        self.index = index
        self.scale = scale

    def __eq__(self,b):
        if isinstance(b,IndirectVar):
            return (self.offset == b.offset
                and self.base == b.base
                and self.index == b.index
                and self.scale == b.scale
                and self.size == b.size)
        return False

    def __ne__(self,b):
        return not self.__eq__(b)

    def __repr__(self):
        return 'IndirectVar({},{!r},{!r},{},{})'.format(self.offset,self.base,self.index,self.scale,self.size)

class FixedRegister(MutableValue,metaclass=_RegisterMetaType):
    __slots__ = 'reg_index',

    def __init__(self,reg_index : int,size : int=SIZE_DEFAULT) -> None:
        super().__init__(size)
        self.reg_index = reg_index

    def __repr__(self):
        return 'FixedRegister({},{})'.format(self.reg_index,self.size)

class StackItem(MutableValue,AddressType):
    __slots__ = 'index','offset'

    def __init__(self,index : int,size : int=SIZE_DEFAULT,offset : int=0) -> None:
        super().__init__(size)
        self.index = index
        self.offset = offset

    def __repr__(self):
        size_str = ''
        if self.size != SIZE_DEFAULT:
            size_str = ','+str(self.size)
        offset_str = ''
        if self.offset:
            offset_str = ','+str(self.offset)
        return 'StackItem({}{}{})'.format(self.index,size_str,offset_str)

class ArgStackItem(MutableValue,AddressType):
    """A function argument passed via the stack.

    'index' refers to how many pointer-size increments the argument is from the
    top of the stack.

    If 'prev_frame' is true, the argument is from the previous stack. In other
    words, the argument was passed to the current function, not a function
    about to be called.

    """
    __slots__ = 'index','prev_frame'

    def __init__(self,index : int,prev_frame : bool=False,size : int=SIZE_DEFAULT) -> None:
        super().__init__(size)
        self.index = index
        self.prev_frame = prev_frame

    def __eq__(self,b):
        if isinstance(b,ArgStackItem):
            return self.index == b.index and self.prev_frame == b.prev_frame and self.size == b.size
        return False

    def __ne__(self,b):
        return not self.__eq__(b)

    def __repr__(self):
        size_str = ''
        if self.size != SIZE_DEFAULT:
            size_str = ',' + str(self.size)
        return 'ArgStackItem({},{!r}{})'.format(self.index,self.prev_frame,size_str)

def to_stack_item(location : StackLocation,size=SIZE_DEFAULT) -> Union[StackItem,ArgStackItem]:
    if location.sect == StackSection.local:
        return StackItem(location.index,size=size)
    if location.sect == StackSection.args:
        return ArgStackItem(location.index,size=size)

    assert location.sect == StackSection.previous
    return ArgStackItem(location.index,True,size=size)

def ensure_same_size(ptr_size,a,*args):
    sa = a.size
    for b in args:
        if (b.size or ptr_size) != (sa or ptr_size):
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
    def __init__(self,reads : bool,writes : bool) -> None:
        self.reads = reads
        self.writes = writes

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
        elif not isinstance(a,p):
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
        raise NotImplementedError()

    def to_ir2(self,args : Sequence) -> 'Instr2':
        raise NotImplementedError()

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

    def to_ir2(self,args):
        return Instr2(self,self.exact_match(args),args)

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

    def best_match(self,args : Sequence) -> Overload:
        if len(args) != 1:
            raise TypeError('there must be exactly one argument')

        return self._sole_overload

    def assembly(self,args : Sequence,addr : int,binary: bytes,annot : Optional[str] = None):
        return '; {} {}'.format(args[0],annot or '')

    def to_ir2(self,args : Sequence):
        return Instr2(self,self.best_match(args),args)

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
    should be inserted directly behind.

    This is used by the register allocator to determine value life-times.

    """
    __slots__ = 'dest','conditional','jump_ops'

    def __init__(self,dest : Target,conditional : bool,jump_ops : int) -> None:
        self.dest = dest
        self.conditional = conditional
        self.jump_ops = jump_ops

    def __repr__(self):
        return 'IRJump({!r},{!r},{})'.format(self.dest,self.conditional,self.jump_ops)

class IndirectMod:
    """Indicates that a variable was updated externally, somehow.

    If the value exists in both a register and the stack, one is made invalid
    depending on which was updated.

    """
    __slots__ = 'var','loc_type'

    def __init__(self,var : Var,loc_type : LocationType) -> None:
        self.var = var
        self.loc_type = loc_type

    def __repr__(self):
        return 'IndirectMod({!r},{!r})'.format(self.var,self.loc_type)

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
        self.loc = copy.copy(loc)

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
    def prolog(self) -> IRCode2:
        raise NotImplementedError()

    def epilog(self) -> IRCode2:
        raise NotImplementedError()

    def get_reg(self,index: int,size : int) -> Any:
        raise NotImplementedError()

    def get_stack_addr(self,index : int,size : int,sect : StackSection) -> Any:
        raise NotImplementedError()

    def get_displacement(self,amount : int,force_wide : bool) -> Any:
        raise NotImplementedError()

    def get_immediate(self,val : int,size : int) -> Any:
        raise NotImplementedError()

    def get_machine_arg(self,arg,displacement):
        if isinstance(arg,FixedRegister):
            return self.get_reg(arg.reg_index,arg.size)
        if isinstance(arg,StackItem):
            return self.get_stack_addr(arg.index,arg.size,StackSection.local)
        if isinstance(arg,ArgStackItem):
            return self.get_stack_addr(
                arg.index,
                arg.size,
                StackSection.previous if arg.prev_frame else StackSection.args)
        if isinstance(arg,Target):
            assert arg.displacement is not None
            return self.get_displacement(arg.displacement - displacement,False)
        if isinstance(arg,Immediate):
            return self.get_immediate(arg.val,arg.size)

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

class OpGen:
    max_args_used = 0

    def __init__(self,abi : abi.Abi) -> None:
        self.abi = abi

    def bin_op(self,a : Value,b : Value,dest : MutableValue,op_type : OpType) -> IRCode:
        raise NotImplementedError()

    def unary_op(self,a : Value,dest : MutableValue,op_type : UnaryOpType) -> IRCode:
        raise NotImplementedError()

    def load_addr(self,addr : IndirectVar,dest : MutableValue) -> IRCode:
        raise NotImplementedError()

    def call(self,func,args : Sequence[Value]=(),store_ret : Optional[Var]=None) -> IRCode:
        self.max_args_used = max(self.max_args_used,len(args))
        return self._call_impl(func,args,store_ret)

    def _call_impl(self,func,args : Sequence[Value],store_ret : Optional[Var]) -> IRCode:
        raise NotImplementedError()

    def jump(self,dest : Target) -> IRCode:
        raise NotImplementedError()

    def jump_if(self,dest : Target,cond : Cmp) -> IRCode:
        raise NotImplementedError()

    def if_(self,cond : Cmp,on_true : IRCode,on_false : Optional[IRCode]) -> IRCode:
        raise NotImplementedError()

    def do_while(self,action : IRCode,cond : Cmp) -> IRCode:
        raise NotImplementedError()

    def jump_table(self,val : Value,targets : Sequence[Target]) -> IRCode:
        raise NotImplementedError()

    def move(self,src : Value,dest : MutableValue) -> IRCode:
        raise NotImplementedError()

    def shift(self,src : Value,shift_dir : ShiftDir,amount : Value,dest : MutableValue) -> IRCode:
        raise NotImplementedError()

    def return_value(self,v : Value) -> IRCode:
        raise NotImplementedError()

    def get_compiler(self,regs_used : int,stack_used : int,args_used : int) -> IRCompiler:
        raise NotImplementedError()

    def get_cur_func_arg(self,i : int) -> Value:
        raise NotImplementedError()

    def get_return_address(self,v : Value) -> IRCode:
        """Store the address of the next instruction to be called after the
        current function returns, into 'v'.

        If the address was pushed to the stack, this will pop it and restore
        the stack pointer to what it was before the current function was
        called, as long as the stack isn't altered between the function call
        and the current point of execution.

        """
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

    def process_indirection(self,instr : Instr,ov : Overload,inds : Sequence[int]) -> Tuple[Instr,Overload]:
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

# noinspection PyAbstractClass
class JumpCondOpGen(OpGen):
    """An implementation of OpGen that uses jumps for control flow"""

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
    """Follow 'code' non-linearly by splitting and merging state with the
    branches.

    Note that back-tracking has not been implemented. Thus it must be possible
    to reach any instruction that is reachable, without jumping backwards.

    """
    def __init__(self,state : T) -> None:
        self.state = state # type: Optional[T]
        self.prior_states = {} # type: Dict[Target,T]
        self.pending_states = collections.defaultdict(list) # type: DefaultDict[Target,List[U]]

        if __debug__:
            self.elided_targets = set() # type: Set[Target]

    def handle_instr(self,op_i : int,op : Instr) -> None:
        pass

    def handle_invalidate_regs(self,op_i : int,op : InvalidateRegs) -> None:
        pass

    def handle_create_var(self,op_i : int,op : CreateVar) -> None:
        pass

    def handle_indirect_mod(self,op_i : int,op : IndirectMod) -> None:
        pass

    def handle_target(self,op_i : int,op : Target) -> None:
        pass

    def handle_irjump(self,op_i : int,op : IRJump) -> None:
        assert self.state is not None

        assert op.dest not in self.elided_targets,'back-tracking support has not been added to this procedure'

        if op.dest in self.prior_states:
            self.conform_prior_state(op_i,op,self.prior_states[op.dest])
        else:
            self.pending_states[op.dest].append(self.make_pending_state(op_i,op))

        if not op.conditional:
            old = self.state
            self.state = None
            self.after_state_change(old)

    def handle_lock_regs(self,op_i : int,op : LockRegs):
        pass

    def handle_unlock_regs(self,op_i : int,op :UnlockRegs):
        pass

    def conform_state_to(self,op_i : int,state : U) -> None:
        pass

    def conform_prior_state(self,op_i : int,op : IRJump,state : T) -> None:
        pass

    def after_state_change(self,old_state : Optional[T]) -> None:
        pass

    def make_pending_state(self,op_i : int,op : IRJump) -> U:
        raise NotImplementedError()

    def make_prior_state(self,op_i : int) -> T:
        raise NotImplementedError()

    def recall_pending_state(self,op_i : int,state : U) -> T:
        raise NotImplementedError()

    def handle_op(self,op_i : int,op : IROp) -> None:
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
            assert op not in self.prior_states

            states = self.pending_states.get(op)

            if states:
                old = self.state

                del self.pending_states[op]
                if self.state is None:
                    self.state = self.recall_pending_state(op_i,states[0])
                    states = states[1:]

                for s in states:
                    self.conform_state_to(op_i,s)

                self.after_state_change(old)

            if self.state is not None:
                self.prior_states[op] = self.make_prior_state(op_i)
                self.handle_target(op_i,op)
            elif __debug__:
                self.elided_targets.add(op)
        elif isinstance(op,IRJump):
            if self.state is not None:
                self.handle_irjump(op_i,op)
        elif isinstance(op,LockRegs):
            if self.state is not None:
                self.handle_lock_regs(op_i,op)
        else:
            assert isinstance(op,UnlockRegs)
            if self.state is not None:
                self.handle_unlock_regs(op_i,op)

    def run(self,code : Iterable[IROp]) -> None:
        for op_i,op in enumerate(code):
            self.handle_op(op_i,op)


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

    def handle_instr(self,op_i : int,op : Instr):
        for var,pd in zip(op.args,op.op.param_dirs):
            if isinstance(var,Var):
                self.create_var_life(var,op_i,pd.reads,pd.writes)
            elif isinstance(var,IndirectVar):
                if isinstance(var.base,Var):
                    self.create_var_life(var.base,op_i,True,False)
                if isinstance(var.index,Var):
                    self.create_var_life(var.index,op_i,True,False)

    def handle_create_var(self,op_i : int,op : CreateVar):
        self.create_var_life(op.var,op_i,False,True)

    def handle_irjump(self,op_i : int,op : IRJump) -> None:
        assert self.state is not None
        state = self.state

        super().handle_irjump(op_i,op)

        if self.state is None:
            state.apply_all(op_i+1)

            prior = self.prior_states.get(op.dest)
            if prior:
                self.state = VarState({life: op_i + 1 for life in prior.read_starts})
            else:
                self.state = state

    def make_prior_state(self,op_i : int) -> VarState:
        assert self.state is not None
        return VarState(self.state.read_starts.copy())

    def make_pending_state(self,op_i : int,op : IRJump) -> Tuple[VarState,int]:
        assert self.state is not None
        return VarState(self.state.read_starts.copy()),op_i

    def recall_pending_state(self,op_i : int,state : Tuple[VarState,int]) -> VarState:
        return state[0]

    def conform_prior_state(self,op_i : int,op : IRJump,state : VarState) -> None:
        assert self.state is not None

        for life in (state.read_starts.keys() | self.state.read_starts.keys()):
            m_rs = self.state.read_starts.get(life)
            if m_rs is None:
                assert life in state.read_starts
                m_rs = op_i + 1

            self.state.read_starts[life] = m_rs

class _CalcVarIntervals(_CalcVarIntervalsCommon):
    def __init__(self):
        super().__init__()

        # maps a lifetime to a set of code positions that we will need to
        # back-track to
        self.back_starts = collections.defaultdict(set) # type: DefaultDict[int,Set[Lifetime]]

        self.block_vars = [] # type: List[Block]

    # Here, "write" means completely overwrite. In the case of partial write,
    # both "read" and "write" will be false.
    def create_var_life(self,var,i,read,write):
        assert self.state is not None

        # the variable doesn't officially exist until after the instruction
        if not read: i += 1

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

            var.lifetime.intervals |= Interval(i,i + 1)
        elif write:
            self.state.apply(var.lifetime,i)

        if read:
            assert var.lifetime is not None
            self.state.read_starts.setdefault(var.lifetime,i + 1)

    def conform_state_to(self,op_i : int,state : Tuple[VarState,int]) -> None:
        assert self.state is not None

        for life in self.state.read_starts:
            # This variable's lifetime needs to be propagated to a later
            # location. Since we are scanning the code backwards, we have
            # to do this in another pass
            self.back_starts[state[1]].add(life)

class _CalcBackVarIntervals(_CalcVarIntervalsCommon):
    def create_var_life(self,var,i,read,write):
        assert self.state

        if isinstance(var,VarPart):
            self.create_var_life(var.block,i,read,False)

        if not read: i += 1

        # the previous pass should have handled it by this point, so don't
        # track it any further
        assert var.lifetime is not None
        self.state.apply(var.lifetime,i,True)


def calc_var_intervals(code : IRCode) -> None:
    c = _CalcVarIntervals()
    for op_i in range(len(code) - 1,-1,-1):
        c.handle_op(op_i,code[op_i])

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
        cb = _CalcBackVarIntervals()
        for op_i in range(len(code)-1,-1,-1):
            cb.handle_op(op_i,code[op_i])

            lives = c.back_starts.get(op_i)
            if lives:
                assert cb.state is not None
                for life in lives:
                    cb.state.read_starts.setdefault(life,op_i+1)


class Filter(Container[T],Generic[T]):
    def __init__(self,include : Optional[Container[T]]=None,exclude : Container[T]=()) -> None:
        self.include = include
        self.exclude = exclude

    def __contains__(self,item):
        return (self.include is None or item in self.include) and item not in self.exclude

class ItvLocation:
    def __init__(self,reg : Optional[int]=None,stack_loc : Optional[StackLocation]=None) -> None:
        self.reg = reg
        self.stack_loc = stack_loc

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

    def to_opt_ir(self,size=SIZE_DEFAULT) -> Union[FixedRegister,StackItem,ArgStackItem,None]:
        """Return an IR value representing a value in this location.

        If this location is both a register and stack item, the return value
        will represent a register. If this location is neither, None is
        returned.

        """
        if self.reg is not None:
            return FixedRegister(self.reg,size=size)

        if self.stack_loc is not None:
            return to_stack_item(self.stack_loc,size)

        return None

    def to_ir(self,size=SIZE_DEFAULT) -> Union[FixedRegister,StackItem,ArgStackItem]:
        """Return an IR value representing a value in this location.

        If this location is both a register and stack item, the return value
        will represent a register. If this location is neither, a ValueError
        will be raised.

        """
        r = self.to_opt_ir(size)
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
        r.itv_locs = collections.defaultdict(ItvLocation,((life,copy.copy(loc)) for life,loc in self.itv_locs.items()))
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

    def _alloc_s(self,life : VarLifetime,itv : Optional[Interval[int]]=None):
        itvl = self.itv_locs[life]
        if itvl.stack_loc is not None: return

        if life.preferred_stack_i is not None:
            assert self.stack_pool[life.preferred_stack_i] is life
            itvl.stack_loc = StackLocation(StackSection.local,life.preferred_stack_i)
        else:
            for i in range(len(self.stack_pool)):
                if self.stack_pool[i] is None:
                    self.stack_pool[i] = life
                    itvl.stack_loc = StackLocation(StackSection.local,i)
                    break
            else:
                itvl.stack_loc = StackLocation(StackSection.local,len(self.stack_pool))
                self.stack_pool.append(life)

            life.preferred_stack_i = itvl.stack_loc.index
        if itv is None: itv = life.itv_at(self.cur_pos)
        self.active_s.add_item((cast(Interval[int],itv),life))

    # stack 'blocks' occupy more than one contiguous spot
    #def _stack_pool_remove(self,itv,i):
    #    assert self.stack_pool[i] is itv
    #    for j in range(i,len(self.stack_pool)):
    #        if self.stack_pool[j] is not itv:
    #            break
    #        self.stack_pool[j] = None

    def is_reg_free(self,r):
        return self.reg_pool[r] is None

    @consistency_check
    def advance(self,pos : int,on_loc_expire : Optional[Callable[[VarLifetime],None]]) -> None:
        self.cur_pos = pos

        while self.active_r:
            itv,life = self.active_r[0]
            loc = self.itv_locs[life]
            assert loc.reg is not None
            if itv.end > pos: break
            del self.active_r[0]
            try:
                new_itv = life.itv_at(pos)
            except ValueError:
                self.reg_pool[loc.reg] = None
                loc.reg = None

                # if loc.stack_loc is not None, the next loop will call
                # on_loc_expire for this instance of VarLifetime
                if on_loc_expire and loc.stack_loc is None: on_loc_expire(life)
            else:
                self.active_r.add_item((new_itv,life))


        while self.active_s:
            itv,life = self.active_s[0]
            loc = self.itv_locs[life]
            assert loc is not None
            assert loc.stack_loc is not None
            assert loc.stack_loc.sect == StackSection.local
            if itv.end > pos: break
            del self.active_s[0]
            try:
                new_itv = life.itv_at(pos)
            except ValueError:
                #self._stack_pool_remove(life,loc.stack_loc.index)
                loc.stack_loc = None
                if on_loc_expire: on_loc_expire(life)
            else:
                self.active_s.add_item((new_itv,life))

        for i,life in enumerate(self.stack_pool):
            if life is not None and pos >= life.intervals.global_end:
                self.stack_pool[i] = None

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

        If the register doesn't have a value, None is returned. Otherwise, the
        return value is an instance of VarLifetime, indicating where the
        value was moved.

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
            self._alloc_s(life)

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
    def to_stack(self,itv : VarLifetime) -> bool:
        """Copy the value to the stack.

        This will place a value onto the stack, if it's not already there. This
        will not free a register.

        """
        if self.itv_locs[itv].stack_loc is not None: return False
        self._alloc_s(itv)
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
            if itvl.stack_loc is not None:
                if itvl.stack_loc.sect == StackSection.local:
                    #self._stack_pool_remove(life,itvl.stack_loc.index)
                    self._remove_active(life,self.active_s)
                itvl.stack_loc = None
        else:
            assert where == LocationType.stack
            if itvl.reg is not None:
                self.reg_pool[itvl.reg] = None
                self._remove_active(life,self.active_r)
                itvl.reg = None

    @consistency_check
    def alloc_block(self,life : VarLifetime,size : int) -> None:
        itvl = self.itv_locs[life]
        if itvl.stack_loc is not None: return

        if life.preferred_stack_i is not None:
            assert all(self.stack_pool[life.preferred_stack_i + i] is life for i in range(size))
            itvl.stack_loc = StackLocation(StackSection.local,life.preferred_stack_i)
        else:
            i = 0
            n = len(self.stack_pool)
            while i < n:
                for j in range(i,min(n,i+size)):
                    if self.stack_pool[j] is not None:
                        i = j
                        break
                else:
                    break
                n += 1

            for j in range(i,i + size):
                if j == len(self.stack_pool):
                    self.stack_pool.append(life)
                else:
                    self.stack_pool[j] = life

            itvl.stack_loc = StackLocation(StackSection.local,i)
            life.preferred_stack_i = i
        self.active_s.add_item((life.itv_at(self.cur_pos),life))

    def interval_loc(self,itv : Lifetime):
        if isinstance(itv,AliasLifetime):
            itv = itv.itv

        assert isinstance(itv,VarLifetime)
        return self.itv_locs[itv]

    def to_ir(self,itv : Lifetime) -> Union[FixedRegister,StackItem,ArgStackItem]:
        return self.interval_loc(itv).to_ir()

def load_to_reg(alloc : LocationScan,itv : VarLifetime,allowed_reg : Container[int],cgen : OpGen,code : IRCode2,val : Optional[Value]=None) -> FixedRegister:
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

def load_to_block(alloc,block):
    assert block.lifetime is not None
    alloc.alloc_block(block.lifetime,len(block))
    return alloc.to_ir(block.lifetime)

class JumpState:
    def __init__(self,alloc : LocationScan,code : IRCode2,jump_ops : int) -> None:
        self.alloc = alloc
        self.code = code
        self.jump_ops = jump_ops

    def extend_code(self,ops : IRCode2):
        if self.jump_ops:
            self.code[-self.jump_ops:-self.jump_ops] = ops
        else:
            self.code.extend(ops)

    def conform_to(self,cgen : OpGen,other : LocationScan) -> None:
        """Move values so that they have the same locations in self.alloc and
        other."""

        # noinspection PyUnresolvedReferences
        for itv in self.alloc.itv_locs.keys() & other.itv_locs.keys():
            loc_self = self.alloc.itv_locs[itv]
            loc_o = other.itv_locs[itv]

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
                    loc_self.stack_loc = None
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
                loc_self.reg = None

def hassubclass(cls : Union[type,Tuple[type,...]],classinfo : Union[type,Tuple[type,...]]) -> Optional[type]:
    if isinstance(cls,tuple):
        for c in cls:
            if issubclass(c,classinfo): return c
    elif issubclass(cls,classinfo):
        return cls

    return None

def ir_preallocated_to_ir2(code):
    r = []
    for instr in code:
        if isinstance(instr,Instr):
            r.append(instr.op.to_ir2(instr.args))
        else:
            assert isinstance(instr,Target)
            r.append(instr)

    return r

class _RegAllocate(FollowNonlinear[LocationScan,JumpState]):
    def __init__(self,cgen : OpGen,alloc : LocationScan) -> None:
        super().__init__(alloc)
        self.cgen = cgen
        self.stack_pool = alloc.stack_pool

        self.new_code = [] # type: List[IRCode2]
        self.new_code_head = [] # type: IRCode2

        self.active_symbols = set() # type: Set[str]

    def move_for_param(self,i,p,new_args,reads):
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
                assert not isinstance(arg,(Block,VarPart)),(
                    "'Block' and 'VarPart' instances cannot be put in registers")
                assert isinstance(itv,VarLifetime)
                dest = load_to_reg(self.state,itv,
                    cast(_RegisterMetaType,fr_type).allowed,self.cgen,self.new_code_head,arg if reads else None)
            else:
                assert hassubclass(p,AddressType)
                if isinstance(arg,Block):
                    assert not reads
                    dest = load_to_block(self.state,arg)
                elif isinstance(arg,VarPart):
                    assert not reads
                    dest = load_to_block(self.state,arg.block)
                else:
                    assert isinstance(itv,VarLifetime)
                    moved = self.state.to_stack(itv)
                    d_loc = self.state.interval_loc(itv)
                    dest = to_stack_item(d_loc.stack_loc)
                    if moved and reads:
                        self.new_code_head.extend(
                            ir_preallocated_to_ir2(self.cgen.move(arg,dest)))
                        self.new_code_head.extend(annotate_symbol_loc(itv,d_loc))

            new_args[i] = dest

    def get_var_loc(self,var):
        assert var.lifetime is not None
        itvl = self.state.interval_loc(var.lifetime)

        if itvl.reg is not None:
            return FixedRegister(itvl.reg)

        assert itvl.stack_loc is not None
        if isinstance(var,VarPart):
            assert itvl.stack_loc.sect == StackSection.local
            return StackItem(itvl.stack_loc.index,offset=var.offset)

        return to_stack_item(itvl.stack_loc)

    def _on_loc_expire(self,life):
        self.new_code_head.extend(annotate_symbol_loc(life,ItvLocation()))

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

            if isinstance(a,Var):
                new_args[i] = self.get_var_loc(a)
            elif isinstance(a,Block):
                new_args[i] = self.get_var_loc(a[0])

        overload = op.op.best_match(new_args)
        inds = []
        for i,ta in enumerate(zip(overload.params,op.args)):
            if hassubclass(ta[0],AddressType) and isinstance(ta[1],IndirectVar):
                inds.append(i)
        if inds:
            op,overload = self.cgen.process_indirection(op,overload,inds)

        new_args = []

        # Second pass: every argument we read-from should already have a
        # location. Some arguments will need to be moved.
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
                continue

            if isinstance(a,Block):
                assert a.lifetime is not None
                self.state.alloc_block(a.lifetime,len(a))
                new_args[i] = self.get_var_loc(a[0])
            elif isinstance(a,Var):
                new_args[i] = self.get_var_loc(a)

            self.move_for_param(i,overload.params[i],new_args,True)

        # writes always apply after the instruction
        self.state.advance(op_i + 1,self._on_loc_expire)

        # Third pass: for the rest of the arguments, create a physical location
        # or if needed, move to another location. Arguments that are neither
        # read-from nor written-to (as in the case of the first argument of
        # x86's 'lea') will still need a location.
        for i,p in enumerate(overload.params):
            pd = op.op.param_dirs[i]

            if not pd.reads:
                self.move_for_param(i,p,new_args,False)

            if pd.writes:
                arg = op.args[i]

                if isinstance(arg,Var) and not isinstance(arg,VarPart):
                    assert isinstance(arg.lifetime,VarLifetime)

                    if hassubclass(p,FixedRegister):
                        self.state.value_updated(arg.lifetime,LocationType.register)
                    elif hassubclass(p,AddressType):
                        self.state.value_updated(arg.lifetime,LocationType.stack)

        self.new_code_head.append(Instr2(op.op,overload,new_args))

    def handle_invalidate_regs(self,op_i : int,op : InvalidateRegs):
        assert self.state is not None

        for r in op.to_free:
            displaced = self.state.free_reg(r,op.pres)
            if displaced is not None:
                disp_l = self.state.interval_loc(displaced)
                self.new_code_head.extend(
                    ir_preallocated_to_ir2(self.cgen.move(FixedRegister(r),disp_l.to_ir())))
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
        assert self.state is not None

        assert isinstance(op.var.lifetime,VarLifetime)
        self.state.value_updated(op.var.lifetime,op.loc_type)

    def handle_target(self,op_i : int,op : Target):
        self.new_code_head.append(op)

    def handle_lock_regs(self,op_i : int,op : LockRegs):
        assert self.state is not None
        self.state.locked_regs.update(op.regs)

    def handle_unlock_regs(self,op_i : int,op : UnlockRegs):
        assert self.state is not None
        self.state.locked_regs.difference_update(op.regs)

    def conform_state_to(self,op_i : int,state : JumpState):
        assert self.state is not None

        if state.alloc.cur_pos != self.state.cur_pos:
            state.alloc.advance(self.state.cur_pos,self._on_loc_expire)
        state.conform_to(self.cgen,self.state)

    def conform_prior_state(self,op_i : int,op : IRJump,state : LocationScan):
        JumpState(self.state,self.new_code_head,op.jump_ops).conform_to(self.cgen,state)

    def make_prior_state(self,op_i : int) -> LocationScan:
        assert self.state is not None
        return self.state.branch()

    def make_pending_state(self,op_i : int,op : IRJump):
        assert self.state is not None

        r = JumpState(self.state.branch(),self.new_code_head,op.jump_ops)
        self.new_code.append(self.new_code_head)
        self.new_code_head = []

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

    def handle_op(self,op_i : int,op : IROp):
        if self.state is not None and self.state.cur_pos != op_i: self.state.advance(op_i,self._on_loc_expire)

        super().handle_op(op_i,op)

def reg_allocate(cgen : OpGen,code : IRCode,regs : int) -> Tuple[IRCode2,int,int]:
    """Convert IRCode into IRCode2.

    This converts all instances of Var into FixedRegister or StackItem, and
    adds extra instructions to shuffle values between registers and the stack,
    as needed.

    As a side-effect, this will also remove unreachable code.

    Note that back-tracking has not been implemented. Thus it must be possible
    to reach any instruction that is reachable, without jumping backwards.

    """
    calc_var_intervals(code)
    ls = LocationScan(regs,cgen.allocater_extra_state())
    stack_pool = ls.stack_pool
    allocater = _RegAllocate(cgen,ls)
    allocater.run(code)

    allocater.new_code.append(allocater.new_code_head)
    # TODO: return actual registers used
    return functools.reduce(operator.concat,allocater.new_code,[]),regs,len(stack_pool)


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
            return vars[x]
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
    vars = collections.defaultdict(Namer('var_'))
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
                else:
                    op = 'read_op'

                line = 'ir.Instr({},{})'.format(op,arg)
                if line != code_str[-1]:
                    code_str.append(line)
        elif isinstance(instr,Target):
            code_str.append(targets[instr])
        elif isinstance(instr,IRJump):
            code_str.append('ir.IRJump({},{!r},0)'.format(targets[instr.dest],instr.conditional))
        elif isinstance(instr,IndirectMod):
            code_str.append('ir.IndirectMod({},ir.LocationType.{})'.format(
                convert_val(instr.var),
                'register' if instr.loc_type == LocationType.register else 'stack'))
        elif isinstance(instr,CreateVar):
            code_str.append('ir.CreateVar({},ir.{!r})'.format(convert_val(instr.var),instr.val))
        elif isinstance(instr,(LockRegs,UnlockRegs,InvalidateRegs)):
            code_str.append('ir.{!r}'.format(instr))

    r = []
    for name in targets.values():
        r.append('        {} = ir.Target()'.format(name))

    for name in vars.values():
        r.append("        {0} = ir.Var('{0}')".format(name))

    for b,name in blocks.items():
        r.append('        {} = ir.Block({})'.format(name,len(b.parts)))

    r.append('        code = [')
    for i,c in enumerate(code_str):
        comma = ','
        if i == len(code_str) - 1: comma = ']'
        r.append('            {}{}'.format(c,comma))

    return '\n'.join(r)
