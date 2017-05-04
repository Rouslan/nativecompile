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


__all__ = ['string_addr','DelayedCompile','CleanupItem','StackCleanupSection',
    'StackCleanup','StitchValue','check_args','Signedness','signed','unsigned',
    'SCmp','make_value','make_arg','make_cmp','SBinCmp','SAndCmp','SOrCmp',
    'SNotCmp','and_','or_','not_','MaybeTrackedValue','TrackedValue',
    'PyObject','BorrowedValue','borrow','StolenValue','steal','ConstValue',
    'TupleItem','tuple_item','ObjArrayValue','CType','type_of','type_flags_of',
    'SImmediate','SIndirect','SFinallyTarget','State','Stitch','destitch']

import sys
import weakref
from functools import reduce
from typing import Any,Callable,cast,Dict,FrozenSet,Generic,Iterable,List,Optional,overload,Tuple,TypeVar,Union

from . import pyinternals
from . import c_types
from .intermediate import *


def string_addr(x):
    if isinstance(x,str): return Symbol(x)
    return x

class DelayedCompile:
    def compile(self,s):
        raise NotImplementedError()

    @staticmethod
    def process(cgen,code):
        r = []

        for op in code:
            if isinstance(op,DelayedCompile):
                r.extend(op.compile(cgen))
            else:
                r.append(op)

        return r

class CleanupItem:
    def __init__(self,loc,free):
        self.loc = loc
        self.free = free

    def __call__(self,s):
        self.free(s,self.loc)

    def __hash__(self):
        return hash(self.loc) ^ hash(self.free)

    def __eq__(self,b):
        if isinstance(b,CleanupItem):
            return self.loc == b.loc and self.free == b.free

        return NotImplemented

class StackCleanupSection(DelayedCompile):
    def __init__(self,s,locations : Iterable[Callable[['Stitch'],None]],dest,action) -> None:
        self.next = None
        self.abi = s.cgen.abi
        self.locations = frozenset(locations) # type: FrozenSet[Callable[['Stitch'],None]]
        self.dest = dest
        self.action = action
        self.start = Target()

    @property
    def displacement(self):
        assert self.start.displacement is not None
        return self.start.displacement

    def __eq__(self,b):
        if isinstance(b,StackCleanupSection):
            assert self.abi is b.abi
            return self.action == b.action and self.locations == b.locations and self.dest == b.dest

        return NotImplemented

    def __ne__(self,b):
        if isinstance(b,StackCleanupSection):
            assert self.abi is b.abi
            return self.action != b.action or self.locations != b.locations or self.dest != b.dest

        return NotImplemented

    def __hash__(self):
        return hash(self.action) ^ hash(self.locations) ^ hash(self.dest)

    # If a subsequent cleanup section's actions are a subset of this section's
    # actions and has the same destination, this section will only perform the
    # actions that are different and jump to the next section to finish the
    # cleanup. For a bigger reduction, the search for subsets could be expanded
    # to prior sections. For an even bigger reduction, jumps could be made into
    # the middle of sections, so that sections need only to intersect; in such
    # a case, different orderings of actions would need to be considered for
    # finding the optimal arrangement.
    def compile(self,cgen):
        s = Stitch(cgen).comment('cleanup start')(self.start)
        best = None
        if self.locations:
            next_ = self.next
            while next_:
                # the comparison between len(next.locations) and
                # len(best.locations) intentionally uses >= because when there
                # are multiple identical sections, all but the last one will
                # just be jumps to the last one
                if len(next_.locations) and (best is None or len(next_.locations) >= len(best.locations)) and self.locations >= next_.locations:
                    best = next_
                next_ = next_.next

            clean_here = self.locations
            if best is not None: clean_here -= best.locations

            # sorted so the output is deterministic
            for free in sorted(clean_here,key=id): free(s)

        if best:
            s.jmp(best.start)
        else:
            if self.action: self.action(s)
            if isinstance(self.dest,StackCleanupSection):
                s.extend(self.dest.compile(cgen))
            else:
                s.jmp(self.dest)

        return DelayedCompile.process(cgen,s.comment('cleanup end').code)

class StackCleanup:
    def __init__(self):
        self.depth = 1
        self.last = {}

    def new_section(self,s,locations,destination,action=None):
        old = self.last.get(destination)
        last = StackCleanupSection(s,locations,destination,action)
        if old is not None: old.next = last
        self.last[destination] = last
        return last


T = TypeVar('T')

class StitchValue(Generic[T]):
    """A value dependent on the state of a Stitch object.

    This is a lazily evaluated value that can be used directly but also serves
    as a DSL for comparisons.

    """
    def __lt__(self,b):
        return SBinCmp.from_dsl(self,b,CmpType.lt)

    def __le__(self,b):
        return SBinCmp.from_dsl(self,b,CmpType.le)

    def __eq__(self,b):
        return SBinCmp.from_dsl(self,b,CmpType.eq)

    def __ne__(self,b):
        return SBinCmp.from_dsl(self,b,CmpType.ne)

    def __gt__(self,b):
        return SBinCmp.from_dsl(self,b,CmpType.gt)

    def __ge__(self,b):
        return SBinCmp.from_dsl(self,b,CmpType.ge)

    def __call__(self,s: 'Stitch') -> T:
        raise NotImplementedError()

StitchT = Union[T,StitchValue[T]]

if __debug__:
    def check_args(f):
        def inner(self,op,*args):
            assert not any(isinstance(a,StitchValue) for a in args)
            f(self,op,*args)
        return inner

    Instr.__init__ = check_args(Instr.__init__)


class Signedness(StitchValue[T]):
    def __init__(self,val,signed : bool) -> None:
        self.val = val
        self.signed = signed

    def __call__(self,s):
        return make_value(s,self.val)

def signed(x) -> Signedness:
    return Signedness(x,True)

def unsigned(x) -> Signedness:
    return Signedness(x,False)

# noinspection PyAbstractClass
class SCmp(StitchValue[Cmp]):
    pass

def make_value(s: 'Stitch',x):
    if isinstance(x,StitchValue):
        return x(s)
    return string_addr(x)

def make_arg(s: 'Stitch',x):
    """Convert x into an instance of Value, if a conversion exists, or return
    x unchanged."""
    r = make_value(s,x)
    return Immediate(r) if isinstance(r,int) else r

def _get_type(x):
    return x.data_type if isinstance(x,Value) else None

def _common_type(a,b):
    if a is None:
        return b
    if b is None:
        return a
    if a == b:
        return a
    return c_types.t_void_ptr

def make_args(s: 'Stitch',*args):
    """Convert arguments to instances of Value.

    This is equivalent to "[make_arg(a) for a in args]" except untyped
    literals will get the same type as the common type of all typed arguments,
    if such a type exists.

    """
    vals = [make_value(s,a) for a in args]
    common_t = reduce(_common_type,map(_get_type,vals),None)
    if common_t is None: common_t = c_types.t_void_ptr

    for i,a in enumerate(vals):
        if isinstance(a,int):
            vals[i] = Immediate(a,common_t)

    return vals


def make_cmp(x):
    if isinstance(x,SCmp):
        return x

    return SBinCmp(x,0,CmpType.ne)

class SBinCmp(SCmp):
    def __init__(self,a,b,op,signed=True):
        self.a = a
        self.b = b
        self.op = op
        self.signed = signed

    def __call__(self,s):
        return BinCmp(make_arg(s,self.a),make_arg(s,self.b),self.op,self.signed)

    @staticmethod
    def from_dsl(a,b,op):
        signed = True
        if isinstance(a,Signedness):
            if isinstance(b,Signedness):
                if a.signed != b.signed:
                    raise ValueError('attempt to compare explicit signed with explicit unsigned values')
                b = b.val
            signed = a.signed
            a = a.val
        elif isinstance(b,Signedness):
            signed = b.signed
            b = b.val

        return SBinCmp(a,b,op,signed)

class SAndCmp(SCmp):
    def __init__(self,a,b):
        self.a = make_cmp(a)
        self.b = make_cmp(b)

    def __call__(self,s):
        return AndCmp(make_value(s,self.a),make_value(s,self.b))

class SOrCmp(SCmp):
    def __init__(self,a,b):
        self.a = make_cmp(a)
        self.b = make_cmp(b)

    def __call__(self,s):
        return OrCmp(make_value(s,self.a),make_value(s,self.b))

class SNotCmp(SCmp):
    def __init__(self,x):
        self.val = make_cmp(x)

    def __call__(self,s):
        return make_value(s,self.val).complement()

and_ = SAndCmp
or_ = SOrCmp
not_ = SNotCmp

ValueT = TypeVar('ValueT',bound=Value)

# noinspection PyAbstractClass
class MaybeTrackedValue(StitchValue[ValueT]):
    def discard(self,s: 'Stitch') -> None:
        raise NotImplementedError()

class TrackedValue(MaybeTrackedValue[ValueT]):
    """A value that requires clean-up.

    When the value is no longer needed in the generated code, 'discard' must be
    called to emit the clean-up code. This must be called in all control-flow
    branches of the generated code.

    """
    def __init__(self,value: ValueT,cleanup: Callable[['Stitch',ValueT],None],own: bool=False) -> None:
        self.value = value
        self.cleanup_func = cleanup
        self.initial_own = own

    def __call__(self,s):
        if self.initial_own: s.state.own(self,s.cleanup.depth)
        return self.value

    def discard(self,s):
        if s.state.owned(self):
            self.cleanup_func(s,self.value)
            s.state.disown(self)

class PyObject(TrackedValue[ValueT]):
    @staticmethod
    def cleanup(s,val):
        s.decref(val)

    @staticmethod
    def nullable_cleanup(s,val):
        s.if_(val).decref(val).endif()

    def __init__(self,value : Union[ValueT,str,None]=None,own : bool=False,nullable : bool=False) -> None:
        if isinstance(value,str):
            value = cast(ValueT,Var(value,data_type=c_types.PyObject_ptr))
            if __debug__:
                value.origin = _get_origin()
        elif value is None:
            value = cast(ValueT,Var(data_type=c_types.PyObject_ptr))
            if __debug__:
                value.origin = _get_origin()
        super().__init__(
            cast(ValueT,value),
            PyObject.nullable_cleanup if nullable else PyObject.cleanup,
            own)

    if __debug__:
        @property
        def origin(self):
            return getattr(self.value,'origin',None)

class BorrowedValue(MaybeTrackedValue[ValueT]):
    def __init__(self,value : TrackedValue[ValueT]) -> None:
        self.value = value

    def __call__(self,s):
        return self.value(s)

    def discard(self,s):
        pass

def borrow(x):
    if isinstance(x,BorrowedValue): return x
    return BorrowedValue(x)

class StolenValue(StitchValue[ValueT]):
    def __init__(self,val : MaybeTrackedValue[ValueT]) -> None:
        self.val = val
        self.loc = None # type: Optional[ValueT]

    def __call__(self,s):
        if self.loc is None:
            if isinstance(self.val,ConstValue):
                self.loc = Var(data_type=c_types.PyObject_ptr)
                s.mov(self.val.value,self.loc)
                s.incref(self.loc)
            else:
                self.loc = self.val(s)

                if isinstance(self.val,TrackedValue):
                    if s.state.owned(self.val):
                        s.state.disown(self.val)
                    elif isinstance(self.val,PyObject):
                        s.incref(self.loc)
                elif isinstance(self.val,BorrowedValue) and isinstance(self.val.value,PyObject):
                    s.incref(self.loc)
                else:
                    raise ValueError('object is neither owned nor reference-counted')

                self.val.discard(s)

        return self.loc

steal = StolenValue

class ConstValue(MaybeTrackedValue[ValueT]):
    """Represents a never-owned Python object."""

    def __init__(self,value: ValueT) -> None:
        self.value = value

    def __call__(self,s):
        return self.value

    def discard(self,s):
        pass


class TupleItem(StitchValue):
    def __init__(self,r,n):
        self.addr = CType('PyTupleObject',r).ob_item
        self.n = n

    def __call__(self,s):
        return (self.addr + SIZE_PTR * self.n)(s)

tuple_item = TupleItem


class ObjArrayValue(TrackedValue[ValueT]):
    def __init__(self,value : Union[ValueT,str,None]=None,own : bool=False) -> None:
        if value is None:
            value = cast(ValueT,Var(data_type=c_types.TPtr(c_types.PyObject_ptr)))
        elif isinstance(value,str):
            value = cast(ValueT,Var(value,c_types.TPtr(c_types.PyObject_ptr)))
        super().__init__(
            cast(ValueT,value),
            (lambda s,val: s.call('free_pyobj_array',val)),
            own)


class CType(StitchValue[ValueT]):
    def __init__(self,t: str,base: Optional[ValueT]=None,index: Optional[Value]=None,scale=SIZE_PTR) -> None:
        self.offsets = pyinternals.member_offsets[t]
        self.base = Var() if base is None else base
        self.index = index
        self.scale = scale

    def __getattr__(self,name):
        offset = self.offsets.get(name)
        if offset is None: raise AttributeError(name)
        return SIndirect(offset,self.base,self.index,self.scale)

    def __call__(self,s):
        return self.base

def type_of(r):
    return CType('PyObject',r).ob_type

def type_flags_of(r):
    return CType('PyTypeObject',r).tp_flags

class SImmediate(StitchValue[Immediate]):
    def __init__(self,val,size=SIZE_PTR):
        self.val = val
        self.size = size

    def __call__(self,s):
        return Immediate(make_value(s,self.val),make_value(s,self.size))

class SIndirect(StitchValue[IndirectVar]):
    def __init__(self,offset=0,base=None,index=None,scale=SIZE_PTR,data_type=c_types.t_void_ptr):
        self.offset = offset
        self.base = base
        self.index = index
        self.scale = scale
        self.data_type = data_type

    def __add__(self,b):
        if isinstance(b,(int,PtrBinomial)):
            return SIndirect(self.offset+b,self.base,self.index,self.scale,self.data_type)

        return NotImplemented

    def __sub__(self,b):
        return self.__add__(-b)

    def __call__(self,s):
        return IndirectVar(
            make_value(s,self.offset),
            make_value(s,self.base),
            make_value(s,self.index),
            make_value(s,self.scale),
            make_value(s,self.data_type))


class Scope:
    def cur_single_scope(self) -> 'Branch':
        raise NotImplementedError()

class Branch:
    """A distinct branch in generated machine code.

    Every instance creates a copy of the 'State' instance.

    """
    def __init__(self,state):
        if __debug__:
            self._origin = _get_origin()
        self.code = []
        self.state = state.copy()

class IfElseScope(Scope):
    def __init__(self,state,test,in_elif=False):
        if __debug__:
            self._origin = _get_origin()

        self.test = test
        self.branch_t = Branch(state)
        self.branch_f = None
        self.in_elif = in_elif

    def cur_single_scope(self) -> Branch:
        return self.branch_t if self.branch_f is None else self.branch_f

    def add_else(self,state):
        self.branch_f = Branch(state)

class SimpleScope(Scope,Branch):
    def cur_single_scope(self) -> 'Branch':
        return self

class MainScope(SimpleScope): pass
class DoWhileScope(SimpleScope): pass

class SFinallyTarget(FinallyTarget):
    def __init__(self) -> None:
        super().__init__()
        self.no_return = False

class FinallyEntry(DelayedCompile):
    def __init__(self,ftarget: SFinallyTarget,next_t: Target) -> None:
        self.ftarget = ftarget
        self.next_t = next_t

    def compile(self,cgen):
        assert self.next_t in self.ftarget.next_targets

        if self.ftarget.no_return:
            return cgen.jump(self.ftarget.start)

        return cgen.enter_finally(self.ftarget,self.next_t)

class FinallyScope(SimpleScope):
    def __init__(self,state: 'State',ftarget: SFinallyTarget) -> None:
        super().__init__(state)
        self.ftarget = ftarget

if __debug__:
    class CompareByID(weakref.ref):
        def __init__(self,val,callback=None):
            super().__init__(val,callback)

            self._hash = id(val)

            self.origin = getattr(val,'origin',None)

        def __eq__(self,b):
            val = self()
            if val is None: return False

            if isinstance(b,CompareByID):
                b = b()
                if b is None: return False

            return val is b

        def __hash__(self):
            return self._hash

        def __repr__(self):
            val = self()
            if val is not None:
                return 'CompareByID({!r})'.format(val)

            return '<expired CompareByID object>'
else:
    class CompareByID:
        def __init__(self,val,callback=None):
            self.val = val

        def __eq__(self,b):
            if isinstance(b,CompareByID):
                return self.val is b.val

            return self.val is b

        def __hash__(self):
            return id(self)

        def __repr__(self):
            return 'CompareByID({!r})'.format(self.val)

        def __call__(self):
            return self.val

class State:
    """Keeps track of implied state in machine code as it's generated."""
    def __init__(self) -> None:
        self._tracked_vals = {} # type: Dict[CompareByID,int]

    def __eq__(self,b):
        if isinstance(b,State):
            for tv in self._tracked_vals.keys() | b._tracked_vals.keys():
                if self._tracked_vals.get(tv) != b._tracked_vals.get(tv):
                    return False
            return True

        return NotImplemented

    def __ne__(self,b):
        r = self.__eq__(b)
        return r if r is NotImplemented else not r

    def _make_wrapper(self,val : TrackedValue) -> CompareByID:
        if __debug__:
            def check_ownership(val,self_ref=weakref.ref(self)):
                self_ = self_ref()
                if self_ is not None:
                    for v in self_._tracked_vals:
                        if v() is None:
                            if v.origin:
                                print('{}:{}: '.format(*v.origin),end='',file=sys.stderr)
                            print('Ref-counted value not freed (in the ' +
                                'generated machine code, not the running ' +
                                'interpreter)',file=sys.stderr)

            return CompareByID(val,check_ownership)

        return CompareByID(val)

    def owned_items(self) -> Iterable[Tuple[TrackedValue,int]]:
        r = []
        for wrap,cd in self._tracked_vals.items():
            val = wrap()
            if val is None:
                raise ValueError('one or more instances of TrackedValue was never disowned')
            r.append((val,cd))
        return r

    def own(self,x : TrackedValue,cleanup_depth : int) -> None:
        """Mark x as being owned.

        x must not be owned, prior to calling this.

        """
        assert not self.owned(x)
        self._tracked_vals[self._make_wrapper(x)] = cleanup_depth

    def disown(self,x : TrackedValue) -> None:
        """Un-mark x as being owned.

        x must be owned, prior to calling this.

        """
        del self._tracked_vals[CompareByID(x)]

    def owned(self,x : TrackedValue) -> bool:
        """Return true iff x is owned"""
        return self._tracked_vals.get(CompareByID(x)) is not None

    def copy(self) -> 'State':
        r = State()
        for wrap,depth in self._tracked_vals.items():
            val = wrap()
            if val is not None: r.own(val,depth)
        return r

if __debug__:
    def _get_origin():
        f = sys._getframe(2)
        while f:
            if f.f_code.co_filename != __file__:
                return f.f_code.co_filename,f.f_lineno

            f = f.f_back

        return None

    class DebugInstr(Instr):
        __slots__ = 'debug_origin',

    def debug_source(code):
        for i,instr in enumerate(code):
            if isinstance(instr,Instr):
                code[i] = DebugInstr(instr.op,*instr.args)
                code[i].debug_origin = _get_origin()
        return code
else:
    def debug_source(code):
        return code

class Stitch:
    """Create machine code concisely using method chaining"""

    def __init__(self,cgen : OpGen,state : Optional[State]=None,cleanup : Optional[StackCleanup]=None) -> None:
        self.cgen = cgen
        self._scopes = [MainScope(state or State())] # type: List[Scope]

        self.cleanup = cleanup or StackCleanup()

    def annotation(self,descr):
        return self.extend(annotate(descr))

    @property
    def code(self):
        assert len(self._scopes) == 1
        return self._cur_scope.code

    @property
    def _cur_scope(self) -> Branch:
        return self._scopes[-1].cur_single_scope()

    @property
    def state(self) -> State:
        return self._cur_scope.state

    def extend(self,x):
        if isinstance(x,Stitch):
            x = x.code

        self._cur_scope.code.extend(x)
        return self

    def append(self,x):
        self._cur_scope.code.append(x)
        return self

    def _debug_append(self,x):
        self._cur_scope.code.extend(debug_source(x))
        return self

    def own(self,x) -> 'Stitch':
        self.state.own(x,self.cleanup.depth)
        return self

    def disown(self,x) -> 'Stitch':
        self.state.disown(x)
        return self

    def endif(self):
        while True:
            top = self._scopes.pop()
            assert isinstance(top,IfElseScope)
            assert top.branch_f is None or top.branch_t.state == top.branch_f.state
            self.extend(self.cgen.if_(top.test,top.branch_t.code,top.branch_f and top.branch_f.code))
            self._cur_scope.state = top.branch_t.state
            if not top.in_elif: break
        return self

    def else_(self):
        top = self._scopes[-1]
        assert isinstance(top,IfElseScope)
        top.add_else(self._scopes[-2].cur_single_scope().state)
        return self

    def _if_(self,test,in_elif):
        self._scopes.append(IfElseScope(self.state,make_value(self,make_cmp(test)),in_elif))
        return self

    def if_(self,test):
        return self._if_(test,False)

    def elif_(self,test):
        return self.else_()._if_(test,True)

    def jump_if(self,test,dest):
        return self._debug_append(self.cgen.jump_if(dest,make_value(self,make_cmp(test))))

    def do(self):
        self._scopes.append(DoWhileScope(self.state))
        return self

    def while_(self,test):
        top = self._scopes.pop()
        assert isinstance(top,DoWhileScope)
        assert top.state == self._scopes[-1].cur_single_scope().state
        return self.extend(self.cgen.do_while(top.code,make_value(self,make_cmp(test))))

    def enter_finally(self,ftarget: SFinallyTarget,next_t: Target):
        return self.append(FinallyEntry(ftarget,next_t))

    def begin_finally(self,ftarget: SFinallyTarget):
        self._scopes.append(FinallyScope(self.state,ftarget))
        return self

    def end_finally(self,no_return=False):
        """Mark the end of the body of a "finally" construct.
        
        If the end of the body is never reached (because e.g. there is an
        unconditional break statement in the body), "no_return" must be True.
        
        """
        top = self._scopes.pop()
        assert isinstance(top,FinallyScope)

        if no_return:
            top.ftarget.no_return = True
            self.append(top.ftarget.start)
            return self.extend(top.code)

        return self.extend(self.cgen.finally_body(top.ftarget,top.code))

    def comment(self,c,*args,inline=False):
        if self.cgen.abi.assembly:
            self.append(Instr(
                inline_comment_desc if inline else comment_desc,
                c.format(*args) if len(args) else c))
        return self

    def incref(self,val,amount=1):
        """Generate instructions equivalent to Py_INCREF"""

        val = make_value(self,val)

        if pyinternals.REF_DEBUG:
            for _ in range(amount):
                self.call('Py_IncRef',val)

            return self

        return self.add(CType('PyObject',val).ob_refcnt,amount)

    def decref(self,val,amount=1):
        """Generate instructions equivalent to Py_DECREF"""
        assert amount > 0

        val = make_value(self,val)

        if pyinternals.REF_DEBUG:
            for _ in range(amount):
                self.call('Py_DecRef',val)

            return self

        ob_type = Var(data_type=c_types.PyTypeObject_ptr)
        (self
            .sub(CType('PyObject',val).ob_refcnt,amount)
            .if_(not_(CType('PyObject',val).ob_refcnt))
            .mov(CType('PyObject',val).ob_type,ob_type)
            .call(CType('PyTypeObject',ob_type).tp_dealloc,val))

        if pyinternals.COUNT_ALLOCS:
            self.call('inc_count',ob_type)

        return self.endif()

    def call(self,func,*args,store_ret=None,callconv=CallConvType.default):
        self._debug_append(self.cgen.call(
            make_arg(self,func),
            [make_arg(self,a) for a in args],
            make_arg(self,store_ret),
            callconv))

        if isinstance(func,str):
            self.comment(func,inline=True)

        return self

    def call_preloaded(self,func,arg_count,store_ret=None,callconv=CallConvType.default):
        self._debug_append(self.cgen.call_preloaded(
            make_arg(self,func),
            arg_count,
            make_arg(self,store_ret),
            callconv))

        if isinstance(func,str):
            self.comment(func,inline=True)

        return self

    def get_threadstate(self,dest):
        # as far as I can tell, after expanding the macros and removing the
        # "dynamic annotations" (see dynamic_annotations.h in the CPython
        # headers), this is all that PyThreadState_GET boils down to:
        return self.mov(IndirectVar(pyinternals.raw_addresses['_PyThreadState_Current']),dest)

    def _bin_op(self,a,b,dest,optype):
        a,b,dest = make_args(self,a,b,dest)
        return self._debug_append(
            self.cgen.bin_op(a,b,a if dest is None else dest,optype))

    @overload
    def add(self,a: StitchT[MutableValue],b: StitchT[Value],c: None) -> 'Stitch':
        pass

    @overload
    def add(self,a: StitchT[Value],b: StitchT[Value],c: StitchT[MutableValue]) -> 'Stitch':
        pass

    def add(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.add)

    @overload
    def sub(self,a: StitchT[MutableValue],b: StitchT[Value],c: None) -> 'Stitch':
        pass

    @overload
    def sub(self,a: StitchT[Value],b: StitchT[Value],c: StitchT[MutableValue]) -> 'Stitch':
        pass

    def sub(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.sub)

    @overload
    def mul(self,a: StitchT[MutableValue],b: StitchT[Value],c: None) -> 'Stitch':
        pass

    @overload
    def mul(self,a: StitchT[Value],b: StitchT[Value],c: StitchT[MutableValue]) -> 'Stitch':
        pass

    def mul(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.mul)

    @overload
    def and_(self,a: StitchT[MutableValue],b: StitchT[Value],c: None) -> 'Stitch':
        pass

    @overload
    def and_(self,a: StitchT[Value],b: StitchT[Value],c: StitchT[MutableValue]) -> 'Stitch':
        pass

    def and_(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.and_)

    @overload
    def or_(self,a: StitchT[MutableValue],b: StitchT[Value],c: None) -> 'Stitch':
        pass

    @overload
    def or_(self,a: StitchT[Value],b: StitchT[Value],c: StitchT[MutableValue]) -> 'Stitch':
        pass

    def or_(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.or_)

    @overload
    def xor(self,a: StitchT[MutableValue],b: StitchT[Value],c: None) -> 'Stitch':
        pass

    @overload
    def xor(self,a: StitchT[Value],b: StitchT[Value],c: StitchT[MutableValue]) -> 'Stitch':
        pass

    def xor(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.xor)

    def _unary_op(self,a,dest,optype):
        a,dest = make_args(self,a,dest)
        return self._debug_append(
            self.cgen.unary_op(a,a if dest is None else dest,optype))

    @overload
    def neg(self,a: StitchT[MutableValue],dest: None) -> 'Stitch':
        pass

    @overload
    def neg(self,a: StitchT[Value],dest: StitchT[MutableValue]) -> 'Stitch':
        pass

    def neg(self,a,dest=None):
        return self._unary_op(a,dest,UnaryOpType.neg)

    def jmp(self,dest,targets=None):
        return self._debug_append(self.cgen.jump(dest,targets))

    def jmp_if(self,dest,test):
        test = make_value(self,make_cmp(test))
        return self._debug_append(self.cgen.jump_if(dest,test))

    def shift(self,direction,val,amount,dest):
        val,dest = make_args(self,val,dest)
        return self._debug_append(self.cgen.shift(val,direction,amount,dest))

    def shl(self,val,amount,dest):
        return self.shift(ShiftDir.left,val,amount,dest)

    def shr(self,val,amount,dest):
        return self.shift(ShiftDir.right,val,amount,dest)

    def mov(self,src,dest):
        src,dest = make_args(self,src,dest)
        return self._debug_append(self.cgen.move(src,dest))

    def lea(self,addr,dest):
        addr = make_arg(self,addr)
        return self._debug_append(self.cgen.load_addr(addr,dest))

    def touched_indirectly(self,val,read,write,loc_type=LocationType.stack):
        """Indicate that 'val' may have been read or updated by something other
        than the code produced by 'self'."""
        self.append(IndirectMod(make_arg(self,val),read,write,loc_type))
        return self

    def create_var(self,var,val):
        self.append(CreateVar(make_arg(self,var),make_arg(self,val)))
        return self

    def jump_table(self,val,targets):
        """Generate a jump table that will jump to the corresponding target
        of the index in 'val'."""
        return self.extend(self.cgen.jump_table(make_arg(self,val),targets))

    def __call__(self,op,*,is_jmp=None):
        op = make_arg(self,op)
        assert not isinstance(op,list)

        if callable(op):
            assert is_jmp is None
            op(self)
        else:
            self._cur_scope.code.append(op)
            self.last_op_is_uncond_jmp = bool(is_jmp)

        return self


def destitch(x):
    return x.code if isinstance(x,Stitch) else x
