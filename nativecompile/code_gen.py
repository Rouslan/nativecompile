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


import operator
import sys
import weakref
from functools import reduce
from typing import Callable,Dict,FrozenSet,Generic,Iterable,List,Optional,Sized,Tuple,TYPE_CHECKING,TypeVar,Union

from . import pyinternals
from . import debug
from .intermediate import *

if TYPE_CHECKING:
    from . import abi


CALL_ALIGN_MASK = 0xf


def aligned_for_call(x):
    return (x + CALL_ALIGN_MASK) & ~CALL_ALIGN_MASK

def string_addr(x):
    if isinstance(x,str): return Immediate(pyinternals.raw_addresses[x])
    return x

address_of = id

class Function:
    def __init__(self,code,padding=0,offset=0,name=None,pyfunc=False,annotation=None,returns=None,params=None):
        self.code = code

        # the size, in bytes, of the trailing padding (nop instructions) in
        # "code"
        self.padding = padding

        self.offset = offset
        self.name = name
        self.pyfunc = pyfunc
        self.annotation = annotation or []
        self.returns = returns
        self.params = params or []

    def __len__(self):
        return len(self.code)

class CompilationUnit:
    def __init__(self,functions):
        self.functions = functions

    def __len__(self):
        return sum(map(len,self.functions))

    def write(self,out):
        for f in self.functions:
            out.write(f.code)


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

class AsmSequence:
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

def resolve_jumps(cgen : OpGen,regs : int,code1 : IRCode,end_targets=(),*,assembly=False) -> Function:
    code,r_used,s_used = reg_allocate(cgen,code1,regs)
    irc = cgen.get_compiler(r_used,s_used,cgen.max_args_used)

    displacement = 0
    pad_size = 0
    annot_size = 0
    annots = [] # type: List[debug.Annotation]

    # this item will be replaced with padding if needed
    late_chunks = [None] # type: List[Optional[Sized]]

    code = irc.prolog() + code + irc.epilog()

    for instr in reversed(code):
        if isinstance(instr,IRAnnotation):
            descr = instr.descr
            if isinstance(descr,IRSymbolLocDescr):
                descr = debug.VariableLoc(
                    descr.symbol,
                    irc.get_machine_arg(descr.loc.to_ir(),displacement) if descr.loc else None)

            annot = debug.Annotation(descr,annot_size)

            if assembly:
                late_chunks.append(AsmSequence([AsmOp(comment_desc,('annotation: {!r}'.format(annot.descr),),b'')]))

            # since appending to a list is O(1) while prepending is O(n), we
            # add the items backwards and reverse the list afterwards
            annots.append(annot)

            annot_size = 0
        elif isinstance(instr,Target):
            instr.displacement = displacement
        else:
            assert isinstance(instr,Instr2)
            chunk = irc.compile_early(make_asm_if_needed(instr,assembly),displacement)

            # items are added backwards for the same reason as above
            late_chunks.append(chunk)
            displacement -= len(chunk)
            annot_size += len(chunk)

    assert annot_size == 0 or not annots, "if there are any annotations, there should be one at the start"

    annots.reverse()

    # add padding for alignment
    if CALL_ALIGN_MASK:
        unpadded = displacement
        displacement = aligned_for_call(displacement)
        pad_size = displacement - unpadded
        if pad_size:
            late_chunks[0] = code_join([irc.compile_early(make_asm_if_needed(c,assembly),0) for c in irc.nops(pad_size)])

    for et in end_targets:
        et.displacement -= displacement

    return Function(
        code_join([irc.compile_late(c) for c in reversed(late_chunks) if c is not None]),
        pad_size,
        annotation=annots)


class DelayedCompile:
    def compile(self):
        raise NotImplementedError()

    @staticmethod
    def process(code):
        r = []
        for op in code:
            if isinstance(op,DelayedCompile):
                r.extend(op.compile())
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
    def compile(self):
        s = Stitch(self.abi).comment('cleanup start')(self.start)
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
                s.extend(self.dest.compile())
            else:
                s.jmp(self.dest)

        return s.comment('cleanup end').code

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

class StitchValue:
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

    def __call__(self,s : 'Stitch') -> T:
        raise NotImplementedError()

#StitchT = Union[T,StitchValue['StitchT']]

if __debug__:
    def check_args(f):
        def inner(self,op,*args):
            assert not any(isinstance(a,StitchValue) for a in args)
            f(self,op,*args)
        return inner

    Instr.__init__ = check_args(Instr.__init__)

class PtrBinomial(StitchValue):
    """A numeric value dependent on the size of a pointer.

    This is a binomial equal to "ptr_factor * ptr_size + val" where ptr_size is
    the size of a pointer on the host machine.

    """

    @staticmethod
    def __new__(cls,val: Union['PtrBinomial',int],
        ptr_factor: int = 0) -> 'PtrBinomial':
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
        def __init__(self,val: Union['PtrBinomial',int],
            ptr_factor: int = 0) -> None:
            self.val = 0
            self.ptr_factor = ptr_factor

    def __call__(self,s):
        return self.ptr_factor * s.cgen.abi.ptr_size + self.val

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

SIZE_PTR = PtrBinomial(0,1)

class Signedness(StitchValue):
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
class SCmp(StitchValue):
    pass

def make_value(s : 'Stitch',x):
    if isinstance(x,StitchValue):
        return x(s)
    return string_addr(x)

def make_arg(s : 'Stitch',x):
    r = make_value(s,x)
    return Immediate(r) if isinstance(r,int) else r

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

# noinspection PyAbstractClass
class MaybeTrackedValue(StitchValue):
    def discard(self,s):
        raise NotImplementedError()

# noinspection PyAbstractClass
class TrackedValue(MaybeTrackedValue):
    """A value that requires clean-up.

    When the value is no longer needed in the generated code, 'discard' must be
    called to emit the clean-up code. This must be called in all control-flow
    branches of the generated code.

    """
    def __init__(self,value : object,cleanup : Callable[['Stitch',object],None],own : bool=False) -> None:
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

class PyObject(TrackedValue):
    def __init__(self,value : Optional[Value]=None,own : bool=False) -> None:
        super().__init__(
            Var() if value is None else value,
            (lambda s,val: s.decref(val)),
            own)

        if __debug__ and value is None:
            self.value.origin = _get_origin()

    if __debug__:
        @property
        def origin(self):
            return getattr(self.value,'origin',None)

class BorrowedValue(MaybeTrackedValue):
    def __init__(self,value : TrackedValue) -> None:
        self.value = value

    def __call__(self,s):
        return self.value(s)

    def discard(self,s):
        pass

def borrow(x):
    if isinstance(x,BorrowedValue): return x
    return BorrowedValue(x)

class StolenValue(StitchValue):
    def __init__(self,val : TrackedValue) -> None:
        self.val = val
        self.loc = None

    def __call__(self,s):
        if self.loc is None:
            if isinstance(self.val,ConstValue):
                self.loc = Var()
                s.mov(self.val.value,self.loc)
                s.incref(self.loc)
            else:
                self.loc = self.val(s)

                if s.state.owned(self.val):
                    s.state.disown(self.val)
                elif isinstance(self.val,PyObject):
                    s.incref(self.loc)
                else:
                    raise ValueError('object is neither owned nor reference-counted')

                self.val.discard(s)

        return self.loc

steal = StolenValue

class ConstValue(MaybeTrackedValue):
    def __init__(self,value : Value) -> None:
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
        return (self.addr + s.cgen.abi.ptr_size * self.n)(s)

tuple_item = TupleItem


class ObjArrayValue(TrackedValue):
    def __init__(self,value : Optional[Value]=None,own : bool=False) -> None:
        super().__init__(
            Var() if value is None else value,
            (lambda s,val: s.call('free_pyobj_array',val)),
            own)


class CType(StitchValue):
    def __init__(self,t,base=None,index=None,scale=SIZE_PTR):
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

class SImmediate(StitchValue):
    def __init__(self,val,size=SIZE_PTR):
        self.val = val
        self.size = size

    def __call__(self,s):
        return Immediate(make_value(s,self.val),make_value(s,self.size))

class SIndirect(StitchValue):
    def __init__(self,offset=0,base=None,index=None,scale=SIZE_PTR,size=SIZE_PTR):
        self.offset = offset
        self.base = base
        self.index = index
        self.scale = scale
        self.size = size

    def __add__(self,b):
        if isinstance(b,int):
            return SIndirect(self.offset+b,self.base,self.index,self.scale,self.size)

        return NotImplemented

    def __sub__(self,b):
        return self.__add__(-b)

    def __call__(self,s):
        return IndirectVar(
            make_value(s,self.offset),
            make_value(s,self.base),
            make_value(s,self.index),
            make_value(s,self.scale),
            make_value(s,self.size))


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

class MainScope(Scope,Branch):
    def cur_single_scope(self) -> 'Branch':
        return self

class DoWhileScope(Scope,Branch):
    def cur_single_scope(self) -> 'Branch':
        return self

class CompareByID(weakref.ref):
    def __init__(self,val,callback=None):
        super().__init__(val,callback)

        self._hash = id(val)

        if __debug__:
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
        for wrap,cd in self._tracked_vals.items():
            if cd is None: continue
            val = wrap()
            if val is not None: yield val,cd

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

    def __init__(self,abi_ : 'abi.Abi',state : Optional[State]=None,cleanup : Optional[StackCleanup]=None) -> None:
        self.cgen = abi_.code_gen(abi_) # type: OpGen
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
        new = DoWhileScope(self.state)
        self._scopes.append(new)
        return self

    def while_(self,test):
        top = self._scopes.pop()
        assert isinstance(top,DoWhileScope)
        assert top.state == self._scopes[-1].cur_single_scope().state
        return self.extend(self.cgen.do_while(top.code,make_value(self,make_cmp(test))))

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

        ob_type = Var()
        (self
            .sub(CType('PyObject',val).ob_refcnt,amount)
            .if_(not_(CType('PyObject',val).ob_refcnt))
            .mov(CType('PyObject',val).ob_type,ob_type)
            .call(CType('PyTypeObject',ob_type).tp_dealloc,val))

        if pyinternals.COUNT_ALLOCS:
            self.call('inc_count',ob_type)

        return self.endif()

    def call(self,func,*args,store_ret=None):
        self._debug_append(self.cgen.call(
            make_arg(self,func),
            [make_arg(self,a) for a in args],
            make_arg(self,store_ret)))

        if isinstance(func,str):
            self.comment(func,inline=True)

        return self

    def get_threadstate(self,dest):
        # as far as I can tell, after expanding the macros and removing the
        # "dynamic annotations" (see dynamic_annotations.h in the CPython
        # headers), this is all that PyThreadState_GET boils down to:
        return self.mov(IndirectVar(pyinternals.raw_addresses['_PyThreadState_Current']),dest)

    def _bin_op(self,a,b,dest,optype):
        a = make_arg(self,a)
        b = make_arg(self,b)
        return self._debug_append(
            self.cgen.bin_op(a,b,a if dest is None else dest,optype))

    def add(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.add)

    def sub(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.sub)

    def mul(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.mul)

    def and_(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.and_)

    def or_(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.or_)

    def xor(self,a,b,dest=None):
        return self._bin_op(a,b,dest,OpType.xor)

    def _unary_op(self,a,dest,optype):
        a = make_arg(self,a)
        return self._debug_append(
            self.cgen.unary_op(a,a if dest is None else dest,optype))

    def neg(self,a,dest=None):
        return self._unary_op(a,dest,UnaryOpType.neg)

    def jmp(self,dest):
        return self._debug_append(self.cgen.jump(dest))

    def jmp_if(self,dest,test):
        test = make_value(self,make_cmp(test))
        return self._debug_append(self.cgen.jump_if(dest,test))

    def shl(self,val,amount,dest):
        val = make_arg(self,val)
        return self._debug_append(self.cgen.shift(val,ShiftDir.left,amount,dest))

    def shr(self,val,amount,dest):
        val = make_arg(self,val)
        return self._debug_append(self.cgen.shift(val,ShiftDir.right,amount,dest))

    def mov(self,src,dest):
        return self._debug_append(self.cgen.move(make_arg(self,src),make_arg(self,dest)))

    def lea(self,addr,dest):
        addr = make_arg(self,addr)
        return self._debug_append(self.cgen.load_addr(addr,dest))

    def get_return_address(self,dest):
        dest = make_arg(self,dest)
        return self._debug_append(self.cgen.get_return_address(dest))

    def touched_indirectly(self,val,loc_type=LocationType.stack):
        """Indicate that 'val' may have been updated by something other than
        the code produced by 'self'.

        This simply makes a value stored in both a register and the stack, be
        stored in only one, depending on which copy was updated.

        """
        self.append(IndirectMod(make_arg(self,val),loc_type))
        return self

    def create_var(self,var,val):
        self.append(CreateVar(make_arg(self,var),make_arg(self,val)))
        return self

    def return_value(self,val):
        return self._debug_append(self.cgen.return_value(make_arg(self,val)))

    def jump_table(self,val,targets):
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
