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


import sys
import operator
import warnings
from collections import namedtuple
from functools import partial,reduce

if __debug__:
    import weakref

from . import pyinternals
from . import debug
from .abi import fits_imm32
from .x86_ops import TEST_MNEMONICS,fits_in_sbyte


CALL_ALIGN_MASK = 0xf
MAX_ARGS = 8
DEBUG_TEMPS = 3
SAVED_REGS = 2 # the number of registers saved *after* the base pointer


def aligned_size(x):
    return (x + CALL_ALIGN_MASK) & ~CALL_ALIGN_MASK


class CompareByIdWrapper:
    def __init__(self,obj):
        self._obj = obj

    def __eq__(self,b):
        if isinstance(b,CompareByIdWrapper):
            b = b._obj

        return self._obj is b

    def __hash__(self):
        return id(self._obj)

    def __getattr__(self,name):
        return getattr(self._obj,name)


address_of = id

def try_len(code):
    tot = 0
    for piece in code:
        f = getattr(piece,'__len__',None)
        if f is None: return None
        tot += f()

    return tot

class StitchValue:
    """A value (usually a register or address) dependent on the ABI and state
    of a Stitch object.

    This is a lazily evaluated value that can be used directly but also serves
    as a DSL for comparisons.

    """
    def __lt__(self,b):
        return OpValue(operator.lt,self,b)

    def __le__(self,b):
        return OpValue(operator.le,self,b)

    def __eq__(self,b):
        return OpValue(operator.eq,self,b)

    def __ne__(self,b):
        return OpValue(operator.ne,self,b)

    def __gt__(self,b):
        return OpValue(operator.gt,self,b)

    def __ge__(self,b):
        return OpValue(operator.ge,self,b)

    def call(self,*args):
        return OpValue((lambda x,*args: x(*args)),self,*args)

    def __call__(self,s):
        raise NotImplementedError()

class AbiConstant(StitchValue):
    def __init__(self,get):
        self.get = get

    def __call__(self,s):
        return self.get(s.abi)

class OpValue(StitchValue):
    def __init__(self,op,*args):
        self.op = op
        self.args = args

    def __call__(self,s):
        return self.op(*[make_value(s,a) for a in self.args])


def _stack_arg_at(abi,n):
    return abi.Address((n - len(abi.r_arg)) * abi.ptr_size + abi.shadow,abi.r_sp)

def arg_dest(n):
    return AbiConstant(lambda abi: abi.r_arg[n] if n < len(abi.r_arg) else _stack_arg_at(abi,n))

def arg_reg(n,fallback):
    return AbiConstant(lambda abi: abi.r_arg[n] if n < len(abi.r_arg) else make_value(abi,fallback))


R_RET = AbiConstant(lambda abi: abi.r_ret)
R_SCRATCH1 = AbiConstant(lambda abi: abi.r_scratch[0])
R_SCRATCH2 = AbiConstant(lambda abi: abi.r_scratch[1])
R_PRES1 = AbiConstant(lambda abi: abi.r_pres[0])
R_PRES2 = AbiConstant(lambda abi: abi.r_pres[1])
R_PRES3 = AbiConstant(lambda abi: abi.r_pres[2])
R_SP = AbiConstant(lambda abi: abi.r_sp)

_TEST = AbiConstant(lambda abi: abi.Test)
TEST_O = AbiConstant(lambda abi: abi.test_O)
TEST_NO = AbiConstant(lambda abi: abi.test_NO)
TEST_B = AbiConstant(lambda abi: abi.test_B)
TEST_NB = AbiConstant(lambda abi: abi.test_NB)
TEST_E = AbiConstant(lambda abi: abi.test_E)
TEST_Z = AbiConstant(lambda abi: abi.test_Z)
TEST_NE = AbiConstant(lambda abi: abi.test_NE)
TEST_NZ = AbiConstant(lambda abi: abi.test_NZ)
TEST_BE = AbiConstant(lambda abi: abi.test_BE)
TEST_A = AbiConstant(lambda abi: abi.test_A)
TEST_S = AbiConstant(lambda abi: abi.test_S)
TEST_NS = AbiConstant(lambda abi: abi.test_NS)
TEST_P = AbiConstant(lambda abi: abi.test_P)
TEST_NP = AbiConstant(lambda abi: abi.test_NP)
TEST_L = AbiConstant(lambda abi: abi.test_L)
TEST_GE = AbiConstant(lambda abi: abi.test_GE)
TEST_LE = AbiConstant(lambda abi: abi.test_LE)
TEST_G = AbiConstant(lambda abi: abi.test_G)

_TEST_LOOKUP = {}
for i,suf in enumerate(TEST_MNEMONICS):
    for s in suf: _TEST_LOOKUP[s] = i


PTR_SIZE = AbiConstant(lambda abi: abi.ptr_size)

class AbiAddress(StitchValue):
    def __init__(self,offset=0,base=None,index=None,scale=1):
        self.offset = offset
        self.base = base
        self.index = index
        self.scale = scale

    def __call__(self,s):
        return s.abi.Address(
            make_value(s,self.offset),
            make_value(s,self.base),
            make_value(s,self.index),
            make_value(s,self.scale))

addr = AbiAddress


class CType:
    def __init__(self,t,base,index=None,scale=PTR_SIZE):
        self.offsets = pyinternals.member_offsets[t]
        self.args = base,index,scale

    def __getattr__(self,name):
        offset = self.offsets.get(name)
        if offset is None: raise AttributeError(name)
        return addr(offset,*self.args)


def raw_addr_if_str(x):
    return pyinternals.raw_addresses[x] if isinstance(x,str) else x

def type_of(r):
    return CType('PyObject',r).ob_type

def type_flags_of(r):
    return CType('PyTypeObject',r).tp_flags


class TupleItem(StitchValue):
    def __init__(self,r,n):
        self.addr = CType('PyTupleObject',r).ob_item
        self.n = n

    def __call__(self,s):
        return self.addr(s) + s.abi.ptr_size * self.n

tuple_item = TupleItem


CMP_LT = 0
CMP_LE = 1
CMP_GT = 2
CMP_GE = 3
CMP_EQ = 4
CMP_NE = 5

CMP_MIRROR = [CMP_GT,CMP_GE,CMP_LT,CMP_LE,CMP_EQ,CMP_NE]
CMP_INVERSE = [CMP_GE,CMP_GT,CMP_LE,CMP_LT,CMP_NE,CMP_EQ]


class RegTest:
    def __init__(self,val,signed=True):
        self.val = val
        self.signed = signed

    def code(self,stitch,dest):
        val = make_value(stitch,self.val)
        assert isinstance(val,stitch.abi.Register)
        stitch.test(val,val).jcc(TEST_Z,dest)

def RegTest_cmp(test):
    def inner(self,b):
        if isinstance(b,RegTest):
            assert self.signed == b.signed
            b = b.val
        return RegCmp(self.val,b,test,self.signed)
    return inner

for f,cmp in (
    ('__lt__',CMP_LT),
    ('__le__',CMP_LE),
    ('__gt__',CMP_GT),
    ('__ge__',CMP_GE),
    ('__eq__',CMP_EQ),
    ('__ne__',CMP_NE)):
    setattr(RegTest,f,RegTest_cmp(cmp))


signed = lambda val: RegTest(val,True)
unsigned = lambda val: RegTest(val,False)

def make_value(s,x):
    if isinstance(x,StitchValue):
        x = x(s)
        assert not isinstance(x,StitchValue)
        return x
    return x

class RegCmp:
    def __init__(self,a,b,cmp,signed=False):
        self.a = a
        self.b = b
        self.cmp = cmp
        self.signed = signed

    def code(self,stitch,dest):
        a = make_value(stitch,self.a)
        b = make_value(stitch,self.b)
        cmp = self.cmp

        assert isinstance(a,(stitch.abi.Register,int)) or isinstance(b,(stitch.abi.Register,int))

        if ((a == 0 and not stitch.address_like(b)) or
                stitch.address_like(a) or
                (b != 0 and isinstance(b,int))):
            a,b = b,a
            cmp = CMP_MIRROR[cmp]

        # test is inverted because we want to jump when the condition is false
        test = ([TEST_GE,TEST_G,TEST_LE,TEST_L,TEST_NE,TEST_E]
                if self.signed else
            [TEST_NB,TEST_A,TEST_BE,TEST_B,TEST_NE,TEST_E])[cmp]

        if b == 0:
            stitch.test(a,a)
        else:
            stitch.cmp(a,b)
        stitch.jcc(test,dest)

    def inverse(self):
        return RegCmp(self.a,self.b,CMP_INVERSE[self.cmp],self.signed)

class RegAnd:
    def __init__(self,a,b):
        self.a = _abivalue_to_regcmp(a)
        self.b = _abivalue_to_regcmp(b)

    def code(self,stitch,dest):
        self.a.code(stitch,dest)
        self.b.code(stitch,dest)

    def inverse(self):
        return RegOr(self.a.inverse(),self.b.inverse())

class RegOr:
    def __init__(self,a,b):
        self.a = _abivalue_to_regcmp(a)
        self.b = _abivalue_to_regcmp(b)

    def code(self,stitch,dest):
        r = stitch.if_(self.a.inverse())
        self.b.code(r,dest)
        return r.endif()

    def inverse(self):
        return RegAnd(self.a.inverse(),self.b.inverse())

and_ = RegAnd
or_ = RegOr

def not_(x):
    return _abivalue_to_regcmp(x).inverse()

def _abivalue_to_regcmp(x):
    if isinstance(x,(RegCmp,RegAnd,RegOr)):
        return x
    if isinstance(x,OpValue):
        if x.op is operator.eq:
            return RegCmp(x.args[0],x.args[1],CMP_EQ)
        if x.op is operator.ne:
            return RegCmp(x.args[0],x.args[1],CMP_NE)

        assert x.op not in [operator.gt,operator.ge,operator.lt,operator.le], 'ordered comparison requires specifying signedness'

    return RegCmp(x,0,CMP_NE)

def code_join(x):
    if isinstance(x[0],bytes):
        return b''.join(x)
    return reduce(operator.add,x)


class JumpTarget:
    used = False
    displacement = None


if __debug__:
    def _get_origin():
        f = sys._getframe(2)
        while f:
            if f.f_code.co_filename != __file__:
                return f.f_lineno,f.f_code.co_filename

            f = f.f_back

        return None

    class SaveOrigin(type):
        def __new__(cls,name,bases,namespace,**kwds):
            old_init = namespace.get('__init__')
            if old_init:
                def __init__(self,*args,**kwds):
                    old_init(self,*args,**kwds)
                    self._origin = _get_origin()

                namespace['__init__'] = __init__

            return type.__new__(cls,name,bases,namespace)

    class SaveOriginWithCompile(type):
        def __new__(cls,name,bases,namespace,**kwds):
            old_init = namespace.get('__init__')
            if old_init:
                def __init__(self,*args,**kwds):
                    old_init(self,*args,**kwds)
                    self._origin = _get_origin()

                namespace['__init__'] = __init__

                old_compile = namespace.get('compile')
                if old_compile:
                    def compile(self,displacement):
                        try:
                            return old_compile(self,displacement)
                        except Exception as e:
                            o = getattr(self,'_origin')
                            if o:
                                e.args += ('origin: {1}:{0}'.format(*o),)
                            raise

                    namespace['compile'] = compile

            return type.__new__(cls,name,bases,namespace)
else:
    SaveOrigin = SaveOriginWithCompile = type


class DelayedCompileEarly(metaclass=SaveOriginWithCompile):
    def compile(self,displacement):
        raise NotImplementedError()

class JumpSource(DelayedCompileEarly):
    def __init__(self,op,abi,target):
        self.op = op
        self.abi = abi
        self.target = target
        target.used = True

    def compile(self,displacement):
        dis = displacement - self.target.displacement
        return self.op(self.abi.Displacement(dis)) if dis else [] # omit useless jumps


CleanupItem = namedtuple('CleanupItem','loc free')

def pyobj_free(s,loc):
    if not isinstance(loc,s.abi.Register):
        s.mov(loc,R_RET)
        loc = R_RET
    s.decref(loc)

class StackCleanupSection(DelayedCompileEarly):
    def __init__(self,s,locations,dest):
        self.next = None
        self.abi = s.abi
        self.locations = frozenset(locations)
        self.dest = dest
        self.start = JumpTarget()

    @property
    def displacement(self):
        assert self.start.displacement is not None
        return self.start.displacement

    def compile(self,displacement):
        s = Stitch(self.abi)(self.start)
        best = None
        if self.locations:
            next_ = self.next
            while next_:
                # the comparison between len(next.locations) and
                # len(best.locations) intentionally uses >= because when there
                # are multiple identical sections, all but the last one will
                # just be jumps to the last one
                if (best is None or len(next_.locations) >= len(best.locations)) and self.locations >= next_.locations:
                    best = next_
                next_ = next_.next

            clean_here = self.locations
            if best is not None: clean_here -= best.locations

            # sorted so the output is deterministic
            for loc,free in sorted(clean_here,key=id): free(s,loc)

        dis = displacement - (best.displacement if best else self.dest.displacement)
        if not best: s.mov(0,R_RET)
        if dis: s.jmp(self.abi.Displacement(dis))

        return s.code

class StackCleanup:
    def __init__(self,destination):
        self.dest = [destination]
        self.last = None

    def push_dest(self,destination):
        assert destination not in self.dest
        self.dest.append(destination)

    def pop_dest(self):
        self.dest.pop()

    def new_section(self,s,locations):
        old = self.last
        self.last = StackCleanupSection(s,locations,self.dest[-1])
        if old is not None and self.dest[-1] is old.dest: old.next = self.last
        return self.last


class DeferredValue:
    def __call__(self,abi):
        raise NotImplementedError()

class DeferredOffsetAddress(DeferredValue):
    def __init__(self,offset,offset_arg,base=None,index=None,scale=1):
        self.offset = offset
        self.offset_arg = offset_arg
        self.base = base
        self.index = index
        self.scale = scale

    def __call__(self,abi):
        assert self.offset_arg is not None
        return abi.Address(self.offset.realize(self.offset_arg) * abi.ptr_size,self.base,self.index,self.scale)

class DeferredOffset(DeferredValue):
    def __init__(self,base,arg,factor=1):
        self.base = base
        self.arg = arg
        self.factor = factor

    def __call__(self,abi):
        return self.base.realize(self.arg) * self.factor

class DeferredOffsetBase(DeferredValue):
    def __init__(self):
        self.base = None

    def __call__(self,abi):
        assert self.base is not None
        return self.base

    def realize(self,arg):
        assert self.base is not None
        return self.base - arg

class DeferredValueInstr(DelayedCompileEarly):
    def __init__(self,abi,op,args):
        self.abi = abi
        self.op = op
        self.args = args

    def compile(self,dispacement):
        return self.op(*[a(self.abi) if isinstance(a,DeferredValue) else a for a in self.args])

def deferred_values(s,op,args):
    args = tuple(make_value(s,a) for a in args)
    if any(isinstance(a,DeferredValue) for a in args):
        return [DeferredValueInstr(s.abi,op,args)]

    return op(*args)

class JumpTable(DelayedCompileEarly):
    # tmp_reg2 can be the same as reg, in which case the value of reg is simply
    # not preserved
    def __init__(self,abi,reg,targets,tmp_reg1,tmp_reg2):
        self.abi = abi
        self.reg = reg
        self.targets = targets
        self.tmp_reg1 = tmp_reg1
        self.tmp_reg2 = tmp_reg2

    def normal_table(self,displacement):
        jmp_size = self.abi.op.JMP_DISP_MIN_LEN
        force_wide = False

        # can we use short jumps?
        dist_extra = 0
        for t in reversed(self.targets):
            if displacement - t.displacement + dist_extra > 127:
                jmp_size = self.abi.op.JMP_DISP_MAX_LEN
                force_wide = True
                break
            dist_extra += jmp_size

        r = []
        dist_extra = 0
        for t in reversed(self.targets):
            r = self.abi.op.jmp(self.abi.Displacement(displacement - t.displacement + dist_extra,force_wide)) + r
            dist_extra += jmp_size

        return self.simple_table(jmp_size) + r


    def simple_table(self,diff):
        s = Stitch(self.abi)
        scale = 1
        if diff in (1,2,4,8):
            scale = diff
            index = self.reg
        else:
            s.imul(self.reg,diff,self.tmp_reg2)
            index = self.tmp_reg2

        rip = getattr(self.abi,'r_rip',None)
        if rip:
            # RIP addresses cannot have indexes so we have to load the start
            # address into a register first

            jmp = code_join(
                self.abi.op.lea(self.abi.Address(0,self.tmp_reg1,index,scale),self.tmp_reg1)
                + self.abi.op.jmp(self.tmp_reg1))
            s.lea(addr(len(jmp),rip),self.tmp_reg1)(jmp)
        else:
            # the offset of the JMP instruction needs to include its own size
            tmp_jmp = code_join(self.abi.op.jmp(self.abi.Address(127,self.tmp_reg1,self.reg,scale)))

            pop = code_join(self.abi.op.pop(self.tmp_reg1))
            jmp = code_join(self.abi.op.jmp(addr(len(pop) + len(tmp_jmp),self.tmp_reg1,index,scale)))

            assert len(jmp) == len(tmp_jmp)

            # TODO: use RelocBuffer to allow using an absolute address offset
            # instead of this CALL + POP chicanery. It could be done by
            # changing pyinternals to always support relocation and tracking
            # the location of relocatable offsets.

            s.call(self.abi.Displacement(0))(pop + jmp)

        return s.code

    def compile(self,displacement):
        if len(self.targets) < 2:
            if not self.targets: return []

            return (
                self.abi.op.test(self.reg,self.reg) +
                self.abi.op.jnz(self.abi.Displacement(displacement - self.targets[0].displacement)))

        diff1 = self.targets[1].displacement - self.targets[0].displacement
        for i in range(2,len(self.targets)):
            if (self.targets[i] - self.targets[i-1]) != diff1:
                return self.normal_table(displacement)

        # If the targets are spaced equally far apart, then the sequence of JMP
        # instructions can be omitted and the target operations can take their
        # place.
        return self.simple_table(-diff1)


class DelayedCompileLate:
    displacement = None

    def compile(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

class JumpRSource(DelayedCompileLate):
    def __init__(self,op,abi,size,target):
        self.op = op
        self.abi = abi
        self.size = size
        self.target = target
        target.used = True

    def compile(self):
        c = code_join(self.op(self.abi.Displacement(self.displacement - self.target.displacement,True)))
        assert len(c) == self.size
        return c

    def __len__(self):
        return self.size

class InnerCall(DelayedCompileLate):
    """A function call with a relative target

    This is just like JumpSource, except the target is a different function and
    the exact offset depends on how much padding is needed between this
    source's function and the target function, which cannot be determined until
    the length of the entire source function is determined.

    :type abi: abi.Abi
    :type target: JumpTarget
    :type jump_instead: bool | None

    """
    def __init__(self,abi,target,jump_instead=False):
        self.op = abi.op.jmp if jump_instead else abi.op.call
        self.length = abi.op.JMP_DISP_MAX_LEN if jump_instead else abi.op.CALL_DISP_LEN
        self.abi = abi
        self.target = target
        target.used = True

    def compile(self):
        r = code_join(self.op(self.abi.Displacement(self.displacement + self.target.displacement,True)))
        assert len(r) == self.length
        return r

    def __len__(self):
        return self.length


class Function:
    def __init__(self,code,padding=0,offset=0,name=None,pyfunc=False,annotation=None):
        self.code = code

        # the size, in bytes, of the trailing padding (nop instructions) in
        # "code"
        self.padding = padding

        self.offset = offset
        self.name = name
        self.pyfunc = pyfunc
        self.annotation = annotation or []

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

def resolve_jumps(op,chunks,end_targets=()):
    displacement = 0
    pad_size = 0
    annot_size = 0
    annots = []

    # this item will be replaced with padding if needed
    late_chunks = [None]

    def early_compile(chunks):
        for c in reversed(chunks):
            if isinstance(c,DelayedCompileEarly):
                yield from early_compile(c.compile(displacement))
            else:
                assert not isinstance(c,list)
                yield c

    for chunk in early_compile(chunks):
        if isinstance(chunk,debug.InnerAnnotation):
            # since appending to a list is O(1) while prepending is O(n), we
            # add the items backwards and reverse the list afterwards
            debug.append_annot(annots,debug.Annotation(chunk,annot_size))
            annot_size = 0

        elif isinstance(chunk,JumpTarget):
            chunk.displacement = displacement

        else:
            if isinstance(chunk,DelayedCompileLate):
                chunk.displacement = displacement

            # items are added backwards for the same reason as above
            late_chunks.append(chunk)
            displacement += len(chunk)
            annot_size += len(chunk)

    assert annot_size == 0 or not annots, "if there are any annotations, there should be one at the start"

    annots.reverse()

    # add padding for alignment
    if CALL_ALIGN_MASK:
        pad_size = aligned_size(displacement) - displacement
        if pad_size:
            late_chunks[0] = code_join(op.nop() * pad_size)
            for et in end_targets:
                et.displacement += pad_size

    code = code_join([(c.compile() if isinstance(c,DelayedCompileLate) else c) for c in reversed(late_chunks) if c is not None])

    for et in end_targets:
        et.displacement += displacement

    return Function(code,pad_size,annotation=annots)



def auto_ne(self,b):
    r = self.__eq__(b)
    if r is NotImplemented: return r
    return not r


class TrackedValue(StitchValue):
    def discard(self,s):
        raise NotImplementedError()

class CountedValue(TrackedValue,metaclass=SaveOrigin):
    def to_addr_movable_val(self,s):
        """If this value is in an address, move it to a register and return the
        new value object, otherwise return self.

        A MOV instruction can't have an address as both a source and
        destination, so if eg: a value needs to be moved to an address and then
        freed, calling this will make sure the value can be moved and wont need
        to be loaded into a register a second time to decrement the reference
        counter.

        :type s: Stitch

        """
        loc = self(s)
        if s.address_like(loc):
            dest = RegObj(s,s.unused_reg(),s.state.owned(self))
            s.mov(loc,dest)

            s.state.disown(self)
            self.discard(s)
            return dest

        return self

    def to_addr(self,s):
        loc = self(s)
        if s.address_like(loc):
            return self

        dest = s.new_stack_value()
        s.mov(loc,dest)

        s.state.disown(self)
        self.discard(s)
        return dest

    def cleanup_free_func(self,s):
        return pyobj_free if s.state.owned(self) else None

class BorrowedValue(TrackedValue):
    def __init__(self,value):
        self.value = value

    def __call__(self,s):
        return self.value(s)

    def discard(self,s,**kwds):
        pass

class BorrowedObj(CountedValue,BorrowedValue):
    pass

def borrow(x):
    if isinstance(x,BorrowedValue): return x
    if isinstance(x,CountedValue): return BorrowedObj(x)
    return BorrowedValue(x)

class StolenValue(StitchValue):
    def __init__(self,val):
        self.val = val
        self.loc = None

    def __call__(self,s):
        if self.loc is None:
            self.loc = self.val(s)

            if isinstance(self.val,CountedValue):
                if s.state.owned(self.val):
                    s.state.disown(self.val)
                else:
                    s.incref(self)

                self.val.discard(s)
            else:
                assert getattr(self.val,'valid',True)
                self.val.valid = False
        return self.loc

steal = StolenValue


def stack_index_to_offset(i):
    # 1 is added to "i" because the byte offset is relative to the end of the
    # stack (the final value is the size of the stack minus this value) and a
    # value of 0 would refer to an item after the end of the stack
    return i + 1

def stack_location(s,i):
    return DeferredOffsetAddress(s.def_stack_offset,stack_index_to_offset(i) if i is not None else None,s.abi.r_sp)

def discard_stack_value(s,offset,**kwds):
    tmp = s.unused_reg()
    # we free the stack slot before calling decref (which calls
    # invalidate_scratch) so a scratch value can use that slot if needed
    s.mov(stack_location(s,offset),tmp)
    s.state.stack[offset] = None
    s.decref(tmp,**kwds)

class PendingStackObj(StitchValue):
    """A Python object with a position on the stack that is yet to be
    determined"""

    def __init__(self):
        self.loc = None
        self.offset = None

    def bind(self,value):
        self.offset = value.offset
        if self.loc:
            self.loc.offset_arg = stack_index_to_offset(self.offset)

    def __call__(self,s):
        if self.loc is None: self.loc = stack_location(s,self.offset)
        return self.loc

class RegValue(TrackedValue):
    """A value stored in a register.

    :type s: Stitch
    :type force: bool | None

    """
    def __init__(self,s,reg=None,force=False):
        if reg is None:
            reg = s.unused_reg()
        else:
            reg = make_value(s,reg)
            if force: s.free_regs(reg)
            assert s.state.regs.get(reg) is None

        s.state.regs[reg] = self
        self.reg = reg

    def discard(self,s):
        assert self.reg and s.state.regs.get(self.reg) is self
        s.state.regs[self.reg] = None

    def free_reg(self,s):
        return False

    def __call__(self,s):
        assert self.reg is not None
        return self.reg

class RegObj(CountedValue,RegValue):
    """A Python object stored in a register.

    :type s: Stitch
    :type owned: bool | None
    :type force: bool | None

    """
    def __init__(self,s,reg=None,owned=True,force=False):
        super().__init__(s,reg,force)
        if owned: s.state.own(self)

    def discard(self,s,**kwds):
        if s.state.owned(self):
            assert self.reg is not None
            s.decref(self.reg,**kwds)
            s.disown(self)
        super().discard(s)

class StackValue(TrackedValue):
    """A value stored in a register that can be pushed to the stack on demand.

    :type s: Stitch

    """
    def __init__(self,s,reg=None):
        self.reg = make_value(s,reg)
        self.offset = None

        if self.reg is not None:
            assert not self.in_register(s)
            s.state.regs[self.reg] = self

    def in_register(self,s):
        return s.state.regs.get(self.reg) is self

    def in_stack(self,s):
        if self.offset is None: return False
        entry = s.state.stack[self.offset]
        return entry and entry[0] is self

    def discard(self,s):
        if self.in_register(s):
            s.state.regs[self.reg] = None

        if self.offset is not None:
            s.state.stack[self.offset] = None

    def __call__(self,s):
        return self.reg if self.in_register(s) else stack_location(s,self.offset)

    def _set_offset(self,offset):
        self.offset = offset
        return self

    def reload_reg(self,s,r=None):
        r = make_value(s,r)

        if not self.in_register(s):
            self.reg = r or s.unused_reg()
            assert s.state.regs.get(self.reg) is None
            s.state.regs[self.reg] = self
            s.mov(stack_location(s,self.offset),self.reg)
        else:
            assert r is None or self.reg == r

    def free_reg(self,s):
        if self.in_register(s):
            s.state.regs[self.reg] = None

        if not self.in_stack(s):
            s.new_stack_value(value_t=self._set_offset)
            s.mov(self.reg,stack_location(s,self.offset))

        return True

class StackObj(CountedValue,StackValue):
    """A Python object stored in a register that can be pushed to the stack on
    demand.

    :type s: Stitch
    :type owned: bool | None

    """
    def __init__(self,s,reg=None,owned=True):
        super().__init__(s,reg)
        if owned: s.state.own(self)

    def discard(self,s,**kwds):
        owned = s.state.owned(self)
        if owned: self.reload_reg(s)
        super().discard(s)
        if owned:
            s.decref(self.reg)
            s.state.disown(self)

    def to_addr(self,s):
        self.free_reg(s)
        return self

class ConstValue(TrackedValue):
    def __init__(self,addr):
        self.addr = addr

    def __call__(self,s):
        return self.addr

    def discard(self,s):
        pass

class ConstObj(CountedValue,ConstValue):
    pass


class BorrowedTempValue(StitchValue):
    def __init__(self,s,val):
        self.s = s
        self.val = val(s) if isinstance(val,type) else val

    def __enter__(self):
        return self.val

    def __exit__(self,*exc):
        pass

    def __call__(self,s):
        return self.val(s)

class TempValue(BorrowedTempValue):
    def __exit__(self,*exc):
        self.val.discard(self.s)


class RegCache(TrackedValue):
    """A value stored in a register that doesn't need to be preserved"""
    def __init__(self,s,reg=None):
        if reg is None: self.value = None
        else: self.validate(s,None if reg is True else reg)

    def validate(self,s,reg=None):
        if reg is None:
            reg = s.unused_reg()
        else:
            reg = make_value(s,reg)

        self.value = RegValue(s,reg)

    def discard(self,s):
        if self.valid: self.value.discard(s)

    @property
    def valid(self):
        return self.value is not None

    def __call__(self,s):
        assert self.valid
        return self.value(s)

    def free_reg(self,s):
        self.discard(s)
        return True


def objarray_free(s,loc):
    s.invoke('free_pyobj_array',loc)

class ObjArrayValue(StackValue):
    valid = True

    def discard(self,s):
        if self.valid:
            objarray_free(s,self)
        super().discard(s)

    @staticmethod
    def cleanup_free_func(s):
        return objarray_free


if __debug__:
    class CountedRef(weakref.ref):
        __slots__ = 'origin','_id'

        def __new__(cls,x,callback=None,origin=None):
            r = weakref.ref.__new__(cls,x,callback)
            r.origin = origin
            r._id = id(x)
            return r

        def __init__(self,x,callback=None,origin=None):
            weakref.ref.__init__(self,x,callback)

        def __eq__(self,b):
            if isinstance(b,CountedRef):
                return self._id == b._id

            if isinstance(b,weakref.ref):
                return self._id == id(b())

            return self._id == id(b)

        def __ne__(self,b):
            return not self.__eq__(b)

        def __hash__(self):
            return hash(self._id)

class State:
    def __init__(self,stack=None,regs=None,owned=None,args=0,unreserved_offset=None):
        self.stack = stack if stack is not None else []
        self.regs = regs if regs is not None else {}
        self.owned_objs = owned if owned is not None else set()
        self.args = args
        self.unreserved_offset = unreserved_offset

    def __eq__(self,b):
        if isinstance(b,State):
            return (self.stack == b.stack
                and self.regs == b.regs
                and self.owned_objs == b.owned_objs
                and self.args == b.args
                and self.unreserved_offset == b.unreserved_offset)

        return NotImplemented

    __ne__ = auto_ne

    if __debug__:
        def _check_ownership(self,val):
            for x in self.owned_objs:
                if x() is None:
                    msg = 'Ref-counted value not freed (in the generated machine code, not the running interpreter).'
                    if x.origin:
                        msg += ' Origin: {1}:{0}'.format(*x.origin)
                    warnings.warn(msg)

    def own(self,x):
        """Mark x as being owned.

        :type x: CountedValue

        """
        if __debug__:
            x = CountedRef(x,self._check_ownership,getattr(x,'_origin',None))
        else:
            x = id(x)

        self.owned_objs.add(x)

    def disown(self,x):
        """Un-mark x as being owned.

        x must be owned, prior to calling this.

        :type x: CountedValue

        """
        if __debug__:
            x = CountedRef(x)
        else:
            x = id(x)

        self.owned_objs.remove(x)

    def owned(self,x):
        """Return true iff x is owned

        :type x: CountedValue
        :rtype: bool

        """
        if __debug__:
            x = CountedRef(x)
        else:
            x = id(x)

        return x in self.owned_objs

    def copy(self):
        return State(self.stack[:],self.regs.copy(),self.owned_objs.copy(),self.args,self.unreserved_offset)

def annot_with_comment(abi,descr,stack):
    return debug.annotate(stack,descr) + abi.comment('annotation: stack={}, {}',stack,descr)

class Stitch:
    """Create machine code concisely using method chaining"""

    def __init__(self,abi,outer=None,jmp_target=None,in_elif=False):
        self.abi = abi
        self.outer = outer
        self.jmp_target = jmp_target
        self.in_elif = in_elif
        self._code = []

        # this is True when the last instruction was an unconditional jump
        # (even if that instruction comes from a DelayedCompiled* object and
        # hasn't been generated yet)
        self.last_op_is_uncond_jmp = False

        if outer:
            self.state = outer.state.copy()
            self.state_if = outer.state
            self.cleanup = outer.cleanup
            self.def_stack_offset = outer.def_stack_offset
            self.special_addrs = outer.special_addrs

            self.usable_tmp_regs = outer.usable_tmp_regs
        else:
            self.state = State()
            self.state_if = None
            self.cleanup = StackCleanup(JumpTarget())
            self.def_stack_offset = DeferredOffsetBase()
            self.special_addrs = {}

            self.usable_tmp_regs = list(abi.r_scratch)
            self.usable_tmp_regs.append(abi.r_ret)
            self.usable_tmp_regs.extend(abi.r_pres)

    def address_like(self,x):
        return isinstance(x,(self.abi.Address,DeferredOffsetAddress))

    @property
    def exc_depth(self):
        return len(self.cleanup.dest)

    def new_stack_value(self,prolog=False,value_t=None,name=None):
        if prolog:
            depth = 0
            if value_t is None: value_t = StackValue(self)._set_offset
        else:
            depth = self.exc_depth
            if value_t is None: value_t = StackObj(self)._set_offset

        for i,s in enumerate(self.state.stack):
            if s is None:
                r = value_t(i)
                self.state.stack[i] = (r,depth)
                return r

        r = value_t(len(self.state.stack))
        self.state.stack.append((r,depth))
        if name: self.special_addrs[name] = r
        return r

    def unused_reg(self):
        for r in self.usable_tmp_regs:
            if self.state.regs.get(r) is None: return r

        for r in reversed(self.usable_tmp_regs):
            if self.state.regs[r].free_reg(self): return r

        assert False,"no free registers"

    def unused_regs(self,num):
        assert num >= 0
        regs = []

        for r in self.usable_tmp_regs:
            if len(regs) == num: return regs
            if self.state.regs.get(r) is None: regs.append(r)

        for r in reversed(self.usable_tmp_regs):
            if len(regs) == num: return regs
            if self.state.regs[r] is not None and self.state.regs[r].free_reg(self): regs.append(r)

        assert len(regs) == num,"not enough free registers"

        return regs

    def temp_reg(self):
        return TempValue(self,RegValue(self,self.unused_reg()))

    def temp_reg_for(self,value):
        loc = value(self)
        if isinstance(loc,self.abi.Register): return BorrowedTempValue(self,value)

        r = self.temp_reg()
        self.mov(loc,r)
        return r

    def invalidate_scratch(self):
        """Make sure there are no scratch registers used.

        This is typically called before issuing function call instructions,
        since scratch registers, by definition, are not preserved across
        function calls.

        """
        self.free_regs(*(self.abi.r_scratch + [self.abi.r_ret]))

    def free_regs(self,*regs):
        for r in regs:
            r = make_value(self,r)
            val = self.state.regs.get(r)
            if val is not None:
                freed = val.free_reg(self)
                assert freed,"register will not be preserved"

    def exc_cleanup(self):
        """Free the stack items created at the current exception handler depth.

        :rtype: Stitch

        """
        self.invalidate_scratch()

        items = []
        for ld in self.state.stack:
            if ld is not None and ld[1] == self.exc_depth:
                cleanup = getattr(ld[0],'cleanup_free_func',None)
                if cleanup is not None:
                    cleanup = cleanup(self)
                    if cleanup is not None:
                        items.append(CleanupItem(make_value(self,ld[0]),cleanup))

        self(self.cleanup.new_section(self,items))
        self.last_op_is_uncond_jmp = True
        return self

    def check_err(self,inverted=False):
        return self.if_(R_RET if inverted else not_(R_RET)).exc_cleanup().endif()

    def reserve_stack(self,move_sp=True):
        """Advance the stack pointer so that self.stack_size * abi.ptr_size
        bytes are reserved and annotate it.

        This function assumes the stack pointer has already been moved by
        len(self.state.stack) * self.abi.ptr_size bytes and only moves the
        stack by the difference.

        If move_sp is False, no instructions are emitted, but an annotation is
        still created.

        :type move_sp: bool
        :rtype: Stitch

        """
        assert self.state.unreserved_offset is None

        self.state.unreserved_offset = len(self.state.stack)

        if move_sp:
            self.sub(DeferredOffset(self.def_stack_offset,len(self.state.stack),self.abi.ptr_size),self.abi.r_sp)

        self(DeferredValueInstr(
            self.abi,
            partial(annot_with_comment,self.abi,debug.PROLOG_END),
            [self.def_stack_offset]))

        return self

    def release_stack(self):
        """Revert the stack pointer and len(self.state.stack) to what they
        were before reserve_stack was called.

        :rtype: Stitch

        """
        assert self.state.unreserved_offset is not None

        self.def_stack_offset.base = aligned_size(
                (len(self.state.stack) + max(MAX_ARGS-len(self.abi.r_arg),0)) * self.abi.ptr_size + self.abi.shadow
            ) // self.abi.ptr_size

        (self
            .add(DeferredOffset(self.def_stack_offset,self.state.unreserved_offset,self.abi.ptr_size),self.abi.r_sp)
            .annotation(debug.EPILOG_START,self.state.unreserved_offset))

        self.state.stack = self.state.stack[0:self.state.unreserved_offset]
        self.state.unreserved_offset = None

        return self

    def save_reg(self,r):
        self.new_stack_value(True)
        return self.push(r).annotation(debug.SaveReg(make_value(self,r)))

    def restore_reg(self,r):
        item,depth = self.state.stack.pop()
        assert depth == 0
        return self.pop(r).annotation(debug.RestoreReg(make_value(self,r)))

    def annotation(self,descr,stack=None):
        if stack is None: stack = len(self.state.stack)
        return self.append(annot_with_comment(self.abi,descr,stack))

    def push_stack_prolog(self,reg,name,descr=None):
        self.mov(reg,self.new_stack_value(True,name=name))
        if descr is not None:
            self.annotation(descr)
        return self

    def func_arg(self,n):
        """Return the address or register where argument n of the current
        function is stored.

        This should not be confused with arg_dest and arg_reg, which operate on
        the arguments of the function about to be called.

        """
        return self.abi.r_arg[n] if n < len(self.abi.r_arg) else DeferredOffsetAddress(self.def_stack_offset,-n,self.abi.r_sp)

    def append(self,x,is_jmp=None):
        self.last_op_is_uncond_jmp = is_jmp
        self._code.extend(x)

        return self

    def branch(self,jmp_target=None,in_elif=False):
        return Stitch(self.abi,self,jmp_target,in_elif)

    def _accommodate_branch(self,stack):
        if len(stack) > len(self.state.stack):
            # the length of the stack array determines how much actual stack
            # space to allocate
            self.state.stack += [None] * (len(stack) - len(self.state.stack))

    def end_branch(self):
        last_op_is_uncond_jmp = self.last_op_is_uncond_jmp
        if self.jmp_target: self(self.jmp_target)

        self.outer.append(self._code)
        self.outer._accommodate_branch(self.state.stack)

        if last_op_is_uncond_jmp:
            self.outer.state = self.state_if
            if self.state_if is None:
                self.outer.last_of_is_uncond_jmp = True
        else:
            assert self.state_if is None or self.state == self.state_if
            self.outer.state = self.state

        return self.outer.end_branch() if self.in_elif else self.outer

    def get_code(self):
        assert self.outer is None
        return self._code

    def set_code(self,c):
        self._code = c

    code = property(get_code,set_code)

    def clear_args(self):
        self.state.args = 0
        return self

    def _if_cond(self,test,in_elif):
        test = make_value(self,test)
        after = JumpTarget()
        self.jcc(~test,after)
        return self.branch(after,in_elif)

    def if_cond(self,test):
        return self._if_cond(test,False)

    def endif(self):
        assert self.jmp_target
        return self.end_branch()

    def else_(self):
        assert self.jmp_target
        e_target = self.jmp_target
        self.jmp_target = JumpTarget()

        self.outer._accommodate_branch(self.state.stack)

        if self.last_op_is_uncond_jmp:
            self.state_if = None
        else:
            self.state_if = self.state
            self.jmp(self.jmp_target)

        self.state = self.outer.state.copy()

        return self(e_target)

    def elif_cond(self,test):
        return self.else_()._if_cond(test,True)

    def _if_(self,test,in_elif):
        after = JumpTarget()
        _abivalue_to_regcmp(test).code(self,after)
        return self.branch(after,in_elif)

    def if_(self,test):
        return self._if_(test,False)

    def elif_(self,test):
        return self.else_()._if_(test,True)

    def do(self):
        return self.branch()

    def while_cond(self,cond):
        cond = make_value(self,cond)

        clen = try_len(self._code)

        if clen is not None:
            jsize = self.abi.op.JCC_MAX_LEN
            if fits_in_sbyte(clen + self.abi.op.JCC_MIN_LEN):
                jsize = self.abi.op.JCC_MIN_LEN

            self.outer._code += self._code
            self.outer.jcc(cond,self.abi.Displacement(-(clen + jsize)))
        else:
            start = JumpTarget()
            self.outer(start)
            self.outer._code += self._code
            self.outer(JumpRSource(partial(self.abi.op.jcc,cond),self.abi,self.abi.op.JCC_MAX_LEN,start))

        return self.outer

    def comment(self,c,*args):
        return self.append(self.abi.comment(c,*args))

    def inline_comment(self,c):
        assert self._code
        if self.abi.assembly:
            assert self._code[-1].ops and not self._code[-1].ops[-1].annot
            self._code[-1].ops[-1].annot = c

        return self

    def unwind_handler(self,down_to=0):
        # since subsequent exception-unwinds overwrite previous exception-
        # unwinds, we could skip everything before the last except block at
        # the cost of making the functions at unwind_exc and unwind_finally
        # check for nulls when freeing the other stack items

        is_exc = param('hblock').type == BLOCK_EXCEPT
        h_free = just(len)(STATE.stack.pstack) - param('hblock').stack

        @just
        def get_handler_blocks(state,down_to):
            for hblock in reversed(state.handler_blocks):
                assert hblock.stack is not None and len(state.stack.pstack) >= hblock.stack

                if (hblock.stack - (3 if hblock.type == BLOCK_EXCEPT else 6)) < down_to: break

                state.stack.pstack.protected_items.pop()

                yield hblock

        return self.append_for(get_handler_blocks(STATE,down_to),'hblock',Stitch(self)
            .lea(STACK[h_free],R_PRES1)
            .mov(-h_free,R_SCRATCH1)
            .inner_call(if_else(is_exc,
                STATE.util_funcs.unwind_exc,
                STATE.util_funcs.unwind_finally))
            .add_to_stack(if_else(is_exc,3,6) - h_free))

    def end_func(self,exception=False):
        self.mov(-len(STATE.stack.pstack.items),R_PRES1)
        if exception: self.mov(0,R_RET)
        return self.jmp(STATE.end)

    def exc_goto_end(self,reraise=False):
        """Go to the inner most handler block or the end of the function if
        there isn't an appropriate block."""

        pre_stack = 0
        block = state.current_except_or_finally()
        if block:
            pre_stack = block.stack - EXCEPT_VALUES

        extra = len(STATE.stack.pstack) - pre_stack

        self.unwind_handler(pre_stack)
        if block:
            assert extra >= 0
            (self
                .lea(STACK[extra],R_PRES1)
                .mov(-extra,R_SCRATCH1)
                .inner_call(
                    STATE.util_funcs.reraise_exc_handler
                        if reraise else
                    STATE.util_funcs.prepare_exc_handler)
                .jmp(block.target))
        else:
            self.end_func()

        return self

    def incref(self,reg=R_RET,amount=1):
        """Generate instructions equivalent to Py_INCREF"""

        if pyinternals.REF_DEBUG:
            # the registers that would otherwise be undisturbed, must be
            # preserved
            (self
                .mov(R_RET,self.special_addrs['temp_ax'])
                .mov(R_SCRATCH1,self.special_addrs['temp_cx'])
                .mov(R_SCRATCH2,self.special_addrs['temp_dx']))

            # call make_value before fiddling with self.state.regs, in case reg
            # depends on self.state.regs
            reg = make_value(self,reg)

            sregs = self.state.regs
            self.state.regs = {}

            for _ in range(amount):
                self.invoke('Py_IncRef',reg)

            self.state.regs = sregs

            return (self
                .mov(self.special_addrs['temp_dx'],R_SCRATCH2)
                .mov(self.special_addrs['temp_cx'],R_SCRATCH1)
                .mov(self.special_addrs['temp_ax'],R_RET))

        return self.add(amount,CType('PyObject',reg).ob_refcnt)

    def decref(self,reg=R_RET,preserve_reg=None,amount=1):
        """Generate instructions equivalent to Py_DECREF"""
        assert amount > 0

        preserve_reg = make_value(self,preserve_reg)

        if pyinternals.REF_DEBUG:
            if preserve_reg:
                self.mov(preserve_reg,self.special_addrs['temp'])
                used = self.state.regs.get(preserve_reg)

            if amount > 1:
                self.mov(reg,self.special_addrs['temp_cx'])

            self.invoke('Py_DecRef',reg)

            for _ in range(amount-1):
                self.invoke('Py_DecRef',self.special_addrs['temp_cx'])

            if preserve_reg:
                self.mov(self.special_addrs['temp'],preserve_reg)
                if used: self.state.regs[preserve_reg] = used
            elif preserve_reg == 0:
                self.mov(0,R_RET)

            return self

        instr = (self
            .sub(amount,CType('PyObject',reg).ob_refcnt)
            .if_cond(TEST_Z))

        if preserve_reg:
            instr.mov(preserve_reg,self.special_addrs['temp'])
            used = self.state.regs.get(preserve_reg)

        instr.mov(CType('PyObject',reg).ob_type,R_SCRATCH1)

        if pyinternals.COUNT_ALLOCS:
            instr.mov(R_SCRATCH1,self.special_addrs['count_allocs_temp'])

        instr.invoke(CType('PyTypeObject',R_SCRATCH1).tp_dealloc,reg)

        if pyinternals.COUNT_ALLOCS:
            instr.invoke('inc_count',self.special_addrs['count_allocs_temp'])

        if preserve_reg:
            instr.mov(self.special_addrs['temp'],preserve_reg)
            if used: self.state.regs[preserve_reg] = used
        elif preserve_reg == 0:
            instr.mov(0,R_RET)

        return instr.endif()

    def fit_addr(self,addr,reg=None,obj=False):
        """If the pyinternals.raw_addresses[addr] can fit into a 32-bit
        immediate value without sign-extension, return an instance of
        Const(Value/Obj), otherwise load the address into reg and return an
        instance of Reg(Value/Obj)."""

        addr = pyinternals.raw_addresses[addr]
        if fits_imm32(self.abi,addr): return (ConstObj if obj else ConstValue)(addr)

        if reg is None: reg = self.unused_reg()
        self.mov(addr,reg)
        return (RegObj if obj else RegValue)(self,reg)

    def _stack_arg_at(self,n):
        assert n >= len(self.abi.r_arg)
        return self.abi.Address((n-len(self.abi.r_arg))*self.abi.ptr_size + self.abi.shadow,self.abi.r_sp)

    def arg_dest(self,n):
        return self.abi.r_arg[n] if n < len(self.abi.r_arg) else self._stack_arg_at(n)

    def push_arg(self,x,tempreg=None,n=None):
        x = make_value(self,x)

        if not tempreg: tempreg = self.unused_reg()

        if n is None:
            n = self.state.args
        else:
            self.state.args = max(self.state.args,n)


        if n < len(self.abi.r_arg):
            reg = self.abi.r_arg[n]
            self.mov(x,reg)
        else:
            dest = self._stack_arg_at(n)

            if self.address_like(x):
                self.mov(x,tempreg).mov(tempreg,dest)
            else:
                self.mov(x,dest)

        if n == self.state.args: self.state.args += 1

        return self

    def call(self,func):
        func = make_value(self,func)
        self.state.args = 0
        self.invalidate_scratch()
        if isinstance(func,str):
            return (self
                .mov(pyinternals.raw_addresses[func],R_RET)
                .append(self.abi.op.call(self.abi.r_ret))
                .inline_comment(func))

        return self.append(self.abi.op.call(func))

    def invoke(self,func,*args,tmpreg=None):
        for a in args:
            self.push_arg(a,tmpreg)

        return self.call(func)

    def arg_reg(self,tempreg=None,n=None):
        """If the nth argument is stored in a register, return that register.
        Otherwise, return tempreg.

        Since push_arg will emit nothing when the source and destination are
        the same, this can be used to eliminate an extra push with opcodes that
        require a register destination. If the given function argument is
        stored in a register, arg_reg will return that register and when passed
        to push_arg, push_arg will emit nothing. If not, tempreg will be
        returned and push_arg will emit the appropriate MOV instruction.

        :type n: int | None

        """
        if n is None: n = self.state.args
        return (self.abi.r_arg[n] if n < len(self.abi.r_arg)
                else (tempreg or self.unused_reg()))

    def inner_call(self,target,jump_instead=False):
        self.invalidate_scratch()
        return self(InnerCall(self.abi,target,jump_instead),is_jmp=jump_instead)

    def get_threadstate(self,reg):
        # as far as I can tell, after expanding the macros and removing the
        # "dynamic annotations" (see dynamic_annotations.h in the CPython
        # headers), this is all that PyThreadState_GET boils down to:
        ts = pyinternals.raw_addresses['_PyThreadState_Current']
        if fits_imm32(self.abi,ts):
            return self.mov(addr(ts),reg)
        return self.op.mov(ts,reg).mov(addr(base=reg),reg)

    def jmp(self,target):
        target = make_value(self,target)
        if isinstance(target,JumpTarget):
            self(JumpSource(self.abi.op.jmp,self.abi,target))
        else:
            self._code += self.abi.op.jmp(target)

        self.last_op_is_uncond_jmp = True
        return self

    def jcc(self,test,target):
        test = make_value(self,test)
        target = make_value(self,target)

        op = self.abi.op.jcc
        return self(JumpSource(partial(op,test),self.abi,target) if isinstance(target,JumpTarget) else op(test,target))

    def cmovcc(self,test,a,b):
        test = make_value(self,test)
        a = make_value(self,a)
        b = make_value(self,b)

        if self.abi.has_cmovcc:
            return self.append(self.abi.op.cmovcc(test,a,b))

        return self.if_cond(test).mov(a,b).endif()

    def mov_addrs(self,a,b,tmp_reg=None):
        a = make_value(self,a)
        b = make_value(self,b)
        if self.address_like(a) and self.address_like(b):
            if tmp_reg is None: tmp_reg = self.unused_reg()
            return self.mov(a,tmp_reg).mov(tmp_reg,b)

        return self.mov(a,b)

    def jump_table(self,reg,targets,tmp_reg1=None,tmp_reg2=None):
        """Create a jump table using "reg" as the index and "targets" as the
        destinations.

        "targets" isn't used until the code is passed to resolve_jumps, so an
        empty sequence can be passed, to be filled later.

        "tmp_reg2" can be the same as "reg", in which case "reg" is simply not
        preserved.

        """
        reg = make_value(self,reg)
        if tmp_reg1 is None:
            if tmp_reg2 is None:
                tmp_reg1,tmp_reg2 = self.unused_regs(2)
            else:
                tmp_reg1 = self.unused_reg()
                tmp_reg2 = make_value(self,tmp_reg2)
        else:
            tmp_reg1 = make_value(self,tmp_reg1)
            if tmp_reg2 is None:
                tmp_reg2 = self.unused_reg()
            else:
                tmp_reg2 = make_value(self,tmp_reg2)

        return self(JumpTable(self.abi,reg,targets,tmp_reg1,tmp_reg2))

    def __getattr__(self,name):
        if name.startswith('j'):
            xcc = self.jcc
            suffix = name[1:]
        elif name.startswith('cmov'):
            xcc = self.cmovcc
            suffix = name[4:]
        else:
            return lambda *args: self.append(deferred_values(self,getattr(self.abi.op,name),args))

        try:
            test = self.abi.Test(_TEST_LOOKUP[suffix])
        except KeyError:
            assert getattr(self.abi.op,name,None) is None
            raise AttributeError(name)

        return partial(xcc,test)

    def __call__(self,op,*,is_jmp=False):
        assert not isinstance(op,list)
        self._code.append(op)
        self.last_op_is_uncond_jmp = is_jmp
        return self

    def __getitem__(self,b):
        """Equivalent to b(self); return self.

        This is to allow calling arbitrary functions on chains of Stitch
        operations without breaking the sequential flow.

        """
        b(self)
        return self


def destitch(x):
    return x.code if isinstance(x,Stitch) else x

