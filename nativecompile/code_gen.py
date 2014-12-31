
import sys
import operator
import warnings
import itertools
from functools import partial,reduce

from . import pyinternals
from . import debug
from .abi import fits_imm32
from .x86_ops import TEST_MNEMONICS


CALL_ALIGN_MASK = 0xf
MAX_ARGS = 6
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
    """A register or address dependent on the ABI and state of a Stitch object.
    
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
R_SP = AbiConstant(lambda abi: abi.r_sp)
R_BP = AbiConstant(lambda abi: abi.r_bp)

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

_ADDRESS = AbiConstant(lambda abi: abi.Address)
def addr(offset=0,base=None,index=None,scale=1):
    return _ADDRESS.call(offset,base,index,scale)


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


class FrameAddr(StitchValue):
    def __init__(self,index):
        self.i = index
    
    def __call__(self,s):
        return s.abi.Address(s.abi.ptr_size * -(SAVED_REGS+1+self.i),s.abi.r_bp)

a = (FrameAddr(i) for i in itertools.count(0))

FRAME = next(a)
GLOBALS = next(a)
BUILTINS = next(a)
LOCALS = next(a)
DEFAULT_TEMP = next(a)

if pyinternals.REF_DEBUG:
    TEMP_AX = next(a)
    TEMP_CX = next(a)
    TEMP_DX = next(a)
elif pyinternals.COUNT_ALLOCS:
    COUNT_ALLOCS_TEMP = next(a)

del a


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
        assert isinstance(self.val,AbstractRegister)
        stitch.test(self.val,self.val).jcc(TEST_Z,dest)

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
        return x(s)
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
        
        if ((a == 0 and not isinstance(b,stitch.abi.Address)) or
                isinstance(a,stitch.abi.Address) or
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
        b.code(r,dest)
        return r.endif()
    
    def inverse(self):
        return RegAnd(self.a.inverse(),self.b.inverse())

and_ = RegAnd
or_ = RegOr

def not_(x):
    return _abivalue_to_regcmp(x).inverse()

def _andor_regcmp(x,type):
    terms = list(x._terms)
    assert terms
    r = terms.pop()
    while terms:
        r = type(terms.pop(),r)
    return r

def _abivalue_to_regcmp(x):
    if isinstance(x,OpValue):
        if x.op is operator.eq:
            return RegCmp(x.args[0],x.args[1],CMP_EQ)
        if x.op is operator.ne:
            return RegCmp(x.args[0],x.args[1],CMP_NE)
        
        assert x.op not in [operator.gt,operator.ge,operator.lt,operator.le], 'ordered comparison requires specifying signedness'
    
    if isinstance(x,StitchValue):
        return RegCmp(x,0,CMP_NE)
    
    assert isinstance(x,(RegCmp,RegAnd,RegOr))
    return x

def code_join(x):
    if isinstance(x[0],bytes):
        return b''.join(x)
    return reduce(operator.add,x)


class JumpTarget:
    used = False
    displacement = None


if __debug__:
    class SaveOrigin(type):
        def __new__(cls,name,bases,namespace,**kwds):
            old_init = namespace.get('__init__')
            if old_init:
                def __init__(self,*args,**kwds):
                    old_init(self,*args,**kwds)
                    
                    # determine where this was defined
                    f = sys._getframe(1)
                    while f and f.f_code.co_filename == __file__: f = f.f_back
                    
                    if f: self._origin = (f.f_lineno,f.f_code.co_filename)
                
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
    
    class DelayedCompileEarly(metaclass=SaveOrigin):
        pass
else:
    class DelayedCompileEarly:
        pass

class JumpSource(DelayedCompileEarly):
    def __init__(self,op,abi,target):
        self.op = op
        self.abi = abi
        self.target = target
        target.used = True

    def compile(self,displacement):
        dis = displacement - self.target.displacement
        return self.op(self.abi.Displacement(dis)) if dis else [b''] # omit useless jumps


class StackCleanupSection(DelayedCompileEarly):
    def __init__(self,s,locations,dest):
        self.next = None
        self.abi = s.abi
        self.locations = frozenset(make_value(s,l) for l in locations)
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
            next = self.next
            while next:
                # the comparison between len(next.locations) and
                # len(best.locations) intentionally uses >= because when there
                # are multiple identical sections, all but the last one will
                # just be jumps to the last one
                if (best is None or len(next.locations) >= len(best.locations)) and self.locations >= next.locations:
                    best = next
                next = next.next
            
            clean_here = self.locations
            if best is not None: clean_here -= best.locations
            
            for obj in clean_here:
                if not isinstance(obj,self.abi.Register):
                    s.mov(obj,R_RET)
                    obj = R_RET
                s.decref(obj)
            
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
    pass

class DeferredOffsetAddress(DeferredValue):
    def __init__(self,offset,offset_arg,base=None,index=None,scale=1):
        self.offset = offset
        self.offset_arg = offset_arg
        self.base = base
        self.index = index
        self.scale = scale
    
    def __call__(self,abi):
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


class DelayedCompileLate:
    displacement = None

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
    late_chunks = [b'']
    
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


class CountedValue(StitchValue):
    def borrow(self):
        return BorrowedValue(self) if self.owned else self
    
    def steal(self,s):
        # this test is needed because self.owned is not always assignable, but
        # a value of True guarantees that it is
        if self.owned:
            self.owned = False
        else:
            s.incref(self)
        
        self.discard(s)
    
    def to_addr_movable_val(self,s):
        """If this value is in an address, move it to a register and return the
        new value object, otherwise return self.
        
        A MOV instruction can't have an address as both a source and
        destination, so if eg: a value needs to be moved to an address and then
        freed, calling this will make sure the value can be moved and wont need
        to be loaded into a register a second time to decrement the reference
        counter.
        
        """
        
        loc = self(s)
        if isinstance(loc,(s.abi.Address,DeferredOffsetAddress)):
            dest = RegValue(s,s.unused_reg(),self.owned)
            s.mov(loc,dest)
            
            # this test is needed because self.owned is not always assignable,
            # but a value of True guarantees that it is
            if self.owned: self.owned = False
            
            self.discard(s)
            return dest
        
        return self
    
    if __debug__:
        def __del__(self):
            if getattr(self,'owned',False):
                warnings.warn('ref-counted value not freed (in the generated machine code, not the running interpreter)')

class BorrowedValue(CountedValue):
    def __init__(self,value):
        assert value.owned
        self.value = value
    
    owned = False
    
    def __call__(self,s):
        return self.value.location(s)
    
    def discard(self,s,**kwds):
        pass

def stack_location(s,offset):
    # 1 is added to "offset" because the byte offset is relative to the end of
    # the stack (the final value is the size of the stack minus this value) and
    # a value of 0 would refer to an item after the end of the stack
    return DeferredOffsetAddress(s.def_stack_offset,offset + 1,s.abi.r_sp)

class RawStackValue(StitchValue):
    """A miscellaneous value stored on the stack"""
    
    def __init__(self,offset):
        assert offset is not None
        self.offset = offset
    
    def discard(self,s,**kwds):
        if self.owned:
            assert self.offset is not None
            
            s.state.stack[self.offset] = None
            self.owned = False
        
        self.offset = None
    
    def __call__(self,s):
        return stack_location(s,self.offset)


def discard_stack_value(s,offset,**kwds):
    tmp = s.unused_reg()
    # we free the stack slot before calling decref (which calls
    # invalidate_scratch) so a scratch value can use that slot if needed
    s.mov(stack_location(s,offset),tmp)
    s.state.stack[offset] = None
    s.decref(tmp,**kwds)

class StackValue(RawStackValue,CountedValue):
    """A Python object stored on the stack"""
    
    def __init__(self,offset):
        super().__init__(offset)
        self.owned = True
    
    def discard(self,s,**kwds):
        if self.owned:
            assert self.offset is not None

            discard_stack_value(s,self.offset,**kwds)
            self.owned = False
        
        self.offset = None

class RegValue(CountedValue):
    """A Python object stored in a register"""
    
    def __init__(self,s,reg,owned=True):
        reg = make_value(s,reg)
        assert reg not in s.state.used_regs
        self.owned = owned
        s.state.used_regs.add(reg)
        self.reg = reg
    
    def discard(self,s,**kwds):
        if self.owned:
            assert self.reg is not None
            s.decref(self.reg,**kwds)
            self.owned = False
        s.state.used_regs.remove(self.reg)
        self.reg = None
    
    def __call__(self,s):
        assert self.reg is not None
        return self.reg

class ScratchValue(CountedValue):
    """A Python object stored in a scratch register that can be pushed to the
    stack on demand"""
    
    def __init__(self,s,reg):
        reg = make_value(s,reg)
        assert reg not in s.state.used_regs
        self.reg = reg
        s.state.used_regs.add(reg)
        s.state.scratch.add(CompareByIdWrapper(self))
        self.offset = None
        self.owned = True
    
    def in_register(self,s):
        return CompareByIdWrapper(self) in s.state.scratch
    
    def discard(self,s,**kwds):
        inreg = self.in_register(s)
        if inreg:
            s.state.used_regs.remove(self.reg)
            s.state.scratch.remove(CompareByIdWrapper(self))

        if self.owned:
            if inreg:
                s.decref(self.reg)
                if self.offset: s.state.stack[self.offset] = None
            else:
                assert self.offset is not None
                discard_stack_value(s,self.offset,**kwds)
        self.owned = False
    
    def __call__(self,s):
        return self.reg if self.in_register(s) else stack_location(s,self.offset)
    
    def _set_offset(self,offset):
        self.offset = offset
        return self
    
    def reload_reg(self,s,r=None):
        r = make_value(s,r)
        
        if not self.in_register(s):
            self.reg = r or s.unused_reg()
            s.state.used_regs.add(self.reg)
            s.state.scratch.add(CompareByIdWrapper(self))
            s.mov(stack_location(s,self.offset),self.reg)
        else:
            assert r is None or self.reg == r
    
    def invalidate_scratch(self,s):
        s.state.used_regs.remove(self.reg)
        
        if not self.offset:
            s.new_stack_value(value_t=self._set_offset)
            s.mov(self.reg,stack_location(s,self.offset))

def reg_or_scratch_value(s,r):
    """Return an instance of RegValue or ScratchValue depending on whether "r"
    is a "preserved" register or not"""
    r = make_value(s,r)
    return (RegValue if r in s.abi.r_pres else ScratchValue)(s,r)

class ConstValue(CountedValue):
    """A Python constant.
    
    Constants are stored in Python code objects and don't need to have their
    reference counts increased or decreased by the code's execution.
    
    """
    def __init__(self,addr):
        self.addr = addr
    
    def __call__(self,s):
        return self.addr
    
    owned = False
    
    def discard(self,s):
        pass


class RegTemp:
    """A miscellaneous value stored in a register"""
    def __init__(self,s,reg=None):
        if reg is not None:
            reg = make_value(s,reg)
            assert reg not in s.state.used_regs
        else:
            reg = s.unused_reg()
        
        self._s = s
        s.state.used_regs.add(reg)
        self.reg = reg
    
    def discard(self):
        self._s.state.used_regs.discard(self.reg)
    
    def __enter__(self):
        return self
    
    def __exit__(self,*exc):
        self.discard()

class BorrowedRegTemp(RegTemp):
    def __init__(self,s,owner,reg):
        reg = make_value(s,reg)
        if __debug__:
            self._s = s
            self.owner = owner
            self._reg = reg
        else:
            self.reg = reg
    
    if __debug__:
        @property
        def reg(self):
            assert self.owner(self._s) == self._reg,"the register might not have the value anymore"
            return self._reg
    
    def discard(self):
        pass

class RegCache(RegTemp):
    """A value stored in a scratch register that doesn't need to be
    preserved"""
    def __init__(self,s,reg=None):
        self._s = s
        self.reg = reg
        if reg is not None: self.validate(make_value(s,reg))
    
    def validate(self,reg=None):
        assert self.reg is None and (reg is None or reg not in self._s.state.used_regs)
        if reg is None: reg = self._s.unused_reg()
        self.reg = reg
        self._s.state.used_regs.add(reg)
        self._s.state.scratch.add(self)
    
    def discard(self):
        super().discard()
        self._s.state.scratch.discard(self)
        self.reg = None
    
    @property
    def valid(self):
        return self.reg is not None
    
    def invalidate_scratch(self,s):
        assert self._s is s
        s.state.used_regs.discard(self.reg)
        self.reg = None


class State:
    def __init__(self,stack=None,used_regs=None,scratch=None,args=0,unreserved_offset=None):
        self.stack = stack if stack is not None else []
        self.used_regs = used_regs if used_regs is not None else set()
        self.scratch = scratch if scratch is not None else set()
        self.args = args
        self.unreserved_offset = unreserved_offset
    
    def __eq__(self,b):
        if isinstance(b,State):
            return (self.stack == b.stack
                and self.used_regs == b.used_regs
                and self.scratch == b.scratch
                and self.args == b.args
                and self.unreserved_offset == b.unreserved_offset)
        
        return NotImplemented
    
    __ne__ = auto_ne
    
    def copy(self):
        return State(self.stack[:],self.used_regs.copy(),self.scratch.copy(),self.args,self.unreserved_offset)

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
        self.last_op_is_uncond_jmp = False
        self.last_op_is_uncond_jmp_if = False
        
        if outer:
            self.state = outer.state.copy()
            self.cleanup = outer.cleanup
            self.def_stack_offset = outer.def_stack_offset
            
            self.usable_tmp_regs = outer.usable_tmp_regs
        else:
            self.state = State()
            self.cleanup = StackCleanup(JumpTarget())
            self.def_stack_offset = DeferredOffsetBase()
            
            self.usable_tmp_regs = list(abi.r_scratch)
            self.usable_tmp_regs.append(abi.r_ret)
            self.usable_tmp_regs.extend(abi.r_pres)
        
        if __debug__:
            self.state_if = None
    
    @property
    def exc_depth(self):
        return len(self.cleanup.dest)
    
    def new_stack_value(self,prolog=False,value_t=None):
        if prolog:
            depth = 0
            if value_t is None: value_t = RawStackValue
        else:
            depth = self.exc_depth
            if value_t is None: value_t = StackValue
        
        depth = 0 if prolog else self.exc_depth
        for i,s in enumerate(self.state.stack):
            if s is None:
                r = value_t(i)
                self.state.stack[i] = (r,depth)
                return r
        
        r = value_t(len(self.state.stack))
        self.state.stack.append((r,depth))
        return r
    
    def unused_reg(self):
        for r in self.usable_tmp_regs:
            if r not in self.state.used_regs: return r

        assert self.state.scratch,"no free registers"
        
        s = self.state.scratch.pop()
        s.invalidate_scratch(self)
        return s.reg
    
    def unused_regs(self,num):
        assert num >= 0
        regs = []
        
        for r in self.usable_tmp_regs:
            if len(regs) == num: return regs
            if r not in self.state.used_regs: regs.append(r)

        while len(regs) < num:
            assert self.state.scratch,"not enough free registers"
        
            s = self.state.scratch.pop()
            regs.append(s(self))
            
            assert isinstance(regs[-1],self.abi.Register)
            
            s.invalidate_scratch(self)
        
        return regs
    
    def unused_reg_temp(self):
        return RegTemp(self,self.unused_reg())
    
    def reg_temp_for(self,value):
        loc = value(self)
        if isinstance(loc,self.abi.Register): return BorrowedRegTemp(self,value,loc)
        
        r = unused_reg_temp(self)
        self.mov(loc,r.reg)
        return r
    
    def invalidate_scratch(self,*regs):
        """Make sure there are no scratch registers used other than any of
        "regs".
        
        This is typically called before issuing function call instructions,
        since scratch registers, by definition, are not preserved across
        function calls.
        
        """
        while self.state.scratch:
            self.state.scratch.pop().invalidate_scratch(self)
        
        assert self.state.used_regs.difference(regs).isdisjoint(self.abi.r_scratch + [self.abi.r_ret])
    
    def exc_cleanup(self):
        """The stack items created at the current exception handler depth"""
        self(self.cleanup.new_section(self,(ld[0] for ld in self.state.stack if ld is not None and ld[1] == self.exc_depth)))
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
        were before reserve_stack was called."""

        assert self.state.unreserved_offset is not None
        
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
    
    def push_stack_prolog(self,reg,descr=None):
        self.mov(reg,self.new_stack_value(True))
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

    def end_branch(self):
        if self.jmp_target: self(self.jmp_target)
 
        assert self.state_if is None or self.state == self.state_if
        
        self.outer.last_op_is_uncond_jmp = self.last_op_is_uncond_jmp_if and self.last_op_is_uncond_jmp
        self.outer.append(self._code)
        
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
        if __debug__: self.state_if = self.state
        self.state = self.outer.state.copy()
        self.last_op_is_uncond_jmp_if = self.last_op_is_uncond_jmp
        if not self.last_op_is_uncond_jmp:
            self.jmp(self.jmp_target)

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
            jsize = JCC_MAX_LEN
            if fits_in_sbyte(cl + JCC_MIN_LEN):
                jsize = JCC_MIN_LEN
            
            self.outer._code += self._code
            self.outer.jcc(cond,self.abi.Displacement(-(clen + jsize)))
        else:
            start = JumpTarget()
            self.outer(start)
            self.outer._code += self._code
            self.outer(JumpRSource(partial(self.abi.op.jcc,cond),self.abi,JCC_MAX_LEN,start))
        
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
                .mov(R_RET,TEMP_AX)
                .mov(R_SCRATCH1,TEMP_CX)
                .mov(R_SCRATCH2,TEMP_DX))
            
            for _ in range(amount):
                self.invoke('Py_IncRef',reg)
            
            return (self
                .mov(TEMP_DX,R_SCRATCH2)
                .mov(TEMP_CX,R_SCRATCH1)
                .mov(TEMP_AX,R_RET))

        return self.add(amount,CType('PyObject',reg).ob_refcnt)

    def decref(self,reg=R_RET,preserve_reg=None,amount=1):
        """Generate instructions equivalent to Py_DECREF"""
        assert amount > 0
        
        preserve_reg = make_value(self,preserve_reg)

        if pyinternals.REF_DEBUG:
            if preserve_reg:
                self.mov(preserve_reg,DEFAULT_TEMP)
                used = preserve_reg in self.state.used_regs
                if used: self.state.used_regs.remove(preserve_reg)

            if amount > 1:
                self.mov(reg,TEMP_CX)
            
            self.invoke('Py_DecRef',reg)
            
            for _ in range(amount-1):
                self.invoke('Py_DecRef',TEMP_CX)

            if preserve_reg:
                self.mov(DEFAULT_TEMP,preserve_reg)
                if used: self.state.used_regs.add(preserve_reg)
            elif preserve_reg == 0:
                self.mov(0,R_RET)

            return self
        
        instr = (self
            .sub(amount,CType('PyObject',reg).ob_refcnt)
            .if_cond(TEST_Z))
        
        if preserve_reg:
            instr.mov(preserve_reg,DEFAULT_TEMP)
            used = preserve_reg in instr.state.used_regs
            if used: instr.state.used_regs.discard(preserve_reg)
        
        instr.mov(CType('PyObject',reg).ob_type,R_SCRATCH1)

        if pyinternals.COUNT_ALLOCS:
            instr.mov(R_SCRATCH1,COUNT_ALLOCS_TEMP)

        instr.invoke(CType('PyTypeObject',R_SCRATCH1).tp_dealloc,reg)

        if pyinternals.COUNT_ALLOCS:
            instr.invoke('inc_count',COUNT_ALLOCS_TEMP)
        
        if preserve_reg:
            instr.mov(DEFAULT_TEMP,preserve_reg)
            if used: instr.state.used_regs.add(preserve_reg)
        elif preserve_reg == 0:
            instr.mov(0,R_RET)
        
        return instr.endif()
    
    def fit_addr(self,addr,reg):
        """If the pyinternals.raw_addresses[addr] can fit into a 32-bit
        immediate value without sign-extension, return
        pyinternals.raw_addresses[addr], otherwise load the address into reg
        and return reg."""
        
        addr = pyinternals.raw_addresses[addr]
        if fits_imm32(self.abi,addr): return addr
        
        self.mov(addr,reg)
        return reg

    def _stack_arg_at(self,n):
        assert n >= len(self.abi.r_arg)
        return self.abi.Address((n-len(self.abi.r_arg))*self.abi.ptr_size + self.abi.shadow,self.abi.r_sp)

    def arg_dest(self,n):
        return self.abi.r_arg[n] if n < len(self.abi.r_arg) else self._stack_arg_at(n)

    def push_arg(self,x,tempreg=None,n=None):
        x = make_value(self,x)
        if not tempreg: tempreg = R_SCRATCH1

        if n is None:
            n = self.state.args
        else:
            self.state.args = max(self.state.args,n)


        if n < len(self.abi.r_arg):
            reg = self.abi.r_arg[n]
            self.mov(x,reg)
        else:
            dest = self._stack_arg_at(n)

            if isinstance(x,(self.abi.Address,DeferredOffsetAddress)):
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
    
    def invoke(self,func,*args):
        for a in args:
            self.push_arg(a)
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

        """
        if n is None: n = self.state.args
        return (self.abi.r_arg[n] if n < len(self.abi.r_arg)
                else (tempreg or R_SCRATCH1))
    
    def inner_call(self,target,jump_instead=False):
        self.invalidate_scratch(self)
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

    def __getattr__(self,name):
        if name[0] == 'j':
            try:
                test = self.abi.Test(_TEST_LOOKUP[name[1:]])
            except KeyError:
                assert getattr(self.abi.op,name,None) is None
                raise AttributeError(name)
            
            return partial(self.jcc,test)
        
        return lambda *args: self.append(deferred_values(self,getattr(self.abi.op,name),args))

    def __call__(self,op,*,is_jmp=False):
        assert not isinstance(op,list)
        self._code.append(op)
        self.last_op_is_uncond_jmp = is_jmp
        return self
    
    def __getitem__(self,b):
        """Equivalent to b(self); return self
        
        This is to allow calling arbitrary functions on chains of Stitch
        operations without breaking the sequential flow.
        
        """
        b(self)
        return self


def destitch(x):
    return x.code if isinstance(x,Stitch) else x

