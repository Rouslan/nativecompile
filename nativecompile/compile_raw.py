
import sys
import dis
import weakref
import operator
import types
import itertools
import collections
from functools import partial, reduce

from . import pyinternals


PRINT_STACK_OFFSET = False


CALL_ALIGN_MASK = 0xf
MAX_ARGS = 4
PRE_STACK = 2
DEBUG_TEMPS = 3
STACK_EXTRA = 1
SAVED_REGS = 2 # the number of registers saved *after* the base pointer


TPFLAGS_INT_SUBCLASS = 1<<23
TPFLAGS_LONG_SUBCLASS = 1<<24
TPFLAGS_LIST_SUBCLASS = 1<<25
TPFLAGS_TUPLE_SUBCLASS = 1<<26
TPFLAGS_BYTES_SUBCLASS = 1<<27
TPFLAGS_UNICODE_SUBCLASS = 1<<28
TPFLAGS_DICT_SUBCLASS = 1<<29
TPFLAGS_BASE_EXC_SUBCLASS = 1<<30
TPFLAGS_TYPE_SUBCLASS = 1<<31


class NCSystemError(SystemError):
    """A SystemError specific to this package.

    This compiler has certain constraints that the Python interpreter doesn't.
    Bytecode produced by CPython should never violate these contraints but
    arbitrary bytecode might.

    """



def aligned_size(x):
    return (x + CALL_ALIGN_MASK) & ~CALL_ALIGN_MASK


class StackManager:
    """Keeps track of values stored on the stack.

    values are layed out as follows, where s=STACK_ITEM_SIZE, 
    m=self.local_mem_size and n=self.offset (the number of stack values):

        %esp + m - s        last stack item
        %esp + m - 2s       second-last stack item
        ...
        %esp + m - (n-x)s   stack item x
        ...
        %esp + m - (n+1)s   stack item 1
        %esp + m - ns       stack item 0
        %esp + m - (n-1)s   where the next stack value will be stored
        ...
        %esp + ys           func argument y
        ...
        %esp + 2s           func argument 2
        %esp + s            func argument 1
        %esp                func argument 0

    """
    def __init__(self,op,abi,local_mem_size):
        assert local_mem_size % abi.ptr_size == 0

        self.op = op
        self.abi = abi
        self.local_mem_size = local_mem_size

        # The number of items saved locally. When this is None, we don't know
        # what the offset is because the previous opcode is a jump past the
        # current opcode. 'resets' is expected to indicate what the offset
        # should be now.
        self.__offset = 0

        self.args = 0

         # if True, the TOS item is in %eax and has not actually been pushed
         # onto the stack
        self.tos_in_eax = False

        self.resets = []

    def check_stack_space(self):
        if (self.__offset + max(self.args-len(self.abi.r_arg),0)) * self.abi.ptr_size > self.local_mem_size:
            raise NCSystemError("Not enough stack space was reserved. This is either a bug or the code object being compiled has an incorrect value for co_stacksize.")

    def get_offset(self):
        return self.__offset

    def set_offset(self,val):
        if val is None:
            self.__offset = None
        else:
            if val < 0:
                raise NCSystemError("The code being compiled tries to pop more items from the stack than have been pushed")

            self.__offset = val
            self.check_stack_space()

    offset = property(get_offset,set_offset)
    
    def push_stack(self,x):
        self.offset += 1
        return self.op.mov(x,self[0])

    def pop_stack(self,x):
        r = self.op.mov(self[0],x)
        self.offset -= 1
        return r

    def _stack_arg_at(self,n):
        assert n >= len(self.abi.r_arg)
        return self.abi.ops.Address(
            (n-len(self.abi.r_arg))*self.abi.ptr_size + self.abi.shadow,
            self.abi.r_sp)

    def push_arg(self,x,tempreg = None,n = None):
        if not tempreg: tempreg = self.abi.r_scratch[0]

        if n is None:
            n = self.args
        else:
            self.args = max(self.args,n)


        if n < len(self.abi.r_arg):
            reg = self.abi.r_arg[n]
            r = [self.op.mov(x,reg)] if x != reg else []
        else:
            dest = self._stack_arg_at(n)

            if isinstance(x,self.abi.ops.Address):
                r = [self.op.mov(x,tempreg),self.op.mov(tempreg,dest)]
            elif isinstance(x,int):
                r = [self.op.movl(x,dest)]
            else:
                r = [self.op.mov(x,dest)]

        if n == self.args: self.args += 1
        self.check_stack_space()
        return r
    
    def push_tos(self,set_again = False):
        """%eax is needed right now so if the TOS item hasn't been pushed onto 
        the stack, do it now."""
        r = [self.push_stack(self.abi.r_ret)] if self.tos_in_eax else []
        self.tos_in_eax = set_again
        return r
    
    def use_tos(self,set_again = False):
        r = self.tos_in_eax
        self.tos_in_eax = set_again
        return r

    def tos(self):
        return self.abi.r_ret if self.tos_in_eax else self[0]

    def conditional_jump(self,target):
        # I'm not sure if a new reset will ever need to be added anywhere except
        # the front
        for i,r in enumerate(self.resets):
            if target == r[0]:
                assert self.offset == r[1]
                return
            if target < r[0]:
                self.resets.insert(i,(target,self.offset))
                return

        self.resets.append((target,self.offset))

    def unconditional_jump(self,target):
        self.conditional_jump(target)
        self.offset = None

    def current_pos(self,pos):
        if self.resets:
            assert pos <= self.resets[0][0]
            if pos == self.resets[0][0]:
                off = self.resets.pop(0)[1]

                assert self.offset is None or off == self.offset
                assert not self.tos_in_eax

                if self.offset is None:
                    self.offset = off

    def __getitem__(self,n):
        """Get the address of the nth stack item.

        Negative indices are allowed as long as you're sure there is space for
        more stack items.

        """
        offset = self.local_mem_size - (self.offset - n) * self.abi.ptr_size
        assert offset >= 0
        return self.abi.ops.Address(offset,self.abi.r_sp)

    def func_arg(self,n):
        """Return the address or register where argument n of the current
        function is stored.

        This should not be confused with push_arg and arg_reg, which operate on
        the arguments of the function about to be called.

        """
        if n < len(self.abi.r_arg):
            return self.abi.r_arg[n]

        addr = self._stack_arg_at(n)
        addr.offset += self.local_mem_size
        return addr

    def call(self,func):
        self.args = 0
        if isinstance(func,str):
            return [
                self.op.mov(pyinternals.raw_addresses[func],self.abi.r_ret),
                self.op.call(self.abi.r_ret)]

        return [self.op.call(func)]

    def arg_reg(self,tempreg=None,n=None):
        """If the nth argument is stored in a register, return that register.
        Otherwise, return tempreg.

        Since push_arg will emit nothing when the source and destination are the
        same, this can be used to eliminate an extra push with opcodes that
        require a register destination. If the given function argument is stored
        in a register, arg_reg will return that register and when passed to
        push_arg, push_arg will emit nothing. If not, tempreg will be returned
        and push_arg will emit the appropriate MOV instruction.

        """
        if n is None: n = self.args
        return (self.abi.r_arg[n] if n < len(self.abi.r_arg) 
                else (tempreg or self.abi.r_scratch[0]))



class JumpTarget:
    used = False
    displacement = None


class JumpSource:
    def __init__(self,op,abi,target):
        self.op = op
        self.abi = abi
        self.target = target
        target.used = True
    
    def compile(self,displacement):
        dis = displacement - self.target.displacement
        return self.op(self.abi.ops.Displacement(dis)) if dis else b'' #optimize away useless jumps


class DelayedCompile:
    pass


class InnerCall(DelayedCompile):
    """A function call with a relative target

    This is just like JumpSource, except the target is a different function and
    the exact offset depends on how much padding is needed between this source's
    function and the target function, which cannot be determined until the
    length of the entire source function is determined.

    """
    def __init__(self,opset,abi,target):
        self.opset = opset
        self.abi = abi
        self.target = target
        target.used = True
        self.displacement = None

    def compile(self):
        return self.opset.call(self.abi.ops.Displacement(self.displacement - self.target.displacement))

    def __len__(self):
        return self.abi.ops.CALL_DISP_LEN


class JumpRSource(DelayedCompile):
    def __init__(self,op,abi,size,target):
        self.op = op
        self.abi = abi
        self.size = size
        self.target = target
        target.used = True
        self.displacement = None

    def compile(self):
        c = self.op(self.abi.ops.Displacement(self.displacement - self.target.displacement,True))
        assert len(c) == self.size
        return c

    def __len__(self):
        return self.size



#def disassemble(co):
#    code = co.co_code
#    labels = findlabels(code)
#
#    n = len(code)
#    i = 0
#    extended_arg = 0
#    free = None
#    while i < n:
#        op = code[i]
#
#        print(repr(i).rjust(4), end=' ')
#        print(opname[op].ljust(20), end=' ')
#        i = i+1
#        if op >= HAVE_ARGUMENT:
#            oparg = code[i] + code[i+1]*256 + extended_arg
#            extended_arg = 0
#            i = i+2
#            if op == EXTENDED_ARG:
#                extended_arg = oparg*65536
#            print(repr(oparg).rjust(5), end=' ')
#            if op in hasconst:
#                print('(' + repr(co.co_consts[oparg]) + ')', end=' ')
#            elif op in hasname:
#                print('(' + co.co_names[oparg] + ')', end=' ')
#            elif op in hasjrel:
#                print('(to ' + repr(i + oparg) + ')', end=' ')
#            elif op in haslocal:
#                print('(' + co.co_varnames[oparg] + ')', end=' ')
#            elif op in hascompare:
#                print('(' + cmp_op[oparg] + ')', end=' ')
#            elif op in hasfree:
#                if free is None:
#                    free = co.co_cellvars + co.co_freevars
#                print('(' + free[oparg] + ')', end=' ')


address_of = id


def interleave_ops(regs,steps):
    """Take the operations from 'steps', give each one a different register than
    before, and interleave their instructions

    This is to take maximum advantage of the CPU's instruction pipeline. The
    operations are only interleaved when pyinternals.REF_DEBUG is false.
    Otherwise the operations are arranged sequentially.  This is to allow the
    use of the code from Frame.incref, which cannot be interleaved when
    ref_debug is true.

    """
    items = []
    pending = []

    try:
        if pyinternals.REF_DEBUG:
            while True:
                items.extend(steps(regs[0]))

        while True:
            for reg in regs:
                s = steps(reg)
                for i in range(len(s)):
                    if i == len(pending):
                        pending.append([])
                    pending[i].append(s[i])
            for p in pending:
                items.extend(p)
            pending = []
    except StopIteration:
        for p in pending:
            items.extend(p)
        return items
        


class Tuning:
    prefer_addsub_over_incdec = True
    build_seq_loop_threshhold = 5
    unpack_seq_loop_threshhold = 5


handlers = [None] * 0xFF

def handler(func,name = None):
    opname = (name or func.__name__)[len('_op_'):]
    def inner(f,*extra):
        r = f()
        if isinstance(f.op,f.abi.ops.Assembly):
            r.comment(opname)
        if f.forward_targets and f.forward_targets[0][0] <= f.byte_offset:
            pos,t,pop = f.forward_targets.pop(0)
            assert pos == f.byte_offset
            r.push_tos()(t)
            if pop:
                r.pop_stack(f.r_scratch[1]).decref(f.r_scratch[1])

        f.stack.current_pos(f.byte_offset)

        if PRINT_STACK_OFFSET:
            print('stack: {}  stack items: {}  opcode: {}'.format(
                f.stack.offset,
                f.stack.offset//f.ptr_size + f.stack.tos_in_eax,
                opname),file=sys.stderr)

        return r + func(f,*extra)

    handlers[dis.opmap[opname]] = inner
    return func

def hasconst(func):
    return handler(
        (lambda f,arg: func(f,f.code.co_consts[arg])),
        func.__name__)

def hasname(func):
    return handler(
        (lambda f,arg: func(f,f.code.co_names[arg])),
        func.__name__)


def get_handler(op):
    h = handlers[op]
    if h is None:
        raise Exception('op code {} is not implemented'.format(dis.opname[op]))
    return h



def raw_addr_if_str(x):
    return pyinternals.raw_addresses[x] if isinstance(x,str) else x

class Frame:
    def __init__(self,op,abi,tuning,local_mem_size,code=None,local_name=None,entry_points=None):
        self.code = code
        self.op = op
        self.abi = abi
        self.tuning = tuning
        self.stack = StackManager(op,abi,local_mem_size)
        self._end = JumpTarget()
        self.local_name = local_name
        self.blockends = []
        self.byte_offset = None
        self.next_byte_offset = None
        self.forward_targets = []
        self.entry_points = entry_points

        # Although JUMP_ABSOLUTE could jump to any instruction, we assume
        # compiled Python code only uses it in certain cases. Thus, we can make
        # small optimizations between successive instructions without needing an
        # extra pass over the byte code to determine all the jump targets.
        self.rtargets = {}


        a = itertools.count(abi.ptr_size * -(SAVED_REGS+1),-abi.ptr_size)
        self.FRAME = self.Address(next(a),abi.r_bp)
        self.GLOBALS = self.Address(next(a),abi.r_bp)
        self.BUILTINS = self.Address(next(a),abi.r_bp)
        self.LOCALS = self.Address(next(a),abi.r_bp)
        self.FAST_LOCALS = self.Address(next(a),abi.r_bp)

        # these are only used when pyinternals.REF_DEBUG is True
        self.TEMP_EAX = self.Address(next(a),abi.r_bp)
        self.TEMP_ECX = self.Address(next(a),abi.r_bp)
        self.TEMP_EDX = self.Address(next(a),abi.r_bp)

    @property
    def r_ret(self):
        return self.abi.r_ret

    @property
    def r_scratch(self):
        return self.abi.r_scratch

    @property
    def r_pres(self):
        return self.abi.r_pres

    @property
    def ptr_size(self):
        return self.abi.ptr_size
    
    def check_err(self,inverted=False):
        if inverted:
            return self.if_eax_is_not_zero(
                [self.op.xor(self.r_ret,self.r_ret)] + 
                self.goto_end())

        return self.if_eax_is_zero(self.goto_end())

    def invoke(self,func,*args):
        return reduce(operator.concat,(self.stack.push_arg(raw_addr_if_str(a)) for a in args)) + self.stack.call(func)

    def _if_eax_is(self,test,opcodes):
        if isinstance(opcodes,(bytes,self.abi.ops.AsmSequence)):
            return [
                self.op.test(self.r_ret,self.r_ret),
                self.op.jcc(~test,self.Displacement(len(opcodes))),
                opcodes]
                
        after = JumpTarget()
        return [
            self.op.test(self.r_ret,self.r_ret),
            JumpSource(partial(self.op.jcc,~test),self.abi,after)
        ] + opcodes + [
            after
        ]

    def if_eax_is_zero(self,opcodes): return self._if_eax_is(self.test_Z,opcodes)
    def if_eax_is_not_zero(self,opcodes): return self._if_eax_is(self.test_NZ,opcodes)

    def goto(self,target):
        return JumpSource(self.op.jmp,self.abi,target)

    def goto_end(self):
        return [self.op.lea(self.stack[0],self.r_pres[0]),JumpSource(self.op.jmp,self.abi,self._end)]

    def inc_or_add(self,x):
        return self.op.add(1,x) if self.tuning.prefer_addsub_over_incdec else self.op.inc(x)

    def incb_or_addb(self,x):
        return self.op.addb(1,x) if self.tuning.prefer_addsub_over_incdec else self.op.incb(x)

    def incl_or_addl(self,x):
        return self.op.addl(1,x) if self.tuning.prefer_addsub_over_incdec else self.op.incl(x)

    def dec_or_sub(self,x):
        return self.op.sub(1,x) if self.tuning.prefer_addsub_over_incdec else self.op.dec(x)

    def decb_or_subb(self,x):
        return self.op.subb(1,x) if self.tuning.prefer_addsub_over_incdec else self.op.decb(x)

    def decl_or_subl(self,x):
        return self.op.subl(1,x) if self.tuning.prefer_addsub_over_incdec else self.op.decl(x)

    def incref(self,reg = None):
        if reg is None: reg = self.r_ret
        if pyinternals.REF_DEBUG:
            # the registers that would otherwise be undisturbed, must be preserved
            return ([
                self.op.mov(self.r_ret,self.TEMP_EAX),
                self.op.mov(self.r_scratch[0],self.TEMP_ECX),
                self.op.mov(self.r_scratch[1],self.TEMP_EDX)
            ] + self.invoke('Py_IncRef',reg) + [
                self.op.mov(self.TEMP_EDX,self.r_scratch[1]),
                self.op.mov(self.TEMP_ECX,self.r_scratch[0]),
                self.op.mov(self.TEMP_EAX,self.r_ret)])

        return [self.incl_or_addl(self.Address(pyinternals.REFCNT_OFFSET,reg))]

    def decref(self,reg = None,preserve_eax = False):
        if reg is None: reg = self.r_ret
        if pyinternals.REF_DEBUG:
            inv = self.invoke('Py_DecRef',reg)
            return [self.op.mov(self.r_ret,self.TEMP_EAX)] + inv + [self.op.mov(self.TEMP_EAX,self.r_ret)] if preserve_eax else inv

        assert reg.reg != self.r_scratch[0].reg
        
        mid = []
        
        if preserve_eax:
            mid.append(self.op.mov(self.r_ret,self.stack[-1]))

        mid += [
            self.op.mov(self.Address(pyinternals.TYPE_OFFSET,reg),self.r_pres[1]),
        ]

        mid += self.invoke(
            self.Address(pyinternals.TYPE_DEALLOC_OFFSET,self.r_pres[1]),
            reg)

        if pyinternals.COUNT_ALLOCS:
            mid += self.invoke('inc_count',self.r_pres[1])
        
        if preserve_eax:
            mid.append(self.op.mov(self.stack[-1],self.r_ret))
        
        mid = join(mid)
        
        return [
            self.decl_or_subl(self.Address(pyinternals.REFCNT_OFFSET,reg)),
            self.op.jnz(self.Displacement(len(mid))),
            mid
        ]

    def rtarget(self):
        t = JumpTarget()
        self.rtargets[self.byte_offset] = t
        return t

    def reverse_target(self,offset):
        try:
            return self.rtargets[offset]
        except KeyError:
            raise NCSystemError('unexpected jump target')

    def forward_target(self,at,pop=False):
        assert at > self.byte_offset

        # there will rarely be more than two targets at any given time
        for i,ft in enumerate(self.forward_targets):
            if ft[0] == at:
                assert ft[2] == pop
                return ft[1]
            if ft[0] > at:
                t = JumpTarget()
                self.forward_targets.insert(i,(at,t,pop))
                return t

        t = JumpTarget()
        self.forward_targets.append((at,t,pop))
        return t

    def jump_to(self,op,max_size,to):
        return JumpSource(op,self.abi,self.forward_target(to)) if to > self.byte_offset else JumpRSource(op,self.abi,max_size,self.reverse_target(to))

    def __call__(self):
        return Stitch(self)


def get_abi_ops_x(x):
    return property(lambda self: getattr(self.abi.ops,x))

for x in (
        'Address',
        'Register',
        'Displacement',
        'test_E',
        'test_Z',
        'test_NE',
        'test_NZ',
        'test_L',
        'CALL_DISP_LEN',
        'JCC_MIN_LEN',
        'JCC_MAX_LEN',
        'JMP_DISP_MIN_LEN',
        'JMP_DISP_MAX_LEN',
        'LOOP_LEN'):
    setattr(Frame,x,get_abi_ops_x(x))



def arg1_as_subscr(func):
    """Some syntax sugar that allows calling a method as:
    object.method[arg1](arg2,...)

    arg1 refers to the first argument *after* self.

    """
    class inner:
        def __init__(self,inst):
            self.inst = inst
        def __getitem__(self,arg1):
            return partial(func,self.inst,arg1)

    return property(inner)


def strs_to_addrs(func):
    return lambda *args: func(*map(raw_addr_if_str,args))


def destitch(x):
    return x.code if isinstance(x,Stitch) else x

class Stitch:
    """Generate a sequence of machine code instructions concisely, using method
    chaining"""
    def __init__(self,frame,code=None):
        self.f = frame
        self.code = code or []

    def goto(self,target):
        self.code.append(self.f.goto(target))
        return self

    def push_tos(self,set_again = False):
        self.code += self.f.stack.push_tos(set_again)
        return self

    def push_stack(self,x):
        self.code.append(self.f.stack.push_stack(x))
        return self

    def pop_stack(self,x):
        self.code.append(self.f.stack.pop_stack(x))
        return self

    def push_arg(self,*args,**kwds):
        self.code += self.f.stack.push_arg(*args,**kwds)
        return self

    def clear_args(self):
        self.f.stack.args = 0
        return self

    def add_to_stack(self,n):
        self.f.stack.offset += n
        return self

    def call(self,x):
        self.code += self.f.stack.call(x)
        return self

    @strs_to_addrs
    def add(self,a,b):
        if a == 1 and not self.f.tuning.prefer_addsub_over_incdec:
            if isinstance(b,self.f.Address):
                c = self.f.op.incl(b)
            else:
                c = self.f.op.inc(b)
        else:
            if isinstance(a,int) and isinstance(b,self.f.Address):
                c = self.f.op.addl(a,b)
            else:
                c = self.f.op.add(a,b)

        self.code.append(c)
        return self

    @strs_to_addrs
    def sub(self,a,b):
        if a == 1 and not self.f.tuning.prefer_addsub_over_incdec:
            if isinstance(b,self.f.Address):
                c = self.f.op.decl(b)
            else:
                c = self.f.op.dec(b)
        else:
            if isinstance(a,int) and isinstance(b,self.f.Address):
                c = self.f.op.subl(a,b)
            else:
                c = self.f.op.sub(a,b)

        self.code.append(c)
        return self

    @strs_to_addrs
    def mov(self,a,b):
        if a == 0 and isinstance(b,self.f.Register):
            c = self.f.op.xor(b,b)
        elif isinstance(a,int) and isinstance(b,self.f.Address):
            c = self.f.op.movl(a,b)
        else:
            c = self.f.op.mov(a,b)

        self.code.append(c)
        return self

    def if_eax_is_zero(self,opcodes):
        self.code += self.f.if_eax_is_zero(destitch(opcodes))
        return self

    def if_eax_is_not_zero(self,opcodes):
        self.code += self.f.if_eax_is_not_zero(destitch(opcodes))
        return self

    @arg1_as_subscr
    def if_cond(self,test,opcodes):
        if isinstance(opcodes,(bytes,self.f.abi.ops.AsmSequence)):
            self.code += [
                self.f.op.jcc(~test,self.f.Displacement(len(opcodes))),
                opcodes]
        else:
            after = JumpTarget()
            self.code += [
                JumpSource(partial(self.f.op.jcc,~test),self.f.abi,after)
            ] + destitch(opcodes) + [
                after
            ]

        return self

    def comment(self,c):
        self.code.append(self.f.op.comment(c))

    def __add__(self,b):
        if isinstance(b,Stitch):
            assert self.f is b.f
            return Stitch(self.f,self.code+b.code)
        if isinstance(b,list):
            return Stitch(self.f,self.code+b)
            
        return NotImplemented
    
    def __iadd__(self,b):
        if isinstance(b,Stitch):
            assert self.f is b.f
            self.code += b.code
            return self
        if isinstance(b,list):
            self.code += b
            return self
        
        return NotImplemented

    def __getattr__(self,name):
        func = getattr(self.f.op,name)
        @strs_to_addrs
        def inner(*args):
            self.code.append(func(*args))
            return self
        return inner

    def __call__(self,op):
        self.code.append(op)
        return self


def _forward_list_func(func):
    def inner(self,*args,**kwds):
        self.code += func(self.f,*args,**kwds)
        return self
    return inner

for func in [
    'check_err',
    'invoke',
    'incref',
    'decref',
    'goto_end']:
    setattr(Stitch,func,_forward_list_func(getattr(Frame,func)))



def _binary_op(f,func):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke(func,f.stack[1],tos)
        .check_err()
        .mov(f.stack[1],f.r_scratch[1])
        .mov(f.r_ret,f.stack[1])
        .decref(f.r_scratch[1])
        .pop_stack(f.r_scratch[1])
        .decref(f.r_scratch[1])
    )

@handler
def _op_BINARY_MULTIPLY(f):
    return _binary_op(f,'PyNumber_Multiply')

@handler
def _op_BINARY_TRUE_DIVIDE(f):
    return _binary_op(f,'PyNumber_TrueDivide')

@handler
def _op_BINARY_FLOOR_DIVIDE(f):
    return _binary_op(f,'PyNumber_FloorDivide')

@handler
def _op_BINARY_ADD(f):
    # TODO: implement the optimization that ceval.c uses for unicode strings
    return _binary_op(f,'PyNumber_Add')

@handler
def _op_BINARY_SUBTRACT(f):
    return _binary_op(f,'PyNumber_Subtract')

@handler
def _op_BINARY_SUBSCR(f):
    return _binary_op(f,'PyObject_GetItem')

@handler
def _op_BINARY_LSHIFT(f):
    return _binary_op(f,'PyNumber_Lshift')

@handler
def _op_BINARY_RSHIFT(f):
    return _binary_op(f,'PyNumber_Rshift')

@handler
def _op_BINARY_AND(f):
    return _binary_op(f,'PyNumber_And')

@handler
def _op_BINARY_XOR(f):
    return _binary_op(f,'PyNumber_Xor')

@handler
def _op_BINARY_OR(f):
    return _binary_op(f,'PyNumber_Or')

@handler
def _op_INPLACE_MULTIPLY(f):
    return _binary_op(f,'PyNumber_InPlaceMultiply')

@handler
def _op_INPLACE_TRUE_DIVIDE(f):
    return _binary_op(f,'PyNumber_InPlaceTrueDivide')

@handler
def _op_INPLACE_FLOOR_DIVIDE(f):
    return _binary_op(f,'PyNumber_InPlaceFloorDivide')

@handler
def _op_INPLACE_MODULO(f):
    return _binary_op(f,'PyNumber_InPlaceRemainder')

@handler
def _op_INPLACE_ADD(f):
    # TODO: implement the optimization that ceval.c uses for unicode strings
    return _binary_op(f,'PyNumber_InPlaceAdd')

@handler
def _op_INPLACE_SUBTRACT(f):
    return _binary_op(f,'PyNumber_InPlaceSubtract')

@handler
def _op_INPLACE_LSHIFT(f):
    return _binary_op(f,'PyNumber_InPlaceLshift')

@handler
def _op_INPLACE_RSHIFT(f):
    return _binary_op(f,'PyNumber_InPlaceRshift')

@handler
def _op_INPLACE_AND(f):
    return _binary_op(f,'PyNumber_InPlaceAnd')

@handler
def _op_INPLACE_XOR(f):
    return _binary_op(f,'PyNumber_InPlaceXor')

@handler
def _op_INPLACE_OR(f):
    return _binary_op(f,'PyNumber_InPlaceOr')


@handler
def _op_POP_TOP(f):
    r = f.decref()
    if not f.stack.use_tos():
        r.insert(0,f.stack.pop_stack(f.r_ret))
    return r

@hasname
def _op_LOAD_NAME(f,name):
    return (f()
        .push_tos(True)
        .mov(address_of(name),f.r_pres[0])
        (InnerCall(f.op,f.abi,f.local_name))
        .check_err()
        .incref()
    )

@hasname
def _op_STORE_NAME(f,name):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .push_arg(tos,n=2)
        .push_arg(address_of(name),n=1)
        .mov(f.LOCALS,f.r_ret)
        .if_eax_is_zero(f()
            .clear_args()
            .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_STORE_MSG')
            .goto_end()
        )
        .mov('PyObject_SetItem',f.r_scratch[0])
        .cmpl('PyDict_Type',f.Address(pyinternals.TYPE_OFFSET,f.r_ret))
        .push_arg(f.r_ret,n=0)
        .if_cond[f.test_E](
            f.op.mov(pyinternals.raw_addresses['PyDict_SetItem'],f.r_scratch[0])
        )
        .call(f.r_scratch[0])
        .check_err(True)
        .pop_stack(f.r_ret)
        .decref()
    )

@hasname
def _op_DELETE_NAME(f,name):
    return (f()
        .push_tos()
        .mov(f.LOCALS,f.r_ret)
        .if_eax_is_zero(f()
            .invoke('PyErr_Format',
                'PyExc_SystemError',
                'NO_LOCALS_DELETE_MSG',
                address_of(name))
            .goto_end()
        )
        .invoke('PyObject_DelItem',f.r_ret,address_of(name))
        .if_eax_is_zero(f()
            .invoke('format_exc_check_arg',
                'PyExc_NameError',
                'NAME_ERROR_MSG',
                address_of(name))
            .goto_end()
        )
    )

@hasname
def _op_LOAD_GLOBAL(f,name):
    return (f()
        .push_tos(True)
        .invoke('PyDict_GetItem',f.GLOBALS,address_of(name))
        .if_eax_is_zero(f()
            .invoke('PyDict_GetItem',f.BUILTINS,address_of(name))
            .if_eax_is_zero(f()
                .invoke('format_exc_check_arg',
                     'PyExc_NameError',
                     'GLOBAL_NAME_ERROR_MSG',
                     address_of(name))
                .goto_end()
            )
        )
        .incref()
    )

@hasname
def _op_STORE_GLOBAL(f,name):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyDict_SetItem',f.GLOBALS,address_of(name),tos)
        .check_err(True)
        .pop_stack(f.r_ret)
        .decref()
    )

@hasconst
def _op_LOAD_CONST(f,const):
    if isinstance(const,types.CodeType):
        const = f.entry_points[id(const)][0]

    return (f()
        .push_tos(True)
        .mov(address_of(const),f.r_ret)
        .incref()
    )

@handler
def _op_CALL_FUNCTION(f,arg):
    argreg = f.stack.arg_reg(n=0)
    return (f()
        .push_tos(True)
        .push_arg(arg,n=1)
        .lea(f.stack[0],argreg)
        .push_arg(argreg,n=0)
        .call('call_function')

        # +1 for the function object
        .add_to_stack(-((arg & 0xFF) + ((arg >> 8) & 0xFF) * 2 + 1))

        .check_err()
    )

@handler
def _op_RETURN_VALUE(f):
    return f().goto_end() if f.stack.use_tos() else f().pop_stack(f.r_ret).goto_end()

@handler
def _op_SETUP_LOOP(f,to):
    f.blockends.append(JumpTarget())
    return []

@handler
def _op_POP_BLOCK(f):
    assert not f.stack.tos_in_eax
    return [f.blockends.pop()]

@handler
def _op_GET_ITER(f):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_GetIter',tos)
        .check_err()
        .pop_stack(f.r_scratch[1])
        .push_stack(f.r_ret)
        .decref(f.r_scratch[1])
    )

@handler
def _op_FOR_ITER(f,to):
    argreg = f.stack.arg_reg(n=0)
    return (f()
        .push_tos(True)
        (f.rtarget())
        .mov(f.stack[0],argreg)
        .mov(f.Address(pyinternals.TYPE_OFFSET,argreg),f.r_ret)
        .mov(f.Address(pyinternals.TYPE_ITERNEXT_OFFSET,f.r_ret),f.r_ret)
        .invoke(f.r_ret,argreg)
        .if_eax_is_zero(f()
            .call('PyErr_Occurred')
            .if_eax_is_not_zero(f()
                .invoke('PyErr_ExceptionMatches','PyExc_StopIteration')
                .check_err()
                .call('PyErr_Clear')
            )
            .goto(f.forward_target(f.next_byte_offset + to,True))
        )
    )

@handler
def _op_JUMP_ABSOLUTE(f,to):
    assert to < f.byte_offset
    return f().push_tos()(
        JumpRSource(f.op.jmp,f.abi,f.JMP_DISP_MAX_LEN,f.reverse_target(to)))

@hasname
def _op_LOAD_ATTR(f,name):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_GetAttr',tos,address_of(name))
        .check_err()
        .pop_stack(f.r_scratch[1])
        .push_stack(f.r_ret)
        .decref(f.r_scratch[1])
    )

def _op_pop_jump_if_(f,to,state):
    dont_jump = JumpTarget()
    jop1,jop2 = (f.op.jz,f.op.jg) if state else (f.op.jg,f.op.jz)
    tos = f.stack.tos()
    r = (f()
        .push_tos()
        .invoke('PyObject_IsTrue',tos)
        .pop_stack(f.r_scratch[1])
        .decref(f.r_scratch[1],True)
        .test(f.r_ret,f.r_ret)
        (JumpSource(jop1,f.abi,dont_jump))
        (f.jump_to(jop2,f.JCC_MAX_LEN,to))
        .mov(0,f.r_ret)
        .goto_end()
        (dont_jump)
    )

    if to > f.byte_offset:
        f.stack.conditional_jump(to)

    return r

@handler
def _op_POP_JUMP_IF_FALSE(f,to):
    return _op_pop_jump_if_(f,to,False)

@handler
def _op_POP_JUMP_IF_TRUE(f,to):
    return _op_pop_jump_if_(f,to,True)

def _op_BUILD_(f,items,new,item_offset,deref):
    r = (f()
        .push_tos(True)
        .invoke(new,items)
        .check_err())

    if items:
        if items >= f.tuning.build_seq_loop_threshhold:
            top = f.stack[0]
            top.scale = 4
            if deref:
                f.stack.tos_in_eax = False
                (r
                    .mov(f.Address(item_offset,f.r_ret),f.r_pres[0])
                    .push_stack(f.r_ret)
                    .mov(0,f.r_ret)
                )

                top.index = f.r_ret

                lbody = (
                    f.op.mov(top,f.r_scratch[1]) +
                    f.op.mov(f.r_scratch[1],f.Address(-f.ptr_size,f.r_pres[0],f.r_scratch[0],f.ptr_size)) +
                    f.inc_or_add(f.r_ret))
            else:
                r.mov(0,f.r_pres[0])

                top.index = f.r_pres[0]

                lbody = (
                    f.op.mov(top,f.r_scratch[1]) +
                    f.op.mov(f.r_scratch[1],f.Address(item_offset-f.ptr_size,f.r_ret,f.r_scratch[0],f.ptr_size)) +
                    f.inc_or_add(f.r_pres[0]))

            (r
                .mov(items,f.r_scratch[0])
                (lbody)
                .loop(f.Displacement(-len(lbody) - f.LOOP_LEN)))

            f.stack.offset -= items
        else:
            if deref:
                r.mov(f.Address(item_offset,f.r_ret),f.r_pres[0])

            for i in reversed(range(items)):
                addr = f.Address(f.ptr_size*i,f.r_pres[0]) if deref else f.Address(item_offset+f.ptr_size*i,f.r_ret)
                r.pop_stack(f.r_scratch[1]).mov(f.r_scratch[1],addr)


    return r

@handler
def _op_BUILD_LIST(f,items):
    return _op_BUILD_(f,items,'PyList_New',pyinternals.LIST_ITEM_OFFSET,True)

@handler
def _op_BUILD_TUPLE(f,items):
    return _op_BUILD_(f,items,'PyTuple_New',pyinternals.TUPLE_ITEM_OFFSET,False)

@handler
def _op_STORE_SUBSCR(f):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_SetItem',f.stack[1],tos,f.stack[2])
        .check_err(True)
        .pop_stack(f.r_ret)
        .decref()
        .pop_stack(f.r_ret)
        .decref()
        .pop_stack(f.r_ret)
        .decref()
    )

def _op_make_callable(f,arg,closure):
    annotations = (arg >> 16) & 0x7fff

    # +1 for the code object
    sitems = (arg & 0xff) + ((arg >> 8) & 0xff) * 2 + annotations + 1

    if closure: sitems += 1
    if annotations: sitems += 1

    argreg = f.stack.arg_reg(n=2)
    return (f()
        .push_tos(True)
        .lea(f.stack[0],argreg)
        .invoke('_make_function',int(bool(closure)),arg,argreg)
        .add_to_stack(-sitems)
        .check_err()
    )

@handler
def _op_MAKE_FUNCTION(f,arg):
    return _op_make_callable(f,arg,False)

@handler
def _op_MAKE_CLOSURE(f,arg):
    return _op_make_callable(f,arg,True)

@handler
def _op_LOAD_FAST(f,arg):
    return (f()
        .push_tos(True)
        .mov(f.FAST_LOCALS,f.r_scratch[0])
        .mov(f.Address(f.ptr_size*arg,f.r_scratch[0]),f.r_ret)
        .if_eax_is_zero(f()
            .invoke('format_exc_check_arg',
                'PyExc_UnboundLocalError',
                'UNBOUNDLOCAL_ERROR_MSG',
                address_of(f.code.co_varnames[arg]))
            .goto_end()
        )
        .incref()
    )

@handler
def _op_STORE_FAST(f,arg):
    r = f()
    if not f.stack.use_tos():
        r.pop_stack(f.r_ret)

    item = f.Address(f.ptr_size*arg,f.r_scratch[0])
    return (r
        .mov(f.FAST_LOCALS,f.r_scratch[0])
        .mov(item,f.r_scratch[1])
        .mov(f.r_ret,item)
        .test(f.r_scratch[1],f.r_scratch[1])
        .if_cond[f.test_NZ](
            join(f.decref(f.r_scratch[1]))
        )
    )

@handler
def _op_UNPACK_SEQUENCE(f,arg):
    assert arg > 0

    r = f()
    if f.stack.use_tos():
        r.mov(f.r_ret,f.r_pres[0])
    else:
        r.pop_stack(f.r_pres[0])

    check_list = JumpTarget()
    else_ = JumpTarget()
    done = JumpTarget()

    s_top = f.stack[-1]
    s_top.scale = f.ptr_size

    # a place to temporarily store the sequence
    seq_store = f.stack[-1-arg]

    (r
        .mov(f.Address(pyinternals.TYPE_OFFSET,f.r_pres[0]),f.r_scratch[1])
        .cmp('PyTuple_Type',f.r_scratch[1])
        (JumpSource(f.op.jne,f.abi,check_list))
            .cmpl(arg,f.Address(pyinternals.VAR_SIZE_OFFSET,f.r_pres[0]))
            (JumpSource(f.op.jne,f.abi,else_)))

    if arg >= f.tuning.unpack_seq_loop_threshhold:
        s_top.index = f.r_ret
        body = join(f()
            .mov(f.Address(pyinternals.TUPLE_ITEM_OFFSET-f.ptr_size,f.r_pres[0],f.r_scratch[0],f.ptr_size),f.r_scratch[1])
            .incref(f.r_scratch[1])
            .mov(f.r_scratch[1],s_top)
            .add(1,f.r_ret).code)
        (r
            .mov(0,f.r_ret)
            .mov(arg,f.r_scratch[0])
            (body)
            .loop(f.Displacement(-len(body) - f.LOOP_LEN)))
    else:
        itr = iter(reversed(range(arg)))
        def unpack_one(reg):
            i = next(itr)
            return (f()
                .mov(f.Address(pyinternals.TUPLE_ITEM_OFFSET + f.ptr_size * i,f.r_pres[0]),reg)
                .incref(reg)
                .mov(reg,f.stack[i-arg])).code

        r += interleave_ops([f.r_ret,f.r_scratch[0],f.r_scratch[1]],unpack_one)


    (r
            .goto(done)
        (check_list)
        .cmp('PyList_Type',f.r_scratch[1])
        (JumpSource(f.op.jne,f.abi,else_))
            .cmpl(arg,f.Address(pyinternals.VAR_SIZE_OFFSET,f.r_pres[0]))
            (JumpSource(f.op.jne,f.abi,else_))
                .mov(f.Address(pyinternals.LIST_ITEM_OFFSET,f.r_pres[0]),f.r_scratch[1]))
    
    if arg >= f.tuning.unpack_seq_loop_threshhold:
        s_top.index = f.r_pres[0]
        body = join(f()
            .mov(f.Address(-f.ptr_size,f.r_scratch[1],f.r_scratch[0],f.ptr_size),f.r_ret)
            .incref(f.r_ret)
            .mov(f.r_ret,s_top)
            .add(1,f.r_pres[0]).code)
        (r
            .mov(f.r_pres[0],seq_store)
            .mov(arg,f.r_scratch[0])
            .mov(0,f.r_pres[0])
            (body)
            .loop(f.Displacement(-(len(body) + f.LOOP_LEN)))
            .mov(seq_store,f.r_pres[0]))
    else:
        itr = iter(reversed(range(arg)))
        def unpack_one(reg):
            i = next(itr)
            return (f()
                .mov(f.Address(f.ptr_size * i,f.r_scratch[1]),reg)
                .incref(reg)
                .mov(reg,f.stack[i-arg])).code

        r += interleave_ops([f.r_ret,f.r_scratch[0]],unpack_one)


    f.stack.offset += arg

    p3 = f.stack.arg_reg(n=3)

    return (r
            .goto(done)
        (else_)
            .lea(f.stack[0],p3)
            .invoke('_unpack_iterable',f.r_pres[0],arg,-1,p3)
            .if_eax_is_zero(f()
                .decref(f.r_pres[0])
                .goto_end()
            )
        (done)
        .decref(f.r_pres[0]))

@handler
def _op_UNPACK_EX(f,arg):
    totalargs = 1 + (arg & 0xff) + (arg >> 8)

    r = f()
    if f.stack.use_tos():
        r.mov(f.r_ret,f.r_pres[0])
    else:
        r.pop(f.r_pres[0])

    f.stack.offset += totalargs
    argreg = f.stack.arg_reg(n=3)

    return (r
        .lea(f.stack[0],argreg)
        .invoke('_unpack_iterable',f.r_pres[0],arg & 0xff,arg >> 8,argreg)
        .decref(f.r_pres[0],True)
        .if_eax_is_zero(f()
            .goto_end()
        )
    )

def false_true_addr(swap):
    return map(
        pyinternals.raw_addresses.__getitem__,
        ('Py_True','Py_False') if swap else ('Py_False','Py_True'))

@handler
def _op_COMPARE_OP(f,arg):
    op = dis.cmp_op[arg]

    def pop_args():
        return (f()
            .mov(f.stack[1],f.r_scratch[1])
            .mov(f.r_ret,f.stack[1])
            .decref(f.r_scratch[1])
            .pop_stack(f.r_scratch[1])
            .decref(f.r_scratch[1])
        )

    if op == 'is' or op == 'is not':
        outcome_a,outcome_b = false_true_addr(op == 'is not')

        r = f()
        if f.stack.use_tos(True):
            r.mov(f.r_ret,f.r_scratch[1])
        else:
            r.pop_stack(f.r_scratch[1])

        temp = f.stack[-1]
        return (r
            .cmp(f.r_scratch[1],f.stack[0])
            .mov(outcome_a,temp)
            .if_cond[f.test_E](
                f.op.movl(outcome_b,temp)
            )
            .decref(f.r_scratch[1])
            .pop_stack(f.r_scratch[1])
            .decref(f.r_scratch[1])
            .mov(temp,f.r_ret)
            .incref()
        )

    if op == 'in' or op == 'not in':
        outcome_a,outcome_b = false_true_addr(op == 'not in')

        tos = f.stack.tos()
        return (f()
            .push_tos()
            .invoke('PySequence_Contains',tos,f.stack[1])
            .test(f.r_ret,f.r_ret)
            .if_cond[f.test_L](f()
                .mov(0,f.r_ret)
                .goto_end()
            )
            .mov(outcome_a,f.r_ret)
            .if_cond[f.test_NZ](
                f.op.mov(outcome_b,f.r_ret)
            )
            .incref()
        ) + pop_args()

    if op == 'exception match':
        tos = f.stack.tos()
        return (f()
            .push_tos()
            .invoke('_exception_cmp',f.stack[1],tos)
            .check_err()
        ) + pop_args()

    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_RichCompare',f.stack[1],tos,arg)
        .check_err()
    ) + pop_args()

@handler
def _op_JUMP_FORWARD(f,arg):
    to = f.next_byte_offset + arg
    r = (f()
        .push_tos()
        .goto(f.forward_target(to))
    )
    f.stack.unconditional_jump(to)
    return r

@handler
def _op_RAISE_VARARGS(f,arg):
    r = f()
    
    if arg == 2:
        p0 = f.stack.arg_reg(tempreg=f.r_scratch[1],n=0)
        p1 = f.r_ret
        if not f.stack.use_tos():
            p1 = f.stack.arg_reg(tempreg=f.r_ret,n=1)
            r.pop_stack(p1)

        (r
            .pop_stack(p0)
            .push_arg(p1,n=1)
            .push_arg(p0,n=0)
        )
    elif arg == 1:
        p0 = f.r_ret
        if not f.stack.use_tos():
            p0 = f.stack.arg_reg(n=0)
            r.pop_stack(p0)

        (r
            .push_arg(0,n=1)
            .push_arg(p0,n=0)
        )
    elif arg == 0:
        (r
            .push_tos()
            .push_arg(0)
            .push_arg(0)
        )
    else:
        raise SystemError("bad RAISE_VARARGS oparg")

    # We don't have to worry about decrementing the reference counts. _do_raise
    # does that for us.
    return (r
        .call('_do_raise')
        .goto_end()
    )

@handler
def _op_BUILD_MAP(f,arg):
    return (f()
        .push_tos(True)
        .invoke('_PyDict_NewPresized',arg)
        .check_err()
    )

def _op_map_store_add(offset,f):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyDict_SetItem',f.stack[2+offset],tos,f.stack[1])
        .check_err(True)
        .pop_stack(f.r_scratch[1])
        .decref(f.r_scratch[1])
        .pop_stack(f.r_scratch[1])
        .decref(f.r_scratch[1])
    )

@handler
def _op_STORE_MAP(f):
    return _op_map_store_add(0,f)

@handler
def _op_MAP_ADD(f,arg):
    return _op_map_store_add(arg-1,f)

@handler
def _op_LOAD_BUILD_CLASS(f):
    return (f()
        .push_tos(True)
        .invoke('PyDict_GetItemString',f.BUILTINS,'__build_class__')
        .if_eax_is_zero(f()
            .invoke('PyErr_SetString','PyExc_ImportError','BUILD_CLASS_ERROR_MSG')
            .goto_end()
        )
        .incref()
   )

@handler
def _op_STORE_LOCALS(f):
    return (f()
        .push_tos()
        .mov(f.LOCALS,f.r_ret)
        .if_eax_is_not_zero(
            join(f.decref())
        )
        .pop_stack(f.r_ret)
        .mov(f.FRAME,f.r_scratch[0])
        .mov(f.r_ret,f.LOCALS)
        .mov(f.r_ret,f.Address(pyinternals.FRAME_LOCALS_OFFSET,f.r_scratch[0]))
    )

@hasname
def _op_STORE_ATTR(f,name):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_SetAttr',tos,address_of(name),f.stack[1])
        .mov(f.r_ret,f.r_pres[0])
        .pop_stack(f.r_ret)
        .decref()
        .pop_stack(f.r_ret)
        .decref()
        .test(f.r_pres[0],f.r_pres[0])
        .if_cond[f.test_NZ](f()
            .mov(0,f.r_ret)
            .goto_end()
        )
    )



def join(x):
    return b''.join(x) if isinstance(x[0],bytes) else reduce(operator.add,x)

def resolve_jumps(op,chunks,end_targets=()):
    displacement = 0
    
    for i in range(len(chunks)-1,-1,-1):
        if isinstance(chunks[i],JumpTarget):
            chunks[i].displacement = displacement
            del chunks[i]
        else:
            if isinstance(chunks[i],JumpSource):
                chunks[i] = chunks[i].compile(displacement)
                if not chunks[i]:
                    del chunks[i]
                    continue
            elif isinstance(chunks[i],DelayedCompile):
                chunks[i].displacement = displacement
            
            displacement += len(chunks[i])

    # add padding for alignment
    if CALL_ALIGN_MASK:
        pad_size = aligned_size(displacement) - displacement
        chunks += [op.nop()] * pad_size
        for et in end_targets:
            et.displacement -= pad_size
    
    code = join([(c.compile() if isinstance(c,DelayedCompile) else c) for c in chunks])

    for et in end_targets:
        et.displacement -= displacement

    return code


def local_name_func(op,abi,tuning):
    # this function uses an ad hoc calling convention and is only callable from
    # the code generated by this module

    local_stack_size = aligned_size((MAX_ARGS + 1) * abi.ptr_size)
    stack_ptr_shift = local_stack_size - abi.ptr_size

    f = Frame(op,abi,tuning,local_stack_size)

    
    else_ = JumpTarget()
    endif = JumpTarget()
    ret = JumpTarget()
    
    return (f()
        .sub(stack_ptr_shift,abi.r_sp)
        .mov(f.LOCALS,f.r_ret)
        .if_eax_is_zero(f()
            .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_LOAD_MSG',f.r_pres[0])
            .add(stack_ptr_shift,abi.r_sp)
            .ret()
        )
        
        # if (%eax)->ob_type != PyDict_Type:
        .cmpl('PyDict_Type',f.Address(pyinternals.TYPE_OFFSET,f.r_ret))
        (JumpSource(f.op.je,f.abi,else_))
            .invoke('PyObject_GetItem',f.r_ret,f.r_pres[0])
            .test(f.r_ret,f.r_ret)
            (JumpSource(f.op.jnz,f.abi,ret))
            
            .invoke('PyErr_ExceptionMatches','PyExc_KeyError')
            .test(f.r_ret,f.r_ret)
            (JumpSource(f.op.jz,f.abi,ret))
            .call('PyErr_Clear')

            .goto(endif)
        (else_)
            .invoke('PyDict_GetItem',f.r_ret,f.r_pres[0])
            .test(f.r_ret,f.r_ret)
            (JumpSource(f.op.jnz,f.abi,ret))
        (endif)
    
        .invoke('PyDict_GetItem',f.GLOBALS,f.r_pres[0])
        .if_eax_is_zero(f()
            .invoke('PyDict_GetItem',f.BUILTINS,f.r_pres[0])
            .if_eax_is_zero(f()
                .invoke('format_exc_check_arg',
                    'NAME_ERROR_MSG',
                    'PyExc_NameError')
            )
        )
        
        (ret)
        .add(stack_ptr_shift,abi.r_sp)
        .ret()
    )



def compile_eval(code,op,abi,tuning,local_name,entry_points):
    """Generate a function equivalent to PyEval_EvalFrame called with f.code"""

    # the stack will have following items:
    #     - return address
    #     - old value of %ebp
    #     - old value of %ebx
    #     - old value of %esi
    #     - Frame object
    #     - GLOBALS
    #     - BUILTINS
    #     - LOCALS
    #     - FAST_LOCALS
    #
    # the first 2 will already be on the stack by the time %esp is adjusted

    stack_first = 7

    if pyinternals.REF_DEBUG:
        # a place to store %eax,%ecx and %edx when increasing reference counts
        # (which calls a function when ref_debug is True)
        stack_first += DEBUG_TEMPS

    local_stack_size = aligned_size(
        (code.co_stacksize + 
         max(MAX_ARGS-len(abi.r_arg),0) + 
         PRE_STACK + 
         stack_first + 
         STACK_EXTRA) * abi.ptr_size + abi.shadow)

    stack_ptr_shift = local_stack_size - (PRE_STACK+SAVED_REGS) * abi.ptr_size

    f = Frame(op,abi,tuning,local_stack_size,code,local_name,entry_points)

    opcodes = (f()
        .push(abi.r_bp)
        .mov(abi.r_sp,abi.r_bp)
        .push(f.r_pres[0])
        .push(f.r_pres[1])
        .sub(stack_ptr_shift,abi.r_sp)
    )
    f.stack.offset = PRE_STACK+SAVED_REGS

    argreg = f.stack.arg_reg(n=0)
    (opcodes
        .mov(f.stack.func_arg(0),f.r_pres[0])
        .lea(f.stack[-1],argreg)
        .mov(0,f.stack[-1])
        .invoke('_EnterRecursiveCall',argreg)
        .check_err(True)

        # as far as I can tell, after expanding the macros and removing the
        # "dynamic annotations" (see dynamic_annotations.h in the CPython
        # headers), this is all that PyThreadState_GET boils down to:
        .mov(f.Address(pyinternals.raw_addresses['_PyThreadState_Current']),f.r_scratch[0])

        .mov(f.Address(pyinternals.FRAME_GLOBALS_OFFSET,f.r_pres[0]),f.r_scratch[1])
        .mov(f.Address(pyinternals.FRAME_BUILTINS_OFFSET,f.r_pres[0]),f.r_ret)

        .push_stack(f.r_pres[0])
        .push_stack(f.r_scratch[1])
        .push_stack(f.r_ret)

        .mov(f.Address(pyinternals.FRAME_LOCALS_OFFSET,f.r_pres[0]),f.r_scratch[1])
        .lea(f.Address(pyinternals.FRAME_LOCALSPLUS_OFFSET,f.r_pres[0]),f.r_ret)

        .push_stack(f.r_scratch[1])
        .push_stack(f.r_ret)

        .mov(f.r_pres[0],f.Address(pyinternals.THREADSTATE_FRAME_OFFSET,f.r_scratch[0]))
    )


    if pyinternals.REF_DEBUG:
        # a place to store %eax,%ecx and %edx when increasing reference counts
        # (which calls a function when ref_debug is True
        f.stack.offset += DEBUG_TEMPS
    
    stack_prolog = f.stack.offset
    
    i = 0
    extended_arg = 0
    f.byte_offset = 0
    while i < len(f.code.co_code):
        bop = f.code.co_code[i]
        i += 1
        
        if bop >= dis.HAVE_ARGUMENT:
            boparg = f.code.co_code[i] + (f.code.co_code[i+1] << 8) + extended_arg
            i += 2
            
            if bop == dis.EXTENDED_ARG:
                extended_arg = bop << 16
            else:
                extended_arg = 0
                
                f.next_byte_offset = i
                opcodes += get_handler(bop)(f,boparg)
                f.byte_offset = i
        else:
            f.next_byte_offset = i
            opcodes += get_handler(bop)(f)
            f.byte_offset = i
    
    
    if f.stack.offset != stack_prolog:
        raise NCSystemError(
            'stack.offset should be {0}, but is {1}'
            .format(stack_prolog,f.stack.offset))

    if f.blockends:
        raise NCSystemError('there is an unclosed block statement')
    
    dr = join(f()
        .mov(f.Address(base=f.r_pres[0]),f.r_scratch[1])
        .decref(f.r_scratch[1])
        .add(f.ptr_size,f.r_pres[0])
        .code)
    
    cmpjl = f.op.cmp(abi.r_bp,f.r_pres[0])
    jlen = -(len(dr) + len(cmpjl) + f.JCC_MIN_LEN)
    assert jlen <= 127
    cmpjl += f.op.jb(f.Displacement(jlen))

    # have to compensate for subtracting from the base pointer below
    f.FRAME.offset += f.ptr_size * stack_prolog
    
    # call Py_DECREF on anything left on the stack and return %eax
    (opcodes
        (f._end)
        .mov(f.r_ret,f.Address(f.ptr_size*-stack_prolog,abi.r_bp))
        .sub(f.ptr_size*stack_prolog,abi.r_bp)
        .jmp(f.Displacement(len(dr)))
        (dr)
        (cmpjl)
        .call('_LeaveRecursiveCall')
        .mov(f.Address(base=abi.r_bp),f.r_ret)
        .mov(f.Address(pyinternals.raw_addresses['_PyThreadState_Current']),f.r_scratch[0])
        .mov(f.FRAME,f.r_scratch[1])
        .add(stack_ptr_shift,abi.r_sp)
        .pop(f.r_pres[1])
        .pop(f.r_pres[0])
        .mov(f.Address(pyinternals.FRAME_BACK_OFFSET,f.r_scratch[1]),f.r_scratch[1])
        .pop(abi.r_bp)
        .mov(f.r_scratch[1],f.Address(pyinternals.THREADSTATE_FRAME_OFFSET,f.r_scratch[0]))
        .ret()
    )

    return opcodes


def compile_raw(_code,abi,binary = True,tuning=Tuning()):
    assert len(abi.r_scratch) >= 2 and len(abi.r_pres) >= 2

    if isinstance(_code,types.CodeType):
        _code = (_code,)

    local_name = JumpTarget()
    entry_points = collections.OrderedDict()
    op = abi.ops if binary else abi.ops.Assembly()

    ceval = partial(compile_eval,
        op=op,
        abi=abi,
        tuning=tuning,
        local_name=local_name,
        entry_points=entry_points)

    # if this ever gets ported to C or C++, this will be a prime candidate for
    # parallelization
    def compile_code_constants(code):
        for c in code:
            if isinstance(c,types.CodeType) and id(c) not in entry_points:
                # reserve the entry
                entry_points[id(c)] = None

                compile_code_constants(c.co_consts)
                entry_points[id(c)] = (
                    pyinternals.create_compiled_entry_point(c),
                    ceval(c))
    
    compile_code_constants(_code)

    functions = []
    end_targets = []
    
    if local_name.used:
        local_name.displacement = 0
        end_targets.append(local_name)
        functions.append(resolve_jumps(op,local_name_func(op,abi,tuning).code))

    entry_points = list(entry_points.values())
    for ep,func in reversed(entry_points):
        functions.insert(0,resolve_jumps(op,func.code,end_targets))

    offset = 0
    for epf,func in zip(entry_points,functions):
        pyinternals.cep_set_offset(epf[0],offset)
        offset += len(func)

    if not binary:
        functions = join(functions)
    
    return functions,[ep for ep,func in entry_points]



