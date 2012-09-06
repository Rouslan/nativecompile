
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
MAX_ARGS = 6
PRE_STACK = 2
DEBUG_TEMPS = 3
STACK_EXTRA = 1 # extra room on the stack for temporary values
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

# copied from Python/cevel.c
WHY_NOT =       0x0001
WHY_EXCEPTION = 0x0002
WHY_RERAISE =   0x0004
WHY_RETURN =    0x0008
WHY_BREAK =     0x0010
WHY_CONTINUE =  0x0020
WHY_YIELD =     0x0040
WHY_SILENCED =  0x0080 


BLOCK_LOOP = 1
BLOCK_EXCEPT = 2
BLOCK_FINALLY = 3

EXCEPT_VALUES = 3


class NCCompileError(SystemError):
    """A problem exists with the bytecode.

    Note: This compiler has certain constraints that the Python interpreter
    doesn't. Bytecode produced by CPython should never violate these contraints
    but arbitrary bytecode might.

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
        self._offset = 0

        self.args = 0

         # if True, the TOS item is in %eax and has not actually been pushed
         # onto the stack
        self.tos_in_eax = False

        self.resets = []

        # this is a stack so nested blocks can have finer restrictions
        self.protected_items = [0]

    def copy(self):
        r = StackManager(self.op,self.abi,self.local_mem_size)
        r._offset = self._offset
        r.tos_in_eax = self.tos_in_eax
        r.resets = self.resets[:]
        r.protected_items = self.protected_items[:]
        return r

    def check_stack_space(self):
        if (self._offset + max(self.args-len(self.abi.r_arg),0)) * self.abi.ptr_size > self.local_mem_size:
            raise NCCompileError("Not enough stack space was reserved. This code object being compiled has an incorrect value for co_stacksize.")

    def get_offset(self):
        return self._offset

    def set_offset(self,val):
        if val is None:
            self._offset = None
        else:
            if val < self.protected_items[-1]:
                raise NCCompileError("The code being compiled tries to pop more items from the stack than allowed at the given context")

            self._offset = val
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
                if self.abi.ptr_size == 8:
                    r = [self.op.movl(x & 0xffffffff,dest),self.op.movl(x >> 32,dest+4)]
                else:
                    r = [self.op.movl(x,dest)]
            else:
                r = [self.op.mov(x,dest)]

        if n == self.args: self.args += 1
        self.check_stack_space()
        return r
    
    def push_tos(self,set_again = False):
        """r_ret is needed right now so if the TOS item hasn't been pushed onto 
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
    def __init__(self,opset,abi,target,jump_instead=False):
        self.op = opset.jmp if jump_instead else opset.call
        self.nop = opset.nop
        self.length = abi.ops.JMP_DISP_MAX_LEN if jump_instead else abi.ops.CALL_DISP_LEN
        self.abi = abi
        self.target = target
        target.used = True
        self.displacement = None

    def compile(self):
        r = self.op(self.abi.ops.Displacement(self.displacement - self.target.displacement))
        if len(r) < self.length:
            r += self.nop() * (self.length - len(r))
        return r

    def __len__(self):
        return self.length


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

        if f.blockends and f.blockends[0].offset <= f.byte_offset:
            b = f.blockends.pop(0)
            assert b.offset == f.byte_offset
            r.push_tos()
            r += b.prepare(f)
            r(b.target)

        f.stack.current_pos(f.byte_offset)

        if PRINT_STACK_OFFSET:
            print('stack items: {}  opcode: {}'.format(
                f.stack.offset + f.stack.tos_in_eax,
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

def hasoffset(func):
    return handler(
        (lambda f,arg: func(f,f.next_byte_offset+arg)),
        func.__name__)


def get_handler(op):
    h = handlers[op]
    if h is None:
        raise Exception('op code {} is not implemented'.format(dis.opname[op]))
    return h


class BlockEnd:
    def __init__(self,type,offset,stack):
        self.type = type
        self.offset = offset
        self.stack = stack
        self.target = JumpTarget()

    def prepare(self,f):
        return []

class LoopBlockEnd(BlockEnd):
    def __init__(self,offset):
        super().__init__(BLOCK_LOOP,offset,None)

        # If a "continue" statement is encountered inside this block, store its
        # destination here and set "stack". This is needed by END_FINALLY.
        self.continue_offset = None

class ExceptBlockEnd(BlockEnd):
    def __init__(self,type,offset,stack,extra_code=(lambda f: [])):
        # "except" and "finally" blocks will have EXCEPT_VALUES values on the
        # stack for them to inspect
        super().__init__(type,offset,stack+EXCEPT_VALUES)

        self.extra_code = extra_code

    def prepare(self,f):
        # run it first in case the stack is altered
        r = self.extra_code(f)

        if f.stack.offset is None:
            f.stack.offset = self.stack
        elif f.stack.offset != self.stack:
            raise NCCompileError('Incorrect stack size at except/finally block')

        f.exc_handler_blocks.append((self.type,self.stack))
        f.stack.protected_items.append(self.stack)

        return r
    

def raw_addr_if_str(x):
    return pyinternals.raw_addresses[x] if isinstance(x,str) else x


class Frame:
    def __init__(self,op,abi,tuning,local_mem_size,code=None,local_name=None,prepare_exc_handler=None,entry_points=None):
        self.code = code
        self.op = op
        self.abi = abi
        self.tuning = tuning
        self.stack = StackManager(op,abi,local_mem_size)
        self.end = JumpTarget()
        self.local_name = local_name
        self.prepare_exc_handler = prepare_exc_handler
        self.blockends = []
        self.exc_handler_blocks = []
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

    def type_of(self,r):
        return self.Address(pyinternals.TYPE_OFFSET,r)

    def type_flags_of(self,r):
        return self.Address(pyinternals.TYPE_FLAGS_OFFSET,r)

    def tuple_item(self,r,n):
        return self.Address(pyinternals.TUPLE_ITEM_OFFSET + self.ptr_size*n,r)

    def current_except_or_finally(self):
        for b in reversed(self.blockends):
            if b.type == BLOCK_EXCEPT or b.type == BLOCK_FINALLY: return b
        return None

    def current_finally(self):
        for b in reversed(self.blockends):
            if b.type == BLOCK_FINALLY: return b
        return None

    def current_loop(self):
        for b in reversed(self.blockends):
            if b.type == BLOCK_LOOP: return b
        return None

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

    def fits_imm32(self,x):
        """Return True if x fits in a 32-bit immediate value without
        sign-extend.

        32-bit immediate values are interpreted as signed. In 64-bit mode, these
        values get sign-extended to 64 bits and thus have their binary
        representation altered, which can make a difference when comparing
        addresses.

        """
        if self.ptr_size == 8:
            return -0x80000000 <= x <= 0x7fffffff

        return True
    
    def check_err(self,inverted=False):
        if inverted:
            return self.if_eax_is_not_zero(
                [self.op.xor(self.r_ret,self.r_ret)] + 
                self.goto_end(True))

        return self.if_eax_is_zero(self.goto_end(True))

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

    def goto_end(self,exception=False):
        """Go to the inner most finally block, or except block if exception is
        True, or the end of the function if there isn't an appropriate block"""
        
        block = self.current_except_or_finally() if exception else self.current_finally()
        if block:
            extra = self.stack.offset - block.stack

            # if this is an exception, the exception values haven't been added
            # to the stack yet
            assert extra >= -EXCEPT_VALUES if exception else extra == 0

            r = []
            if exception:
                r = [self.op.lea(self.stack[0],self.r_pres[0]),
                          self.op.mov(-(extra+EXCEPT_VALUES),self.r_scratch[0]),
                          InnerCall(self.op,self.abi,self.prepare_exc_handler)]

            r.append(JumpSource(self.op.jmp,self.abi,block.target))
            return r

        return [
            self.op.mov(self.stack.protected_items[-1] - self.stack.offset,self.r_pres[0]),
            JumpSource(self.op.jmp,self.abi,self.end)]

    def incref(self,reg=None,amount=1):
        if reg is None: reg = self.r_ret
        if pyinternals.REF_DEBUG:
            # the registers that would otherwise be undisturbed, must be preserved
            return ([
                self.op.mov(self.r_ret,self.TEMP_EAX),
                self.op.mov(self.r_scratch[0],self.TEMP_ECX),
                self.op.mov(self.r_scratch[1],self.TEMP_EDX)
            ] + (self.invoke('Py_IncRef',reg) * amount) + [
                self.op.mov(self.TEMP_EDX,self.r_scratch[1]),
                self.op.mov(self.TEMP_ECX,self.r_scratch[0]),
                self.op.mov(self.TEMP_EAX,self.r_ret)])

        return [self.add(amount,self.Address(pyinternals.REFCNT_OFFSET,reg))]

    def decref(self,reg=None,preserve_reg=None,amount=1):
        """Generate instructions equivalent to Py_DECREF

        Note: the register spcified by r_pres[1] is used and not preserved by
        these instructions. Additionally, if preserve_reg is not None,
        self.stack[-1] will be used.

        """
        assert STACK_EXTRA >= 1 or not preserve_reg

        if reg is None: reg = self.r_ret

        if pyinternals.REF_DEBUG:
            inv = self.invoke('Py_DecRef',reg) * amount
            return [self.op.mov(preserve_reg,self.TEMP_EAX)] + inv + [self.op.mov(self.TEMP_EAX,preserve_reg)] if preserve_reg else inv

        assert reg.reg != self.r_pres[1].reg
        
        mid = []
        
        if preserve_reg:
            mid.append(self.op.mov(preserve_reg,self.stack[-1]))

        mid += [
            self.op.mov(self.Address(pyinternals.TYPE_OFFSET,reg),self.r_pres[1]),
        ]

        mid += self.invoke(
            self.Address(pyinternals.TYPE_DEALLOC_OFFSET,self.r_pres[1]),
            reg)

        if pyinternals.COUNT_ALLOCS:
            mid += self.invoke('inc_count',self.r_pres[1])
        
        if preserve_reg:
            mid.append(self.op.mov(self.stack[-1],preserve_reg))
        
        mid = join(mid)
        
        return [
            self.sub(amount,self.Address(pyinternals.REFCNT_OFFSET,reg)),
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
            raise NCCompileError('unexpected jump target')

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

    def clean_stack(self,always_clean=False,addr=None,index=None):
        """Free items above the current stack top

        the register specified by index (default: r_pres[0]) will be a negative
        number whose absolute size indicates how many values need to be freed.
        The values needing freeing will all be above the top of the stack.

        If always_clean is True, this will always free at least one item (and
        will use slightly fewer instructions), so if there are no values, this
        should be jumped over.

        """

        addr = addr or self.stack[0]
        addr.index = index or self.r_pres[0]
        addr.scale = self.ptr_size

        assert (addr.base != self.r_ret and
                addr.base not in self.r_scratch and
                index != self.r_ret and
                index != self.r_pres[1] and
                addr.base != index)

        preserve_reg = None
        if addr.index in self.r_scratch: preserve_reg = addr.index

        dr = join(
            [self.op.mov(addr,self.r_ret)] +
            self.decref(preserve_reg=preserve_reg) +
            [self.add(1,addr.index)])

        size = len(dr) + self.JCC_MIN_LEN
        cmpj = self.op.jnz(self.Displacement(-size))
        assert len(cmpj) == self.JCC_MIN_LEN

        r = [dr,cmpj]
        if not always_clean:
            r = [
                self.op.test(addr.index,addr.index),
                self.op.jz(self.Displacement(size))
            ] + r

        return r

    def add(self,a,b):
        if a == 1 and not self.tuning.prefer_addsub_over_incdec:
            if isinstance(b,self.Address):
                return self.op.incl(b)
            return self.op.inc(b)

        if isinstance(a,int) and isinstance(b,self.Address):
            return self.op.addl(a,b)
        return self.op.add(a,b)

    def sub(self,a,b):
        if a == 1 and not self.tuning.prefer_addsub_over_incdec:
            if isinstance(b,self.Address):
                return self.op.decl(b)
            return self.op.dec(b)

        if isinstance(a,int) and isinstance(b,self.Address):
            return self.op.subl(a,b)
        return self.op.sub(a,b)

    # Note: this function assumes little-endian format
    def mov(self,a,b):
        if a != b:
            if a == 0 and isinstance(b,self.Register):
                return [self.op.xor(b,b)]
            if isinstance(a,int) and isinstance(b,self.Address):
                if self.ptr_size == 8:
                    return [self.op.movl(a & 0xffffffff,b),self.op.movl(a >> 32,b+4)]
                return [self.op.movl(a,b)]
            return [self.op.mov(a,b)]

        return []

    def cmp(self,a,b):
        assert (not isinstance(a,int)) or self.fits_imm32(a)

        if isinstance(a,int) and isinstance(b,self.Address):
            return self.op.cmpl(a,b)
        return self.op.cmp(a,b)

    def get_threadstate(self,reg):
        # as far as I can tell, after expanding the macros and removing the
        # "dynamic annotations" (see dynamic_annotations.h in the CPython
        # headers), this is all that PyThreadState_GET boils down to:
        addr = pyinternals.raw_addresses['_PyThreadState_Current']
        if self.fits_imm32(addr):
            return [self.op.mov(self.Address(addr),reg)]
        return [self.op.mov(addr,reg),self.op.mov(self.Address(base=reg),reg)]

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
    """Generate a sequence of machine code instructions concisely using method
    chaining"""
    def __init__(self,frame,code=None):
        self.f = frame
        self.code = code or []
        self.stack = frame.stack

    def branch(self):
        r = Stitch(self.f,self.code)
        r.stack = self.stack.copy()
        return r

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

    def push_arg(self,x,*args,**kwds):
        self.code += self.f.stack.push_arg(raw_addr_if_str(x),*args,**kwds)
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
        self.code.append(self.f.add(a,b))
        return self

    @strs_to_addrs
    def sub(self,a,b):
        self.code.append(self.f.sub(a,b))
        return self

    @strs_to_addrs
    def mov(self,a,b):
        self.code += self.f.mov(a,b)
        return self

    @strs_to_addrs
    def cmp(self,a,b):
        self.code.append(self.f.cmp(a,b))
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
    'goto_end',
    'clean_stack',
    'get_threadstate']:
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
    dict_addr = pyinternals.raw_addresses['PyDict_Type']
    fits = f.fits_imm32(dict_addr)

    tos = f.stack.tos()
    r = (f()
        .push_tos()
        .push_arg(tos,n=2)
        .push_arg(address_of(name),n=1)
        .mov(f.LOCALS,f.r_ret)
    )
    if not fits:
        r.mov(dict_addr,f.r_scratch[1])

    return (r
        .if_eax_is_zero(f()
            .clear_args()
            .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_STORE_MSG')
            .goto_end(True)
        )
        .mov('PyObject_SetItem',f.r_scratch[0])
        .push_arg(f.r_ret,n=0)
        .cmp(dict_addr if fits else f.r_scratch[1],f.type_of(f.r_ret))
        .if_cond[f.test_E](f()
            .mov('PyDict_SetItem',f.r_scratch[0])
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
            .goto_end(True)
        )
        .invoke('PyObject_DelItem',f.r_ret,address_of(name))
        .if_eax_is_zero(f()
            .invoke('format_exc_check_arg',
                'PyExc_NameError',
                'NAME_ERROR_MSG',
                address_of(name))
            .goto_end(True)
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
                .goto_end(True)
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
    r = f()
    block = f.current_finally()
    if block:
        (r
            .push_tos()
            .mov('Py_None',f.r_scratch[0])
            .push_stack(f.r_scratch[0])
            .push_stack(f.r_scratch[0])
            .incref(f.r_scratch[0],2)
        )
    elif not f.stack.use_tos():
        r.pop_stack(f.r_ret)

    r.goto_end()
    f.stack.offset = None
    return r

@hasoffset
def _op_SETUP_LOOP(f,to):
    f.blockends.append(LoopBlockEnd(to))
    return []

@handler
def _op_POP_BLOCK(f):
    return []

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

@hasoffset
def _op_FOR_ITER(f,to):
    argreg = f.stack.arg_reg(n=0)
    return (f()
        .push_tos(True)
        (f.rtarget())
        .mov(f.stack[0],argreg)
        .mov(f.type_of(argreg),f.r_ret)
        .mov(f.Address(pyinternals.TYPE_ITERNEXT_OFFSET,f.r_ret),f.r_ret)
        .invoke(f.r_ret,argreg)
        .if_eax_is_zero(f()
            .call('PyErr_Occurred')
            .if_eax_is_not_zero(f()
                .invoke('PyErr_ExceptionMatches','PyExc_StopIteration')
                .check_err()
                .call('PyErr_Clear')
            )
            .goto(f.forward_target(to,True))
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
        .decref(f.r_scratch[1],f.r_ret)
        .test(f.r_ret,f.r_ret)
        (JumpSource(jop1,f.abi,dont_jump))
        (f.jump_to(jop2,f.JCC_MAX_LEN,to))
        .mov(0,f.r_ret)
        .goto_end(True)
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
                    f.add(1,f.r_ret))
            else:
                r.mov(0,f.r_pres[0])

                top.index = f.r_pres[0]

                lbody = (
                    f.op.mov(top,f.r_scratch[1]) +
                    f.op.mov(f.r_scratch[1],f.Address(item_offset-f.ptr_size,f.r_ret,f.r_scratch[0],f.ptr_size)) +
                    f.add(1,f.r_pres[0]))

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
            .goto_end(True)
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

    tt_addr = pyinternals.raw_addresses['PyTuple_Type']
    tt_fits = f.fits_imm32(tt_addr)
    if not tt_fits:
        r.mov(tt_addr,f.r_scratch[0])

    lt_addr = pyinternals.raw_addresses['PyList_Type']
    lt_fits = f.fits_imm32(lt_addr)
    if not lt_fits:
        r.mov(lt_addr,f.r_pres[1])

    (r
        .mov(f.type_of(f.r_pres[0]),f.r_scratch[1])
        .cmp(tt_addr if tt_fits else f.r_scratch[0],f.r_scratch[1])
        (JumpSource(f.op.jne,f.abi,check_list))
            .cmp(arg,f.Address(pyinternals.VAR_SIZE_OFFSET,f.r_pres[0]))
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
                .mov(f.tuple_item(f.r_pres[0],i),reg)
                .incref(reg)
                .mov(reg,f.stack[i-arg])).code

        r += interleave_ops([f.r_ret,f.r_scratch[0],f.r_scratch[1]],unpack_one)


    (r
            .goto(done)
        (check_list)
        .cmp(lt_addr if lt_fits else f.r_pres[1],f.r_scratch[1])
        (JumpSource(f.op.jne,f.abi,else_))
            .cmp(arg,f.Address(pyinternals.VAR_SIZE_OFFSET,f.r_pres[0]))
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
                .goto_end(True)
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
        .decref(f.r_pres[0],f.r_ret)
        .if_eax_is_zero(f()
            .goto_end(True)
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
        if not f.stack.use_tos(True):
            r.pop_stack(f.r_ret)

        return (r
            .mov(outcome_a,f.r_pres[0])
            .cmp(f.r_ret,f.stack[0])
            .if_cond[f.test_E](
                f.mov(outcome_b,f.r_pres[0])
            )
            .decref(f.r_ret)
            .pop_stack(f.r_ret)
            .decref(f.r_ret)
            .mov(f.r_pres[0],f.r_ret)
            .incref(f.r_pres[0])
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
                .goto_end(True)
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

@hasoffset
def _op_JUMP_FORWARD(f,to):
    r = (f()
        .push_tos()
        .goto(f.forward_target(to))
    )
    f.stack.unconditional_jump(to)
    return r

@handler
def _op_RAISE_VARARGS(f,arg):
    # TODO: when arg is 0 and we are in an except block that is followed by a
    # finally block, we can probably just jump into the finally block and avoid
    # calling _do_raise and prepare_exc_handler_func

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
        .goto_end(True)
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
            .goto_end(True)
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
            .goto_end(True)
        )
    )

@hasname
def _op_IMPORT_FROM(f,name):
    tos = f.stack.tos()
    return (f()
        .push_tos(True)
        .invoke('PyObject_GetAttr',tos,address_of(name))
        .if_eax_is_zero(f()
            .invoke('PyErr_ExceptionMatches','PyExc_AttributeError')
            .if_eax_is_not_zero(join(f()
               .invoke('PyErr_Format','CANNOT_IMPORT_MSG',address_of(name))
               .mov(0,f.r_ret)
            .code))
            .goto_end(True)
        )
    )

@handler
def _op_IMPORT_STAR(f):
    r = f()
    if not f.stack.use_tos():
        r.pop_stack(f.r_ret)

    return (r
        # import_from decrements the reference count for us
        .invoke('import_all_from',f.FRAME,f.r_ret)
        .check_err(True)
    )

# Currently, the machine code produced by this function fails on x86_64 when
# Python is built with debugging. GCC seems to use a different calling
# convention for variadic functions in this case.
@hasname
def _op_IMPORT_NAME(f,name):
    endif = JumpTarget()
    r_imp = f.r_pres[0]

    def prepare_args(n):
        return (f()
            .mov(f.LOCALS,f.r_ret)
            .push_arg(n)
            .push_arg(address_of(name))
            .push_arg(f.GLOBALS)
            .push_arg(f.r_ret)
            .test(f.r_ret,f.r_ret)
            .if_cond[f.test_Z](join(f()
                .push_arg('Py_None',n=f.stack.args-1).code
            ))
            .push_arg(f.stack[0]))

    r = (f()
        .push_tos()
        .invoke('PyDict_GetItemString',f.BUILTINS,'__import__')
        .if_eax_is_zero(f()
            .invoke('PyErr_SetString','PyExc_ImportError','IMPORT_NOT_FOUND_MSG')
            .goto_end(True)
        )
        .mov(f.r_ret,r_imp)
        .incref()
        .invoke('PyLong_AsLong',f.stack[1])
        .cmp(-1,f.r_ret)
        .if_cond[f.test_NE](f()
            .call('PyErr_Occurred')
            .if_eax_is_not_zero(
                prepare_args(5)
                .push_arg(f.stack[1])
                .call('PyTuple_Pack')
                .goto(endif)
            )
        ) +
        # else
            prepare_args(4)
            .call('PyTuple_Pack')
        (endif)
    )

    a_args = f.stack[-1]
    return (r
        .mov(f.r_ret,a_args)
        .pop_stack(f.r_ret)
        .decref()
        .pop_stack(f.r_ret)
        .decref()
        .mov(a_args,f.r_ret)
        .test(f.r_ret,f.r_ret)
        .if_cond[f.test_Z](f()
            .decref(r_imp)
            .goto_end(True)
        )

        # TODO: have this be able to call compiled code
        .invoke('PyObject_Call',r_imp,f.r_ret,0)

        .push_stack(f.r_ret)
        .decref(r_imp)
        .mov(a_args,f.r_ret)
        .decref()
        .mov(f.stack[0],f.r_ret)
        .if_eax_is_zero(f()
            .add_to_stack(-1)
            .goto_end(True)
            .add_to_stack(1)
        )
    )

#@handler
#def _op_SETUP_WITH(f,arg):
#    r = f()
#    if f.use_tos():
#        r.mov(f.r_ret,f.r_pres[0])
#    else:
#        r.pop_stack(f.r_pres[0])
#
#    return (r
#        .invoke('special_lookup',f.r_pres[0],'__exit__','exit_cache')
#        .if_eax_is_zero(f()
#            .decref(f.r_pres[0])
#            .mov(0,f.r_ret)
#            .goto_end(True)
#        )
#        .push_stack(f.r_ret)
#        .invoke('special_lookup',f.r_pres[0],'__enter__','enter_cache')
#        .decref(f.r_pres[0],f.r_ret)
#        .mov(f.r_ret,f.r_pres[0])
#        .check_err()
#        .invoke('PyObject_CallFunctionObjArgs',f.r_ret,0)
#        .decref(f.r_pres[0],f.r_ret)
#        .check_err()


@hasoffset
def _op_SETUP_EXCEPT(f,to):
    f.blockends.append(ExceptBlockEnd(BLOCK_EXCEPT,to,f.stack.offset))
    return []

@hasoffset
def _op_SETUP_FINALLY(f,to):
    # When a finally block is entered without a jump, only one value will be
    # pushed onto the stack for it to look at, so we pad the stack with two more
    def add_nones(f):
        return (f()
            .mov('Py_None',f.scratch[0])
            .push_stack(f.scratch[0])
            .push_stack(f.scratch[0])
            .incref(f.scratch[0],2)
        )

    f.blockends.append(ExceptBlockEnd(BLOCK_FINALLY,to,f.stack.offset,add_nones))
    return []

def pop_handler_block(f,type):
    def block_mismatch():
        return NCCompileError(
            'There is a POP_EXCEPT instruction not correctly matched with a SETUP_EXCEPT instruction'
                if type == BLOCK_EXCEPT else
            'There is an END_FINALLY instruction not correctly matched with a SETUP_FINALLY instruction')

    if not f.exc_handler_blocks:
        raise block_mismatch()
    btype,stack = f.handler_blocks.pop()
    if btype != type:
        raise block_mismatch()
    if f.stack.offset + f.stack.tos_in_eax != stack:
        raise NCCompileError(
            'Incorrect stack size at POP_EXCEPT instruction'
                if type == BLOCK_EXCEPT else
            'Incorrect stack size at END_FINALLY instruction')

    # this shouldn't be possible
    assert not f.stack.tos_in_eax

    f.stack.protected_items.pop()

@handler
def _op_POP_EXCEPT(f):
    pop_handler_block(f,BLOCK_EXCEPT)

    # if this value changes, the code here will need updating
    assert EXCEPT_VALUES == 3

    return (f()
        .get_threadstate(f.r_pres[1])

        # our values are stored in reverse order compared to how the Python
        # interpreter stores them
        .mov(f.stack[2],f.r_pres[0])
        .mov(f.stack[1],f.r_ret)
        .mov(f.stack[0],f.r_scratch[0])
        .mov(f.r_pres[0],f.Address(pyinternals.THREADSTATE_TYPE_OFFSET,f.r_pres[1]))
        .mov(f.r_ret,f.Address(pyinternals.THREADSTATE_VALUE_OFFSET,f.r_pres[1]))
        .mov(f.r_scratch[0],f.Address(pyinternals.THREADSTATE_TRACEBACK_OFFSET,f.r_pres[1]))
        .decref(f.r_scratch[0],preserve_reg=f.r_ret)
        .decref(f.r_ret)
        .decref(f.r_pres[0])
        .add_to_stack(-3)
    )

@handler
def _op_END_FINALLY(f):
    # if this value changes, the code here will need updating
    assert EXCEPT_VALUES == 3

    # find the next "finally" block and the next loop block if it comes before
    # the "finally" block
    nextl = None
    nextf = None
    for b in reversed(f.blockends):
        if b.type == BLOCK_LOOP:
            if not nextl: nextl = b
        elif b.type == BLOCK_FINALLY:
            nextf = b
            break

    pop_handler_block(f,BLOCK_FINALLY)

    not_long = JumpTarget()
    not_except = JumpTarget()
    proceed = JumpTarget()
    err = JumpTarget()

    r_tmp1 = f.stack.arg_reg(f.r_scratch[0],0)
    r_tmp2 = f.stack.arg_reg(f.r_scratch[1],1)
    r_tmp3 = f.stack.arg_reg(f.r_ret,2)

    n_addr = pyinternals.raw_addresses['Py_None']
    n_fits = f.fits_imm32(n_addr)

    # TODO: check if there even was a return keyword in the try block and omit
    # the WHY_RETURN code if there wasn't

    # our values are stored in reverse order compared to how the Python
    # interpreter stores them
    r = f().mov(f.stack[EXCEPT_VALUES-1],r_tmp1)
    
    if not n_fits:
        r.mov(n_addr,f.r_pres[1])

    (r
        .test(TPFLAGS_LONG_SUBCLASS,f.type_flags_of(r_tmp1))
        .jz(not_long)
    )

    if nextf and not nextl:
        if f.stack.offset != nextf.stack:
            raise NCCompileError('Incorrect stack size')

        # The value of r_tmp1 will be WHY_RETURN or WHY_CONTINUE (WHY_SILENCED
        # is not yet handled). Either way, just go to the next finally block.
        r.goto(nextf.target)
    else:
        f_stack_diff = f.stack.offset - nextf.stack

        # since there is at least one loop between here and the next finally
        # block, there should be some values to pop
        assert f_stack_diff

        may_continue = nextl.continue_offset is not None
        not_ret = JumpTarget()
        (r
            .invoke('PyLong_AsLong',r_tmp1)
            .cmp(WHY_RETURN,f.r_ret)
            .jne(not_ret if may_continue else err) # "WHY_SILENCED" not yet handled
        )

        if nextf:
            # free the extra values and shift our three values down
            for n in range(f_stack_diff):
                r.mov(f.stack[n+EXCEPT_VALUES],f.r_ret).decref()

            (r
                .mov(f.stack[2],f.r_ret)
                .mov(f.stack[1],f.r_scratch[0])
                .mov(f.stack[0],f.r_scratch[1])
                .mov(f.r_ret,f.stack[2+f_stack_diff])
                .mov(f.r_scratch[0],f.stack[1+f_stack_diff])
                .mov(f.r_scratch[1],f.stack[f_stack_diff])
                .goto(nextf.target)
            )
        else:
            # the function epilog will clean up the stack for us
            r.goto_end()

        if may_continue:
            assert EXCEPT_VALUES >= 2
            (r
                (not_ret)
                .cmp(WHY_CONTINUE,f.r_ret)
                .jne(err) # "WHY_SILENCED" not yet handled
                .branch()
                .pop_stack(f.r_ret)
                .decref(amount=EXCEPT_VALUES-1) # the top two values are padding and are both None
                .add_to_stack(2-EXCEPT_VALUES)
                .pop_stack(f.r_ret)
                .decref()
                (JumpRSource(
                    f.op,
                    f.abi,
                    f.abi.JMP_DISP_MAX_LEN,
                    f.reverse_target(nextl.continue_offset)))
            )

    (r
        (not_long)

        .mov(f.type_of(r_tmp1),r_tmp2)
        .test(TPFLAGS_TYPE_SUBCLASS,f.type_flags_of(r_tmp2))
        .jz(not_except)
        .test(TPFLAGS_BASE_EXC_SUBCLASS,f.type_flags_of(r_tmp1))
        .jz(not_except)
            .branch()
            .pop_stack(r_tmp3)
            .pop_stack(r_tmp2)

            # PyErr_Restore steals references
            .invoke('PyErr_Restore',r_tmp1,r_tmp2,r_tmp3)
            .goto_end(True)
        (not_except)

        .cmp(n_addr if n_fits else f.r_pres[1],r_tmp1)
        .if_cond[f.test_NE](f()
            (err)
            .invoke('PyErr_SetString','PyExc_SystemError','BAD_EXCEPTION_MSG')
            .goto_end(True)
        )
        .decref(r_tmp1,amount=EXCEPT_VALUES) # all values are None
        .add_to_stack(-EXCEPT_VALUES)
    )

    return r

@handler
def _op_NOP(f):
    return []

@handler
def _op_DUP_TOP(f):
    tos = f.stack.tos()
    return (f()
        .push_tos(True)
        .mov(tos,f.r_ret)
        .incref()
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

    # TODO: have just one copy shared by all compiled modules

    local_stack_size = aligned_size((MAX_ARGS + 1) * abi.ptr_size)
    stack_ptr_shift = local_stack_size - abi.ptr_size

    f = Frame(op,abi,tuning,local_stack_size)

    
    else_ = JumpTarget()
    endif = JumpTarget()
    ret = JumpTarget()

    d_addr = pyinternals.raw_addresses['PyDict_Type']
    fits = f.fits_imm32(d_addr)

    r = f()
    if not fits:
        r.mov(d_addr,f.r_scratch[0])
    
    return (f()
        .sub(stack_ptr_shift,abi.r_sp)
        .mov(f.LOCALS,f.r_ret)
        .if_eax_is_zero(f()
            .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_LOAD_MSG',f.r_pres[0])
            .add(stack_ptr_shift,abi.r_sp)
            .ret()
        )
        
        .cmp(d_addr if fits else f.r_scratch[0],f.type_of(f.r_ret))
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

def prepare_exc_handler_func(op,abi,tuning):
    # this function uses an ad hoc calling convention and is only callable from
    # the code generated by this module

    # TODO: have just one copy shared by all compiled modules

    # r_scratch[0] is expected to have the number of items that need to be
    # popped from the stack

    # r_pres[0] is expected to be what the address of the top of the stack will
    # be once the items have been popped


    # if this value changes, the code here will need updating
    assert EXCEPT_VALUES == 3

    local_stack_size = aligned_size((MAX_ARGS + 1) * abi.ptr_size)
    stack_ptr_shift = local_stack_size - abi.ptr_size

    f = Frame(op,abi,tuning,local_stack_size)

    exc = f.Address(-abi.ptr_size,f.r_pres[0])
    val = f.Address(-abi.ptr_size*2,f.r_pres[0])
    tb = f.Address(-abi.ptr_size*3,f.r_pres[0])

    r = (f()
        .sub(stack_ptr_shift,abi.r_sp)
        .clean_stack(addr=f.r_pres[0],index=f.r_scratch[0])

        # Normally, a finally block will have 1 to 3 items on the stack for it
        # to pop, but our stack management uses absolute offsets instead of push
        # and pop instructions so we don't have this flexibility. Thus we place
        # the items in reverse order and pad the stack with Nones if needed (in
        # this case, it's not).
        .get_threadstate(f.r_pres[1])
        .mov(f.Address(pyinternals.THREADSTATE_TRACEBACK_OFFSET,f.r_pres[1]),f.r_scratch[0])
        .mov(f.Address(pyinternals.THREADSTATE_VALUE_OFFSET,f.r_pres[1]),f.r_scratch[1])
        .mov(f.Address(pyinternals.THREADSTATE_TYPE_OFFSET,f.r_pres[1]),f.r_ret)
        .mov(f.r_scratch[0],tb)
        .mov(f.r_scratch[1],val)
        .if_eax_is_zero(f()
            .mov('Py_None',f.r_ret)
            .incref()
        )
        .mov(f.r_ret,exc)
    )

    load_args = []
    for n,item in enumerate((exc,val,tb)):
        dest = f.stack.func_arg(n)
        reg = (f.r_scratch[0],f.r_scratch[1],f.r_ret)[n]
        if isinstance(dest,f.Address):
            load_args.insert(n,f.op.lea(item,reg))
            load_args.append(f.op.mov(reg,dest))
        else:
            load_args.append(f.op.lea(item,dest))
    load_args = join(load_args)

    (r
        (load_args)
        .call('PyErr_Fetch')
        (load_args)
        .call('PyErr_NormalizeException')
        .invoke('PyException_SetTraceback',val,tb)
        .mov(exc,f.r_ret)
        .mov(val,f.r_scratch[0])
        .mov(tb,f.r_scratch[1])
        .incref(f.r_ret)
        .incref(f.r_scratch[0])
        .test(f.r_scratch[1],f.r_scratch[1])
        .if_cond[f.test_Z](
            f.op.mov('Py_None',f.r_scratch[1])
        )
        .mov(f.r_ret,f.Address(pyinternals.THREADSTATE_TYPE_OFFSET,f.r_pres[1]))
        .mov(f.r_scratch[0],f.Address(pyinternals.THREADSTATE_TYPE_OFFSET,f.r_pres[1]))
        .incref(f.r_scratch[1])
        .mov(f.r_scratch[1],f.Address(pyinternals.THREADSTATE_TYPE_OFFSET,f.r_pres[1]))
        .add(stack_ptr_shift,abi.r_sp)
        .ret()
    )

    return r


def compile_eval(code,op,abi,tuning,local_name,prepare_exc_handler,entry_points):
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

    f = Frame(op,abi,tuning,local_stack_size,code,local_name,prepare_exc_handler,entry_points)

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

        .get_threadstate(f.r_scratch[0])

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
    f.stack.protected_items.append(f.stack.offset)
    
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
    
    
    if f.stack.offset is not None and f.stack.offset != stack_prolog:
        raise NCCompileError(
            'stack.offset should be {0}, but is {1}'
            .format(stack_prolog,f.stack.offset))
    f.stack.offset = stack_prolog
    f.stack.protected_items.pop()
    assert len(f.stack.protected_items) == 1

    if f.blockends:
        raise NCCompileError('there is an unclosed block statement')
    

    # call Py_DECREF on anything left on the stack and return %eax
    (opcodes
        (f.end)
        .mov(f.r_ret,f.stack[0])
        .clean_stack()
        .call('_LeaveRecursiveCall')
        .mov(f.stack[0],f.r_ret)
        .get_threadstate(f.r_scratch[0])
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
    prepare_exc_handler = JumpTarget()
    entry_points = collections.OrderedDict()
    op = abi.ops if binary else abi.ops.Assembly()

    ceval = partial(compile_eval,
        op=op,
        abi=abi,
        tuning=tuning,
        local_name=local_name,
        prepare_exc_handler=prepare_exc_handler,
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

    if prepare_exc_handler.used:
        prepare_exc_handler.displacement = 0
        end_targets.append(prepare_exc_handler)
        functions.append(resolve_jumps(op,prepare_exc_handler_func(op,abi,tuning).code))

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



