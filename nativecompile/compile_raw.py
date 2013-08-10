
import sys
import dis
import weakref
import operator
import types
import itertools
import collections
from functools import partial, reduce, update_wrapper, wraps

from . import pyinternals
from . import debug
from .x86_ops import TEST_MNEMONICS, fits_in_sbyte, fits_in_sdword


PRINT_STACK_OFFSET = False


CALL_ALIGN_MASK = 0xf
MAX_ARGS = 6
DEBUG_TEMPS = 3
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

# copied from Python/ceval.c
WHY_NOT =       0x0001
WHY_EXCEPTION = 0x0002
WHY_RERAISE =   0x0004
WHY_RETURN =    0x0008
WHY_BREAK =     0x0010
WHY_CONTINUE =  0x0020
WHY_YIELD =     0x0040
WHY_SILENCED =  0x0080

CO_GENERATOR = 0x0020


BLOCK_LOOP = 1
BLOCK_EXCEPT = 2
BLOCK_FINALLY = 3
BLOCK_WITH = 4

EXCEPT_VALUES = 6

CALL_FLAG_VAR = 1
CALL_FLAG_KW = 2


class NCCompileError(SystemError):
    """A problem exists with the bytecode.

    Note: This compiler has certain constraints that the Python interpreter
    doesn't. Bytecode produced by CPython should never violate these constraints
    but arbitrary bytecode might.

    """


def aligned_size(x):
    return (x + CALL_ALIGN_MASK) & ~CALL_ALIGN_MASK


class StackManager:
    """Keeps track of values stored on the stack.

    values are laid out as follows, where s=STACK_ITEM_SIZE,
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
        self._tos_in_r_ret = False

        self.resets = []

        # this is a stack so nested blocks can have finer restrictions
        self.protected_items = [0]

        # indicates what the value of _offset was before reserve_stack was
        # called
        self._unreserved_offset = None

    def copy(self):
        r = StackManager(self.op,self.abi,self.local_mem_size)
        r._offset = self._offset
        r._tos_in_r_ret = self._tos_in_r_ret
        r.resets = self.resets[:]
        r.protected_items = self.protected_items[:]
        r._unreserved_offset = self._unreserved_offset
        return r

    def check_stack_space(self):
        if (self._offset + max(self.args-len(self.abi.r_arg),0)) * self.abi.ptr_size > self.local_mem_size:
            raise NCCompileError("Not enough stack space was reserved. This code object being compiled has an incorrect value for co_stacksize.")

    @property
    def offset(self):
        return self._offset

    @property
    def tos_in_r_ret(self):
        return self._tos_in_r_ret

    def set_offset(self,val,descr=None):
        if val is None:
            assert not self._tos_in_r_ret
            self._offset = None
        else:
            if val < self.protected_items[-1]:
                raise NCCompileError("The code being compiled tries to pop more items from the stack than allowed at the given scope")

            self._offset = val
            self.check_stack_space()

        return self.annotate(descr)

    def annotate(self,descr=None):
        return debug.annotate(self.offset,self._tos_in_r_ret,descr)

    def push_stack(self,x,descr=None):
        ann = self.set_offset(self.offset + 1,descr)
        return [self.op.mov(x,self[0])] + ann

    def pop_stack(self,x):
        return [self.op.mov(self[0],x)] + self.set_offset(self.offset - 1)

    def _stack_arg_at(self,n):
        assert n >= len(self.abi.r_arg)
        return self.abi.ops.Address(
            (n-len(self.abi.r_arg))*self.abi.ptr_size + self.abi.shadow,
            self.abi.r_sp)

    def arg_dest(self,n):
        return self.abi.r_arg[n] if n < len(self.abi.r_arg) else self._stack_arg_at(n)

    def push_arg(self,x,tempreg = None,n = None):
        assert self._unreserved_offset is not None, "reserve_stack must be called first"

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

    def push_tos(self):
        """r_ret is needed right now so if the TOS item hasn't been pushed onto
        the stack, do it now."""
        if self._tos_in_r_ret:
            self._tos_in_r_ret = False
            return self.push_stack(self.abi.r_ret)

        return []

    def use_tos(self):
        if self._tos_in_r_ret:
            self._tos_in_r_ret = False
            return True,self.annotate()

        return False,[]

    def set_tos(self):
        """r_ret is now set to the new top of stack"""
        assert not self._tos_in_r_ret
        self._tos_in_r_ret = True
        return self.annotate()

    def tos(self):
        return self.abi.r_ret if self._tos_in_r_ret else self[0]

    def conditional_jump(self,target,stack_extra=0):
        # I'm not sure if a new reset will ever need to be added anywhere
        # except the front
        offset = self.offset + stack_extra
        for i,r in enumerate(self.resets):
            if target == r[0]:
                assert offset == r[1]
                return
            if target < r[0]:
                self.resets.insert(i,(target,offset))
                return

        self.resets.append((target,offset))

    def unconditional_jump(self,target):
        self.conditional_jump(target)
        return self.set_offset(None)

    def current_pos(self,pos):
        if self.resets:
            assert pos <= self.resets[0][0]
            if pos == self.resets[0][0]:
                off = self.resets.pop(0)[1]

                if self.offset is None:
                    return self.set_offset(off)
                elif off != self.offset:
                    raise NCCompileError('The stack size is not consistent at a particular location depending on branch taken')

        return []

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
        assert self._unreserved_offset is not None, "reserve_stack must be called first"

        self.args = 0
        if isinstance(func,str):
            return [
                self.op.mov(pyinternals.raw_addresses[func],self.abi.r_ret),
                self.op.call(self.abi.r_ret)]

        return [self.op.call(func)]

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
        if n is None: n = self.args
        return (self.abi.r_arg[n] if n < len(self.abi.r_arg)
                else (tempreg or self.abi.r_scratch[0]))

    def reserve_stack(self,move_sp=True):
        """Advance the stack pointer so that self.local_mem_size bytes are
        reserved and annotate it.

        This function assumes the stack pointer has already been moved by
        self.offset * self.abi.ptr_size bytes and only moves the stack by the
        difference.

        If move_sp is False, no instructions are emitted, but an annotation is
        still created.

        """
        assert self._unreserved_offset is None and self._offset is not None

        self._unreserved_offset = self._offset
        r= []
        if move_sp:
            r.append(self.op.sub(
                self.local_mem_size - self._offset * self.abi.ptr_size,
                self.abi.r_sp))

        return r + self.annotate(
                debug.PrologEnd(self.local_mem_size // self.abi.ptr_size))

    def release_stack(self):
        """Revert the stack pointer and self.offset to what they were before
        reserve_stack was called."""

        assert self._unreserved_offset is not None

        old_o = self._unreserved_offset
        self._unreserved_offset = None

        return [
            self.op.add(
                self.local_mem_size - old_o * self.abi.ptr_size,
                self.abi.r_sp)
        ] + self.set_offset(old_o,debug.EPILOG_START)



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
    displacement = None


class InnerCall(DelayedCompile):
    """A function call with a relative target

    This is just like JumpSource, except the target is a different function and
    the exact offset depends on how much padding is needed between this
    source's function and the target function, which cannot be determined until
    the length of the entire source function is determined.

    """
    def __init__(self,opset,abi,target,jump_instead=False):
        self.op = opset.jmp if jump_instead else opset.call
        self.length = abi.ops.JMP_DISP_MAX_LEN if jump_instead else abi.ops.CALL_DISP_LEN
        self.abi = abi
        self.target = target
        target.used = True

    def compile(self):
        r = self.op(self.abi.ops.Displacement(self.displacement + self.target.displacement,True))
        assert len(r) == self.length
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

    def compile(self):
        c = self.op(self.abi.ops.Displacement(self.displacement - self.target.displacement,True))
        assert len(c) == self.size
        return c

    def __len__(self):
        return self.size


class YieldMov(DelayedCompile):
    def __init__(self,ops,yield_from,yield_to,dest):
        #assert isinstance(dest,abi.ops.Address)

        self.op = ops.movl(0,dest)
        self.ops = ops
        self.from_ = yield_from
        yield_from.used = True
        self.to = yield_to
        yield_to.used = True

    def compile(self):
        return self.ops.with_new_imm_dword(
            self.op,
            # displacement is relative to the end of the code, so we subtract
            # the destination from the source
            self.from_.displacement - self.to.displacement)

    def __len__(self):
        return len(self.op)



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
    build_set_loop_threshhold = 5
    mem_copy_loop_threshhold = 9


handlers = [None] * 0xFF

def handler(func,name = None):
    opname = (name or func.__name__)[len('_op_'):]
    is_jump_forward = opname == 'JUMP_FORWARD'
    def inner(f,*extra):
        r = f()


        if f.forward_targets and f.forward_targets[0][0] <= f.byte_offset:
            pos,t,pop = f.forward_targets.pop(0)
            if pos != f.byte_offset:
                raise NCCompileError('A jump or branch instruction has a target not aligned to an instruction')
            r.push_tos()(t)
            if pop:
                r.pop_stack(f.r_scratch[1]).decref(f.r_scratch[1])

        if f.blockends and f.blockends[-1].offset <= f.byte_offset:
            b = f.blockends.pop()
            if b.offset != f.byte_offset:
                raise NCCompileError('A block purports to end at a byte offset not aligned to an instruction')
            if f.stack.offset is not None:
                r.push_tos()
            r += b.prepare(f)
            r(b.target)

        r += f.stack.current_pos(f.byte_offset)
        if f.stack.offset is not None:
            if PRINT_STACK_OFFSET:
                print('stack items: {}  opcode: {}'.format(
                    f.stack.offset + f.stack.tos_in_r_ret,
                    opname),file=sys.stderr)

            r.comment(opname)

            r += func(f,*extra)
        else:
            if PRINT_STACK_OFFSET:
                print('opcode {} omitted'.format(opname),file=sys.stderr)

            r.comment(opname + ' omitted')

            om = getattr(func,'omitted',None)
            if om: om(f,*extra)

        f.last_op_is_jump = is_jump_forward
        return r

    handlers[dis.opmap[opname]] = inner
    return func


def hasconst(func):
    return update_wrapper((lambda f,arg: func(f,f.code.co_consts[arg])),func)

def hasname(func):
    return update_wrapper((lambda f,arg: func(f,f.code.co_names[arg])),func)

def hasoffset(func):
    return update_wrapper((lambda f,arg: func(f,f.next_byte_offset+arg)),func)


def get_handler(op):
    h = handlers[op]
    if h is None:
        raise Exception('op code {} is not implemented'.format(dis.opname[op]))
    return h


class BlockEnd:
    def __init__(self,type,offset,stack=None):
        self.type = type
        self.offset = offset # byte offset in the byte-code
        self.target = JumpTarget()
        self.stack = stack

    def prepare(self,f):
        return []

class LoopBlockEnd(BlockEnd):
    def __init__(self,offset):
        super().__init__(BLOCK_LOOP,offset)

        # If a "continue" statement is encountered inside this block, store its
        # destination here and set "stack". This is needed by END_FINALLY.
        self.continue_offset = None

        self.has_break = False


class HandlerBlock:
    def __init__(self,type,stack,has_return=False):
        self.type = type
        self.stack = stack
        self.has_return = has_return

class ExceptBlockEnd(BlockEnd):
    def __init__(self,type,offset,stack,protect=EXCEPT_VALUES,extra_code=(lambda f: [])):
        # "except" and "finally" blocks will have EXCEPT_VALUES values on the
        # stack for them to inspect
        super().__init__(type,offset,stack+EXCEPT_VALUES)
        self.protect = stack + protect
        self.extra_code = extra_code
        self.has_return = False

    def prepare(self,f):
        # run it first in case it alters the stack
        r = self.extra_code(f)

        if f.stack.offset is None:
            r += f.stack.set_offset(self.stack)
        elif f.stack.offset != self.stack:
            raise NCCompileError('Incorrect stack size at except/finally block')

        f.handler_blocks.append(HandlerBlock(self.type,self.protect,self.has_return))
        f.stack.protected_items.append(self.protect)

        return r

def is_finally(x):
    return x.type == BLOCK_FINALLY or x.type == BLOCK_WITH

class OmittedExceptBlockEnd:
    def __init__(self,type,offset):
        self.type = type
        self.offset = offset # byte offset in the byte-code

    def prepare(self,f):
        f.handler_blocks.append(HandlerBlock(self.type,None))
        return []


class CType:
    def __init__(self,abi,t,base,index=None,scale=None):
        self.abi = abi
        self.offsets = pyinternals.member_offsets[t]
        self.args = base,index,(scale or abi.ptr_size)

    def __getattr__(self,name):
        offset = self.offsets.get(name)
        if offset is None: raise AttributeError(name)
        return self.abi.ops.Address(offset,*self.args)


def raw_addr_if_str(x):
    return pyinternals.raw_addresses[x] if isinstance(x,str) else x


class UtilityFunctions:
    def __init__(self):
        self.local_name = JumpTarget()
        self.global_name = JumpTarget()
        self.prepare_exc_handler = JumpTarget()
        self.reraise_exc_handler = JumpTarget()
        self.unwind_exc = JumpTarget()
        self.unwind_finally = JumpTarget()
        self.swap_exc_state = JumpTarget()

class Frame:
    def __init__(self,op,abi,tuning,local_mem_size,code=None,util_funcs=None,entry_points=None):
        self.code = code
        self.op = op
        self.abi = abi
        self.tuning = tuning
        self.stack = StackManager(op,abi,local_mem_size)
        self.end = JumpTarget()
        self.yield_start = JumpTarget()
        self.util_funcs=util_funcs
        self.blockends = []
        self.handler_blocks = []
        self.byte_offset = None
        self.next_byte_offset = None
        self.forward_targets = []
        self.entry_points = entry_points
        self.last_op_is_jump = False

        # Although JUMP_ABSOLUTE could jump to any instruction, we assume
        # compiled Python code only uses it in certain cases. Thus, we can make
        # small optimizations between successive instructions without needing
        # an extra pass over the byte code to determine all the jump targets.
        self.rtargets = {}

        # Indicates how many values tracked by Frame.stack were added as part
        # of the function prolog. This is needed by YIELD_VALUE.
        self.stack_prolog = None

        # The address where the throw flag is stored. This is used upon
        # resumption into YIELD_VALUE (or at the start if the generator just
        # started) and is invalid afterwards
        self.throw_flag_store = None


        a = (self.Address(i,abi.r_bp) for i in
             itertools.count(abi.ptr_size * -(SAVED_REGS+1),-abi.ptr_size))

        self.FRAME = next(a)
        self.GLOBALS = next(a)
        self.BUILTINS = next(a)
        self.LOCALS = next(a)
        self.FAST_LOCALS = next(a)
        self.TEMP = next(a)

        if pyinternals.REF_DEBUG:
            self.TEMP_EAX = next(a)
            self.TEMP_ECX = next(a)
            self.TEMP_EDX = next(a)
        elif pyinternals.COUNT_ALLOCS:
            self.COUNT_ALLOCS_TEMP = next(a)

    def type_of(self,r):
        return self.ctype('PyObject',r).ob_type

    def type_flags_of(self,r):
        return self.ctype('PyTypeObject',r).tp_flags

    def tuple_item(self,r,n):
        return self.ctype('PyTupleObject',r).ob_item + self.ptr_size * n

    def current_except_or_finally(self):
        for b in reversed(self.blockends):
            if b.type == BLOCK_EXCEPT or is_finally(b): return b
        return None

    def current_finally(self):
        for b in reversed(self.blockends):
            if is_finally(b): return b
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

    def ctype(self,*args):
        return CType(self.abi,*args)

    def dword(self,r):
        return self.Register(self.abi.ops.SIZE_D,r.code)

    def fits_imm32(self,x):
        """Return True if x fits in a 32-bit immediate value without
        sign-extend.

        32-bit immediate values are interpreted as signed. In 64-bit mode,
        these values get sign-extended to 64 bits and thus have their binary
        representation altered, which can make a difference when comparing
        addresses.

        """
        if self.ptr_size == 8:
            return fits_in_sdword(x)

        return True

    def check_err(self,inverted=False):
        if inverted:
            return self.if_eax_is_not_zero(self.exc_goto_end())

        return self.if_eax_is_zero(self.exc_goto_end())

    def invoke(self,func,*args,stack=None):
        if not stack: stack = self.stack
        return join([stack.push_arg(raw_addr_if_str(a)) for a in args]) + stack.call(func)

    def _if_eax_is(self,test,opcodes):
        if isinstance(opcodes,(bytes,self.abi.ops.AsmSequence)):
            return [
                self.op.test(self.r_ret,self.r_ret),
                self.jcc(~test,self.Displacement(len(opcodes))),
                opcodes]

        after = JumpTarget()
        return [
            self.op.test(self.r_ret,self.r_ret),
            self.jcc(~test,after)
        ] + opcodes + [
            after
        ]

    def if_eax_is_zero(self,opcodes): return self._if_eax_is(self.test_Z,opcodes)
    def if_eax_is_not_zero(self,opcodes): return self._if_eax_is(self.test_NZ,opcodes)

    def goto(self,target):
        return JumpSource(self.op.jmp,self.abi,target)

    def do_while_cond(self,cond,opcodes):
        if isinstance(opcodes,(bytes,self.abi.ops.AsmSequence)):
            cl = len(opcodes)

            jsize = self.JCC_MAX_LEN
            if fits_in_sbyte(cl + self.JCC_MIN_LEN):
                jsize = self.JCC_MIN_LEN

            return [opcodes,self.jcc(cond,self.Displacement(-(cl+jsize)))]

        start = JumpTarget()
        return [
            start
        ] + opcodes + [
            JumpRSource(partial(self.jcc,cond),self.abi,self.JCC_MAX_LEN,start)
        ]

    def unwind_handler(self,down_to=0,stack=None):
        if not stack: stack = self.stack

        assert len(stack.protected_items) > len(self.handler_blocks)

        r = []

        # since subsequent exception-unwinds overwrite previous exception-
        # unwinds, we could skip everything before the last except block at the
        # cost of making the functions at unwind_exc and unwind_finally check
        # for nulls when freeing the other stack items

        for hblock in reversed(self.handler_blocks):
            assert hblock.stack is not None and stack.offset >= hblock.stack

            is_exc = hblock.type == BLOCK_EXCEPT

            if (hblock.stack - (3 if is_exc else 6)) < down_to: break

            stack.protected_items.pop()

            h_free = stack.offset - hblock.stack

            r.append(self.op.lea(stack[h_free],self.r_pres[0]))
            r += self.mov(-h_free,self.r_scratch[0])
            r.append(InnerCall(
                self.op,
                self.abi,
                self.util_funcs.unwind_exc
                    if is_exc else
                self.util_funcs.unwind_finally))
            h_free += 3 if is_exc else 6
            r += stack.set_offset(stack.offset - h_free)

        return r

    def end_func(self,exception=False,stack=None):
        if not stack: stack = self.stack

        to_free = -stack.offset
        if self.stack_prolog is not None: to_free += self.stack_prolog

        r = self.mov(to_free,self.r_pres[0])
        if exception: r += self.mov(0,self.r_ret)
        r.append(self.goto(self.end))

        return r

    def exc_goto_end(self,reraise=False,stack=None):
        """Go to the inner most handler block or the end of the function if
        there isn't an appropriate block.

        stack defaults to self.stack and is not altered.

        """
        stack = (stack or self.stack).copy()

        pre_stack = 0
        block = self.current_except_or_finally()
        if block:
            pre_stack = block.stack - EXCEPT_VALUES

        r = self.unwind_handler(pre_stack,stack)

        if block:
            extra = stack.offset - pre_stack
            assert extra >= 0

            r.append(self.op.lea(stack[extra],self.r_pres[0]))
            r += self.mov(-extra,self.r_scratch[0])
            r.append(InnerCall(
                self.op,
                self.abi,
                self.util_funcs.reraise_exc_handler
                    if reraise else
                self.util_funcs.prepare_exc_handler))

            r.append(self.goto(block.target))
        else:
            r += self.end_func(True,stack)

        return r

    def incref(self,reg=None,amount=1):
        """Generate instructions equivalent to Py_INCREF"""

        if reg is None: reg = self.r_ret
        if pyinternals.REF_DEBUG:
            # the registers that would otherwise be undisturbed, must be
            # preserved
            return ([
                self.op.mov(self.r_ret,self.TEMP_EAX),
                self.op.mov(self.r_scratch[0],self.TEMP_ECX),
                self.op.mov(self.r_scratch[1],self.TEMP_EDX)
            ] + (self.invoke('Py_IncRef',reg) * amount) + [
                self.op.mov(self.TEMP_EDX,self.r_scratch[1]),
                self.op.mov(self.TEMP_ECX,self.r_scratch[0]),
                self.op.mov(self.TEMP_EAX,self.r_ret)])

        return [self.add(amount,self.ctype('PyObject',reg).ob_refcnt)]

    def decref(self,reg=None,preserve_reg=None,amount=1):
        """Generate instructions equivalent to Py_DECREF"""
        assert amount > 0

        if reg is None: reg = self.r_ret

        if pyinternals.REF_DEBUG:
            inv = self.invoke('Py_DecRef',reg)

            if amount > 1:
                inv = self.mov(reg,self.TEMP_ECX) + inv + self.invoke('Py_DecRef',self.TEMP_ECX) * (amount-1)

            if preserve_reg:
                inv = [self.op.mov(preserve_reg,self.TEMP)] + inv + [self.op.mov(self.TEMP,preserve_reg)]

            return inv

        mid = []

        if preserve_reg:
            mid.append(self.op.mov(preserve_reg,self.TEMP))

        mid.append(self.op.mov(
            self.ctype('PyObject',reg).ob_type,
            self.r_scratch[0]))

        if pyinternals.COUNT_ALLOCS:
            mid.append(self.op.mov(self.r_scratch[0],self.COUNT_ALLOCS_TEMP))

        mid.extend(self.invoke(
            self.ctype('PyTypeObject',self.r_scratch[0]).tp_dealloc,
            reg))

        if pyinternals.COUNT_ALLOCS:
            mid.extend(self.invoke('inc_count',self.COUNT_ALLOCS_TEMP))

        if preserve_reg:
            mid.append(self.op.mov(self.TEMP,preserve_reg))
        elif preserve_reg == 0:
            mid.extend(self.mov(0,self.r_ret))

        mid = join(mid)

        return [
            self.sub(amount,self.ctype('PyObject',reg).ob_refcnt),
            self.jnz(self.Displacement(len(mid))),
            mid]

    def new_rtarget(self):
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
        return (JumpSource(op,self.abi,self.forward_target(to))
                if to > self.byte_offset else
            JumpRSource(op,self.abi,max_size,self.reverse_target(to)))

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
        cmpj = self.jnz(self.Displacement(-size))
        assert len(cmpj) == self.JCC_MIN_LEN

        r = [dr,cmpj]
        if not always_clean:
            r = [
                self.op.test(addr.index,addr.index),
                self.jz(self.Displacement(size))
            ] + r

        return r

    def add(self,a,b):
        if a == 1 and not self.tuning.prefer_addsub_over_incdec:
            if isinstance(b,self.Address):
                return self.op.incl(b)
            return self.op.inc(b)

        if isinstance(a,int) and isinstance(b,self.Address):
            return (self.op.addq if self.ptr_size == 8 else self.op.addl)(a,b)
        return self.op.add(a,b)

    def sub(self,a,b):
        if a == 1 and not self.tuning.prefer_addsub_over_incdec:
            if isinstance(b,self.Address):
                return self.op.decl(b)
            return self.op.dec(b)

        if isinstance(a,int) and isinstance(b,self.Address):
            return (self.op.subq if self.ptr_size == 8 else self.op.subl)(a,b)
        return self.op.sub(a,b)

    # Note: this function assumes little-endian format
    def mov(self,a,b):
        if a != b:
            if a == 0 and isinstance(b,self.Register):
                return [self.op.xor(b,b)]
            if isinstance(a,int) and isinstance(b,self.Address):
                if self.ptr_size == 8:
                    if fits_in_sdword(a):
                        return [self.op.movq(a,b)]
                    return [self.op.movl(a & 0xffffffff,b),self.op.movl(a >> 32,b+4)]
                return [self.op.movl(a,b)]
            return [self.op.mov(a,b)]

        return []

    def cmp(self,a,b):
        assert (not isinstance(a,int)) or self.fits_imm32(a)

        if isinstance(a,int) and isinstance(b,self.Address):
            return (self.op.cmpq if self.ptr_size == 8 else self.op.cmpl)(a,b)
        return self.op.cmp(a,b)

    def get_threadstate(self,reg):
        # as far as I can tell, after expanding the macros and removing the
        # "dynamic annotations" (see dynamic_annotations.h in the CPython
        # headers), this is all that PyThreadState_GET boils down to:
        addr = pyinternals.raw_addresses['_PyThreadState_Current']
        if self.fits_imm32(addr):
            return [self.op.mov(self.Address(addr),reg)]
        return [self.op.mov(addr,reg),self.op.mov(self.Address(base=reg),reg)]

    def jcc(self,test,x):
        if isinstance(x,JumpTarget):
            return JumpSource(partial(self.op.jcc,test),self.abi,x)

        return self.op.jcc(test,x)

    def __call__(self):
        return Stitch(self)


def get_abi_ops_x(x):
    return property(lambda self: getattr(self.abi.ops,x))

for x in (
        'Address',
        'Register',
        'Displacement',
        'Test',
        'test_E',
        'test_Z',
        'test_NE',
        'test_NZ',
        'test_L',
        'test_G',
        'test_LE',
        'test_GE',
        'test_NB',
        'test_A',
        'test_BE',
        'test_B',
        'CALL_DISP_LEN',
        'JCC_MIN_LEN',
        'JCC_MAX_LEN',
        'JMP_DISP_MIN_LEN',
        'JMP_DISP_MAX_LEN',
        'LOOP_LEN'):
    setattr(Frame,x,get_abi_ops_x(x))

def frame_jx(val):
    return property(lambda self: partial(self.jcc,self.Test(val)))

for val,mns in enumerate(TEST_MNEMONICS):
    p = frame_jx(val)
    setattr(Frame,'j'+mns[0],p)
    for mn in mns[1:]: setattr(Frame,'j'+mn,p)
del p


class _subscr_proxy:
    def __init__(self,inst,func,arg1):
        self.inst = inst
        self.func = func
        self.arg1 = arg1
    def __getitem__(self,arg2):
        return self.func(self.inst,self.arg1,arg2)

def arg2_as_subscr(func):
    """Some syntax sugar that allows calling a method as:
    object.method(arg1)[arg2]

    arg1 refers to the first argument *after* self.

    """
    return lambda self,arg1: _subscr_proxy(self,func,arg1)


def strs_to_addrs(func):
    return lambda *args: func(*map(raw_addr_if_str,args))


def destitch(x):
    return x.code if isinstance(x,Stitch) else x


class _do_while_cond_proxy2:
    def __init__(self,stitch,code):
        self.stitch = stitch
        self.code = code

    def while_cond(self,cond):
        self.stitch.code.append(self.stitch.f.do_while_cond(cond,self.code))
        return self.stitch

class _do_while_cond_proxy1:
    def __init__(self,stitch):
        self.stitch = stitch

    def __getitem__(self,code):
        return _do_while_cond_proxy2(self.stitch,code)


CMP_LT = 0
CMP_LE = 1
CMP_GT = 2
CMP_GE = 3
CMP_EQ = 4
CMP_NE = 5

CMP_MIRROR = [CMP_GT,CMP_GE,CMP_LT,CMP_LE,CMP_EQ,CMP_NE]


class RegTest:
    def __init__(self,val,signed=True):
        self.val = val
        self.signed = signed

    def code(self,f,dest):
        assert isinstance(self.val,f.Register)
        return [f.op.test(self.val,self.val),f.jcc(f.test_Z,dest)]

def regtest_cmp(cmp):
    def inner(self,b):
        if isinstance(b,RegTest):
            assert self.signed == b.signed
            b = b.val
        return RegCmp(self.val,b,cmp,self.signed)
    return inner

for i,f in enumerate(['lt','le','gt','ge','eq','ne']):
    setattr(RegTest,'__{}__'.format(f),regtest_cmp(i))

signed = lambda val: RegTest(val,True)
unsigned = lambda val: RegTest(val,False)

class RegCmp:
    def __init__(self,a,b,cmp,signed):
        self.a = a
        self.b = b
        self.cmp = cmp
        self.signed = signed

    def code(self,f,dest):
        a = self.a
        b = self.b
        cmp = self.cmp
        if ((a == 0 and not isinstance(b,f.Address)) or
                isinstance(a,f.Address) or
                (b != 0 and isinstance(b,int))):
            a,b = b,a
            cmp = CMP_MIRROR[cmp]

        # test is inverted because we want to jump when the condition is false
        test = ([f.test_GE,f.test_G,f.test_LE,f.test_L,f.test_NE,f.test_E]
                    if self.signed else
                [f.test_NB,f.test_A,f.test_BE,f.test_B,f.test_NE,f.test_E])[cmp]

        return [
            f.op.test(a,a) if b == 0 else f.cmp(a,b),
            f.jcc(test,dest)]


F_METHODS = set('j'+mn for mn in itertools.chain.from_iterable(TEST_MNEMONICS))
for op in ['jcc','goto','add','sub']: F_METHODS.add(op)
F_METHODS = frozenset(F_METHODS)

class Stitch:
    """Generate a sequence of machine code instructions concisely using method
    chaining"""
    def __init__(self,frame,code=None):
        self.f = frame
        self._code = code or []
        self.stacks = [frame.stack]

    @property
    def stack(self):
        return self.stacks[-1]

    def branch(self):
        self.stacks.append(self.stack.copy())
        return self

    def end_branch(self):
        assert len(self.stacks) > 1
        self.stacks.pop()
        self._code += self.stack.annotate()
        return self

    @property
    def code(self):
        assert len(self.stacks) == 1
        return self._code

    def use_tos(self):
        r,c = self.stack.use_tos()
        self._code += c
        return r

    def push_arg(self,x,*args,**kwds):
        self._code += self.stack.push_arg(raw_addr_if_str(x),*args,**kwds)
        return self

    def clear_args(self):
        self.stack.args = 0
        return self

    def add_to_stack(self,n,descr=None):
        self._code += self.stack.set_offset(self.stack.offset + n,descr)
        return self

    def set_stack(self,n,descr=None):
        self._code += self.stack.set_offset(n,descr)
        return self

    @strs_to_addrs
    def mov(self,a,b):
        self._code += self.f.mov(a,b)
        return self

    @strs_to_addrs
    def cmp(self,a,b):
        self._code.append(self.f.cmp(a,b))
        return self

    def if_eax_is_zero(self,opcodes):
        self._code += self.f.if_eax_is_zero(destitch(opcodes))
        return self

    def if_eax_is_not_zero(self,opcodes):
        self._code += self.f.if_eax_is_not_zero(destitch(opcodes))
        return self

    def clean_stack(self,always_clean=False,addr=None,index=None):
        # making sure we're using self.stack instead of self.f.stack
        self._code += self.f.clean_stack(
            always_clean,
            addr if addr is not None else self.stack[0],
            index)
        return self

    @arg2_as_subscr
    def if_cond(self,test,code):
        if isinstance(code,(bytes,self.f.abi.ops.AsmSequence)):
            self._code.append(self.f.jcc(~test,self.f.Displacement(len(code))))
            self._code.append(code)
        else:
            after = JumpTarget()
            self._code.append(self.f.jcc(~test,after))
            self._code += destitch(code)
            self._code.append(after)

        return self

    @arg2_as_subscr
    def if_(self,test,code):
        if isinstance(test,self.f.Register):
            test = RegTest(test)

        if isinstance(code,(bytes,self.f.abi.ops.AsmSequence)):
            self._code += test.code(self.f,self.f.Displacement(len(code)))
            self._code.append(code)
        else:
            after = JumpTarget()
            self._code += test.code(self.f,after)
            self._code += destitch(code)
            self._code.append(after)

        return self

    do = property(_do_while_cond_proxy1)

    def comment(self,c):
        if isinstance(self.f.op,self.f.abi.ops.Assembly):
            self._code.append(self.f.op.comment(c))
        return self

    def save_reg(self,r):
        return self.push(r).add_to_stack(1,debug.SaveReg(r))

    def restore_reg(self,r):
        return self.pop(r).add_to_stack(-1,debug.RestoreReg(r))

    def del_stack_items(self,n):
        for _ in range(n):
            self._code += self.stack.pop_stack(self.f.r_ret)
            self._code += self.f.decref()
        return self

    def __add__(self,b):
        assert len(self.stacks) == 1

        if isinstance(b,Stitch):
            assert self.f is b.f
            return Stitch(self.f,self._code+b._code)
        if isinstance(b,list):
            return Stitch(self.f,self._code+b)

        return NotImplemented

    def __iadd__(self,b):
        if isinstance(b,Stitch):
            assert self.f is b.f
            self._code += b._code
            return self
        if isinstance(b,list):
            self._code += b
            return self

        return NotImplemented

    def __getattr__(self,name):
        func = getattr((self.f if name in F_METHODS else self.f.op),name)
        @strs_to_addrs
        def inner(*args):
            self._code.append(func(*args))
            return self
        return inner

    def __call__(self,op):
        self._code.append(op)
        return self


def _forward_stack_func(func):
    def inner(self,*args,**kwds):
        self._code += func(self.stack,*args,**kwds)
        return self
    return inner

for func in [
    'push_tos',
    'set_tos',
    'push_stack',
    'pop_stack',
    'call',
    'reserve_stack',
    'release_stack']:
    setattr(Stitch,func,_forward_stack_func(getattr(StackManager,func)))

Stitch.set_stack = _forward_stack_func(StackManager.set_offset)

def _forward_list_func(func):
    def inner(self,*args,**kwds):
        self._code += func(self.f,*args,**kwds)
        return self
    return inner

for func in [
    'check_err',
    'incref',
    'decref',
    'get_threadstate']:
    setattr(Stitch,func,_forward_list_func(getattr(Frame,func)))

def _forward_list_func(func):
    def inner(self,*args,**kwds):
        self._code += func(self.f,*args,stack=self.stack,**kwds)
        return self
    return inner

for func in [
    'invoke',
    'exc_goto_end',
    'unwind_handler',
    'end_func']:
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
        .del_stack_items(1)
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

def _op_add(f,func):
    not_unicode = JumpTarget()
    end = JumpTarget()
    err = JumpTarget()

    r_arg1 = f.stack.arg_reg(f.r_scratch[0],0)
    r_arg2 = f.stack.arg_reg(f.r_ret,1)

    r = f()

    str_addr = pyinternals.raw_addresses['PyUnicode_Type']
    if not f.fits_imm32(str_addr):
        r.mov(str_addr,f.r_pres[1])
        str_addr = f.r_pres[1]

    future = f.code.co_code[f.next_byte_offset:]
    nextop = future[0]
    nextarg = 0
    if nextop >= dis.HAVE_ARGUMENT:
        nextarg = (future[2] << 8) + future[1]

    tos = f.stack.tos()
    return (r
        .push_tos()
        .mov(tos,r_arg2)
        .mov(f.stack[1],r_arg1)
        .mov(f.type_of(r_arg2),f.r_scratch[1])
        .mov(f.type_of(r_arg1),f.r_pres[0])

        .cmp(str_addr,f.r_scratch[1])
        .jne(not_unicode)
        .cmp(str_addr,f.r_pres[0])
        .jne(not_unicode)
            .invoke('unicode_concatenate',
                r_arg1,
                r_arg2,
                f.FRAME,
                nextop,
                nextarg
            )
            .test(f.r_ret,f.r_ret)
            .jz(err)

            # unicode_concatenate consumes the reference to r_arg1 (f.stack[1])
            .mov(f.r_ret,f.stack[1])

            .goto(end)
        (not_unicode)
            .invoke(func,r_arg1,r_arg2)
            .if_eax_is_zero(f()
                (err)
                .exc_goto_end()
            )
            .mov(f.stack[1],f.r_scratch[0])
            .mov(f.r_ret,f.stack[1])
            .decref(f.r_scratch[0])
        (end)
        .del_stack_items(1)
    )

@handler
def _op_BINARY_ADD(f):
    return _op_add(f,'PyNumber_Add')

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
    return _op_add(f,'PyNumber_InPlaceAdd')

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
    r = f()
    if not r.use_tos():
        r.pop_stack(f.r_ret)
    return r.decref()

def _op_load_name(func,f,name):
    return (f()
        .push_tos()
        .mov(address_of(name),f.r_pres[0])
        (InnerCall(f.op,f.abi,func))
        .check_err()
        .set_tos()
    )

@handler
@hasname
def _op_LOAD_NAME(f,name):
    return _op_load_name(f.util_funcs.local_name,f,name)

@handler
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
        dict_addr = f.r_scratch[1]

    return (r
        .if_eax_is_zero(f()
            .clear_args()
            .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_STORE_MSG')
            .exc_goto_end()
        )
        .mov('PyObject_SetItem',f.r_scratch[0])
        .push_arg(f.r_ret,n=0)
        .if_(signed(dict_addr) == f.type_of(f.r_ret)) [f()
            .mov('PyDict_SetItem',f.r_scratch[0])
        ]
        .call(f.r_scratch[0])
        .check_err(True)
        .del_stack_items(1)
    )

@handler
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
            .exc_goto_end()
        )
        .invoke('PyObject_DelItem',f.r_ret,address_of(name))
        .if_eax_is_not_zero(f()
            .invoke('format_exc_check_arg',
                'PyExc_NameError',
                'NAME_ERROR_MSG',
                address_of(name))
            .exc_goto_end()
        )
    )

@handler
@hasname
def _op_LOAD_GLOBAL(f,name):
    return _op_load_name(f.util_funcs.global_name,f,name)

@handler
@hasname
def _op_STORE_GLOBAL(f,name):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyDict_SetItem',f.GLOBALS,address_of(name),tos)
        .check_err(True)
        .del_stack_items(1)
    )

@handler
@hasconst
def _op_LOAD_CONST(f,const):
    if isinstance(const,types.CodeType):
        const = f.entry_points[id(const)][0]

    return (f()
        .push_tos()
        .mov(address_of(const),f.r_ret)
        .set_tos()
        .incref()
    )

@handler
def _op_CALL_FUNCTION(f,arg):
    argreg = f.stack.arg_reg(n=0)
    return (f()
        .push_tos()
        .push_arg(arg,n=1)
        .lea(f.stack[0],argreg)
        .push_arg(argreg,n=0)
        .call('call_function')

        # +1 for the function object
        .add_to_stack(-((arg & 0xFF) + ((arg >> 8) & 0xFF) * 2 + 1))

        .check_err()
        .set_tos()
    )

@handler
def _op_RETURN_VALUE(f):
    # .unwind_handler() alters StackManager.protected_items
    r = f().branch()

    block = f.current_finally()

    if f.handler_blocks:
        limit = 0
        if block: limit = block.stack - EXCEPT_VALUES

        r.push_tos()
        valaddr = r.stack[0]
        r.add_to_stack(-1)
        r.unwind_handler(limit)
        r.mov(valaddr,f.r_ret)
    elif not r.use_tos():
        r.pop_stack(f.r_ret)

    if block:
        r.mov('Py_None',f.r_scratch[0])
        r.add_to_stack(EXCEPT_VALUES-2)
        r.push_stack(f.r_ret)

        for i in range(1,EXCEPT_VALUES-1):
            r.mov(f.r_scratch[0],r.stack[i])

        r.incref(f.r_scratch[0],EXCEPT_VALUES-2)

        # CPython keeps an array of pre-allocated integer objects between -5
        # and 256. For values in that range, this function never raises an
        # exception, thus no error checking is done here.
        r.invoke('PyLong_FromLong',WHY_RETURN)
        r.push_stack(f.r_ret)

        r.goto(block.target)

    else:
        r.end_func()

    for b in f.blockends:
        if isinstance(b,ExceptBlockEnd):
            b.has_return = True

    r.end_branch()
    r.use_tos()
    return r.set_stack(None)

@handler
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

@handler
@hasoffset
def _op_FOR_ITER(f,to):
    argreg = f.stack.arg_reg(n=0)
    return (f()
        .push_tos()
        (f.new_rtarget())
        .mov(f.stack[0],argreg)
        .mov(f.type_of(argreg),f.r_ret)
        .mov(f.ctype('PyTypeObject',f.r_ret).tp_iternext,f.r_ret)
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
        .set_tos()
    )

@handler
def _op_JUMP_ABSOLUTE(f,to):
    assert to < f.byte_offset
    return f().push_tos()(
        JumpRSource(f.op.jmp,f.abi,f.JMP_DISP_MAX_LEN,f.reverse_target(to)))

@handler
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

def _op_pop_jump_if_(f,to,jop1,jop2):
    dont_jump = JumpTarget()
    tos = f.stack.tos()
    r = (f()
        .push_tos()
        .invoke('PyObject_IsTrue',tos)
        .pop_stack(f.r_scratch[1])
        .decref(f.r_scratch[1],f.r_ret)
        .test(f.r_ret,f.r_ret)
        (jop1(dont_jump))
        (f.jump_to(jop2,f.JCC_MAX_LEN,to))
        .exc_goto_end()
        (dont_jump)
    )

    if to > f.byte_offset:
        f.stack.conditional_jump(to)

    return r

@handler
def _op_POP_JUMP_IF_FALSE(f,to):
    return _op_pop_jump_if_(f,to,f.jg,f.jz)

@handler
def _op_POP_JUMP_IF_TRUE(f,to):
    return _op_pop_jump_if_(f,to,f.jz,f.jg)

def _op_build_(f,items,new,ctype,deref):
    item_offset = pyinternals.member_offsets[ctype]['ob_item']

    r = (f()
        .push_tos()
        .invoke(new,items)
        .check_err()
        .set_tos())

    if items:
        if items >= f.tuning.build_seq_loop_threshhold:
            top = f.stack[0]
            top.scale = 4
            if deref:
                f.stack.tos_in_r_ret = False
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
                .loop(f.Displacement(-len(lbody) - f.LOOP_LEN))
                .add_to_stack(-items))
        else:
            if deref:
                r.mov(f.Address(item_offset,f.r_ret),f.r_pres[0])

            for i in reversed(range(items)):
                addr = f.Address(f.ptr_size*i,f.r_pres[0]) if deref else f.Address(item_offset+f.ptr_size*i,f.r_ret)
                r.pop_stack(f.r_scratch[1]).mov(f.r_scratch[1],addr)


    return r

@handler
def _op_BUILD_LIST(f,items):
    return _op_build_(f,items,'PyList_New','PyListObject',True)

@handler
def _op_BUILD_TUPLE(f,items):
    return _op_build_(f,items,'PyTuple_New','PyTupleObject',False)

@handler
def _op_STORE_SUBSCR(f):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_SetItem',f.stack[1],tos,f.stack[2])
        .check_err(True)
        .del_stack_items(3)
    )

def _op_make_callable(f,arg,closure):
    annotations = (arg >> 16) & 0x7fff

    # +2 for the name and code object
    sitems = (arg & 0xff) + ((arg >> 8) & 0xff) * 2 + annotations + 2

    if closure: sitems += 1
    if annotations: sitems += 1

    arg3reg = f.stack.arg_reg(n=2)

    return (f()
        .push_tos()
        .lea(f.stack[0],arg3reg)
        .invoke('_make_function',int(bool(closure)),arg,arg3reg)
        .add_to_stack(-sitems)
        .check_err()
        .set_tos()
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
        .push_tos()
        .mov(f.FAST_LOCALS,f.r_scratch[0])
        .mov(f.Address(f.ptr_size*arg,f.r_scratch[0]),f.r_ret)
        .if_eax_is_zero(f()
            .invoke('format_exc_check_arg',
                'PyExc_UnboundLocalError',
                'UNBOUNDLOCAL_ERROR_MSG',
                address_of(f.code.co_varnames[arg]))
            .exc_goto_end()
        )
        .set_tos()
        .incref()
    )

@handler
def _op_STORE_FAST(f,arg):
    r = f()
    if not r.use_tos():
        r.pop_stack(f.r_ret)

    item = f.Address(f.ptr_size*arg,f.r_scratch[0])
    return (r
        .mov(f.FAST_LOCALS,f.r_scratch[0])
        .mov(item,f.r_scratch[1])
        .mov(f.r_ret,item)
        .if_(f.r_scratch[1]) [
            join(f.decref(f.r_scratch[1]))
        ]
    )

@handler
def _op_UNPACK_SEQUENCE(f,arg):
    assert arg > 0

    r = f()
    if r.use_tos():
        r.mov(f.r_ret,f.r_pres[0])
    else:
        r.pop_stack(f.r_pres[0])

    check_list = JumpTarget()
    else_ = JumpTarget()
    done = JumpTarget()

    s_top = f.stack[-1]
    s_top.scale = f.ptr_size

    # a place to temporarily store the sequence
    seq_store = f.TEMP

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
        .jne(check_list)
            .cmp(arg,f.ctype('PyVarObject',f.r_pres[0]).ob_size)
            .jne(else_))

    if arg >= f.tuning.unpack_seq_loop_threshhold:
        s_top.index = f.r_ret
        body = join(f()
            .mov(f.ctype('PyTupleObject',f.r_pres[0],f.r_scratch[0]).tp_item - f.ptr_size,f.r_scratch[1])
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
        .jne(else_)
            .cmp(arg,f.ctype('PyVarObject',f.r_pres[0]).ob_size)
            .jne(else_)
                .mov(f.ctype('PyListObject',f.r_pres[0]).ob_item,f.r_scratch[1]))

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


    p3 = f.stack.arg_reg(n=3)

    return (r
            .goto(done)
        (else_)
            .lea(f.stack[-arg],p3)
            .invoke('_unpack_iterable',f.r_pres[0],arg,-1,p3)
            .add_to_stack(arg)
            .if_eax_is_zero(f()
                .decref(f.r_pres[0])
                .exc_goto_end()
            )
        (done)
        .decref(f.r_pres[0]))

@handler
def _op_UNPACK_EX(f,arg):
    totalargs = 1 + (arg & 0xff) + (arg >> 8)

    r = f()
    if r.use_tos():
        r.mov(f.r_ret,f.r_pres[0])
    else:
        r.pop(f.r_pres[0])

    argreg = f.stack.arg_reg(n=3)

    return (r
        .lea(f.stack[-totalargs],argreg)
        .invoke('_unpack_iterable',f.r_pres[0],arg & 0xff,arg >> 8,argreg)
        .add_to_stack(totalargs)
        .decref(f.r_pres[0],preserve_reg=f.r_ret)
        .check_err()
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
            .del_stack_items(1)
        )

    if op == 'is' or op == 'is not':
        outcome_a,outcome_b = false_true_addr(op == 'is not')

        r = f()
        if not r.use_tos():
            r.pop_stack(f.r_ret)

        return (r
            .mov(outcome_a,f.r_pres[0])
            .if_(signed(f.r_ret) == f.stack[0]) [
                join(f.mov(outcome_b,f.r_pres[0]))
            ]
            .decref(f.r_ret)
            .pop_stack(f.r_ret)
            .decref(f.r_ret)
            .mov(f.r_pres[0],f.r_ret)
            .set_tos()
            .incref(f.r_pres[0])
        )

    if op == 'in' or op == 'not in':
        outcome_a,outcome_b = false_true_addr(op == 'not in')

        tos = f.stack.tos()
        return (f()
            .push_tos()
            .invoke('PySequence_Contains',tos,f.stack[1])
            .if_(signed(f.r_ret) < 0) [f()
                .exc_goto_end()
            ]
            .mov(outcome_a,f.r_ret)
            .if_cond(f.test_NZ) [
                f.op.mov(outcome_b,f.r_ret)
            ]
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
        if not r.use_tos():
            p1 = f.stack.arg_reg(tempreg=f.r_ret,n=1)
            r.pop_stack(p1)

        r.pop_stack(p0)
        r.push_arg(p1,n=1)
        r.push_arg(p0,n=0)
    elif arg == 1:
        p0 = f.r_ret
        if not r.use_tos():
            p0 = f.stack.arg_reg(n=0)
            r.pop_stack(p0)

        r.push_arg(0,n=1)
        r.push_arg(p0,n=0)
    elif arg == 0:
        r.push_tos()
        r.push_arg(0)
        r.push_arg(0)
    else:
        raise SystemError("bad RAISE_VARARGS oparg")

    # We don't have to worry about decrementing the reference counts. _do_raise
    # does that for us. It also returns 0 unconditionally.
    return (r
        .call('_do_raise')
        .exc_goto_end()
        .set_stack(None))

@handler
def _op_BUILD_MAP(f,arg):
    return (f()
        .push_tos()
        .invoke('_PyDict_NewPresized',arg)
        .check_err()
        .set_tos()
    )

def _op_map_store_add(offset,f):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyDict_SetItem',f.stack[2+offset],tos,f.stack[1])
        .check_err(True)
        .del_stack_items(2)
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
        .push_tos()
        .invoke('_load_build_class',f.BUILTINS)
        .check_err()
        .set_tos()
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
        .mov(f.r_ret,f.ctype('PyFrameObject',f.r_scratch[0]).f_locals)
    )

@handler
@hasname
def _op_STORE_ATTR(f,name):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_SetAttr',tos,address_of(name),f.stack[1])
        .mov(f.r_ret,f.r_pres[0])
        .del_stack_items(2)
        .if_(f.r_pres[0]) [f()
            .exc_goto_end()
        ]
    )

@handler
@hasname
def _op_IMPORT_FROM(f,name):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_GetAttr',tos,address_of(name))
        .if_eax_is_zero(f()
            .invoke('PyErr_ExceptionMatches','PyExc_AttributeError')
            .if_eax_is_not_zero(join(f()
               .invoke('PyErr_Format','CANNOT_IMPORT_MSG',address_of(name))
            .code))
            .exc_goto_end()
        )
        .set_tos()
    )

@handler
def _op_IMPORT_STAR(f):
    r = f()
    if not r.use_tos():
        r.pop_stack(f.r_ret)

    return (r
        # import_from decrements the reference count for us
        .invoke('import_all_from',f.FRAME,f.r_ret)
        .check_err(True)
    )

# Currently, the machine code produced by this function fails on x86_64 when
# Python is built with debugging. GCC seems to use a different calling
# convention for variadic functions in this case.
@handler
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
            .if_eax_is_zero(join(f()
                .push_arg('Py_None',n=f.stack.args-1).code
            ))
            .push_arg(f.stack[0]))

    r = (f()
        .push_tos()
        .invoke('_PyDict_GetItemId',f.BUILTINS,'PyId___import__')
        .if_eax_is_zero(f()
            .invoke('PyErr_SetString','PyExc_ImportError','IMPORT_NOT_FOUND_MSG')
            .exc_goto_end()
        )
        .mov(f.r_ret,r_imp)
        .incref()
        .invoke('PyLong_AsLong',f.stack[1])
        .if_(signed(f.r_ret) != -1) [f()
            .call('PyErr_Occurred')
            .if_eax_is_not_zero(
                prepare_args(5)
                .push_arg(f.stack[1])
                .call('PyTuple_Pack')
                .goto(endif)
            )
        ] +
        # else
            prepare_args(4)
            .call('PyTuple_Pack')
        (endif)
    )

    r_args = f.r_pres[1]
    return (r
        .mov(f.r_ret,r_args)
        .del_stack_items(2)
        .if_(signed(r_args) == 0) [f()
            .decref(r_imp)
            .exc_goto_end()
        ]

        .invoke('PyObject_Call',r_imp,r_args,0)

        .push_stack(f.r_ret)
        .decref(r_imp)
        .decref(r_args)
        .mov(f.stack[0],f.r_ret)
        .if_eax_is_zero(f()
            .add_to_stack(-1)
            .exc_goto_end()
            .add_to_stack(1)
        )
    )

def block_start(type,**kwds):
    def decorator(func):
        @wraps(func)
        @hasoffset
        def inner(f,to):
            r = func(f,to)

            f.blockends.append(ExceptBlockEnd(type,to,f.stack.offset,**kwds))
            f.stack.conditional_jump(to,EXCEPT_VALUES)

            return r

        inner.omitted = hasoffset(
            lambda f,to: f.blockends.append(OmittedExceptBlockEnd(type,to)))

        return inner
    return decorator

# When a finally block is entered without a jump, only one value will be
# pushed onto the stack for it to look at, so we pad the stack with 5 more
def add_nones(f):
    # if stack.offset is None, the finally block is never reached by
    # fall-through, only by an exception, return, break or continue
    if f.stack.offset is not None:
        r = f()
        r.comment("'finally' fixup")
        if not r.use_tos():
            r.pop_stack(f.r_ret)

        r.mov('Py_None',f.r_scratch[0])
        for x in range(EXCEPT_VALUES-1):
            r.push_stack(f.r_scratch[0])

        return (r
            .push_stack(f.r_ret)
            .incref(f.r_scratch[0],EXCEPT_VALUES-1)
        )
    return []

# unlike "except" and "finally" blocks, "with" blocks don't have user-defined
# handler code, so this function makes sure the next two opcodes are
# WITH_CLEANUP and END_FINALLY
def with_cleanup(f):
    c = f.code.co_code
    if (len(c) < f.byte_offset + 2 or
            c[f.byte_offset] != dis.opmap['WITH_CLEANUP'] or
            c[f.byte_offset+1] != dis.opmap['END_FINALLY']):
        raise NCCompileError('There is a "with" block that is not properly cleaned up')

    return add_nones(f)

@handler
@block_start(BLOCK_WITH,extra_code=with_cleanup)
def _op_SETUP_WITH(f,to):
    r = f()
    if r.use_tos():
        r.mov(f.r_ret,f.r_pres[0])
    else:
        r.pop_stack(f.r_pres[0])

    return (r
        .invoke('special_lookup',f.r_pres[0],'PyId___exit__')
        .if_eax_is_zero(f()
            .decref(f.r_pres[0])
            .exc_goto_end()
        )
        .push_stack(f.r_ret)
        .invoke('special_lookup',f.r_pres[0],'PyId___enter__')
        .decref(f.r_pres[0],preserve_reg=f.r_ret)
        .mov(f.r_ret,f.r_pres[0])
        .check_err()
        .invoke('PyObject_CallFunctionObjArgs',f.r_ret,0)
        .decref(f.r_pres[0],preserve_reg=f.r_ret)
        .check_err()
        .set_tos()
    )

# if this value changes, the code here will need updating
assert EXCEPT_VALUES == 6
@handler
@block_start(BLOCK_EXCEPT,protect=3)
def _op_SETUP_EXCEPT(f,to):
    return []

@handler
@block_start(BLOCK_FINALLY,extra_code=add_nones)
def _op_SETUP_FINALLY(f,to):
    return []

def pop_handler_block(f,type):
    def block_mismatch():
        raise NCCompileError(
            '{0} instruction not correctly matched with {1}'.format(*([
                ('POP_EXCEPT','SETUP_EXCEPT'),
                ('END_FINALLY','SETUP_FINALLY'),
                ('CLEANUP_WITH','SETUP_WITH')][type-BLOCK_EXCEPT])))

    if not f.handler_blocks:
        raise block_mismatch()
    block = f.handler_blocks.pop()
    if block.type != type:
        raise block_mismatch()

    if block.stack is not None:
        if f.stack.offset is not None:
            if f.stack.offset + f.stack.tos_in_r_ret != block.stack:
                raise NCCompileError(
                    'Incorrect stack size at {} instruction'.format(
                        ['POP_EXCEPT','END_FINALLY','CLEANUP_WITH'][type-BLOCK_EXCEPT]))

            # this shouldn't be possible
            assert not f.stack.tos_in_r_ret

        f.stack.protected_items.pop()

    return block

@handler
def _op_WITH_CLEANUP(f):
    # The checks need to be done here but we let END_FINALLY do all the work so
    # the block gets put back
    b = pop_handler_block(f,BLOCK_WITH)
    f.handler_blocks.append(b)

    assert f.code.co_code[f.next_byte_offset] == dis.opmap['END_FINALLY']

    return []

@handler
def _op_POP_EXCEPT(f):
    pop_handler_block(f,BLOCK_EXCEPT)

    return (f()
        .mov(0,f.r_scratch[0])
        .lea(f.stack[0],f.r_pres[0])
        (InnerCall(f.op,f.abi,f.util_funcs.unwind_exc))
        .add_to_stack(-3))

_op_POP_EXCEPT.omitted = lambda f: pop_handler_block(f,BLOCK_EXCEPT)

@handler
def _op_END_FINALLY(f):
    # if this value changes, the code here will need updating
    assert EXCEPT_VALUES == 6

    r_tmp1 = f.stack.arg_reg(f.r_scratch[0],0)
    r_tmp2 = f.stack.arg_reg(f.r_scratch[1],1)
    r_tmp3 = f.stack.arg_reg(f.r_ret,2)

    # In addition to ending actual "finally" blocks, CPython uses END_FINALLY
    # to reraise exceptions in exception handlers when the exception doesn't
    # match any of the types that the "except" blocks handle. In such a case
    # there will be a JUMP_FORWARD instruction right before END_FINALLY, to
    # skip END_FINALLY when the exception is handled. Since there is no reason
    # to skip that instruction in a genuine "finally" block, checking if the
    # previous instruction was JUMP_FORWARD should be a reliable way to
    # determine which is the case.
    if f.last_op_is_jump:
        assert not f.stack.tos_in_r_ret

        err = JumpTarget()
        r = f()
        return (r
            .mov(f.stack[0],r_tmp1)
            .mov(f.type_of(r_tmp1),r_tmp2)
            .testl(TPFLAGS_TYPE_SUBCLASS,f.type_flags_of(r_tmp2))
            .jz(err)
            .testl(TPFLAGS_BASE_EXC_SUBCLASS,f.type_flags_of(r_tmp1))
            .jz(err)

            .branch()
                .add_to_stack(-1)
                .pop_stack(r_tmp2)
                .pop_stack(r_tmp3)

                # PyErr_Restore steals references
                .invoke('PyErr_Restore',r_tmp1,r_tmp2,r_tmp3)
                .lea(r.stack[0],f.r_pres[0])
                .mov(0,f.r_scratch[0])
                (InnerCall(f.op,f.abi,f.util_funcs.unwind_exc))
                .add_to_stack(-3)
                .exc_goto_end(True)
            .end_branch()

            (err)
            .invoke('PyErr_SetString','PyExc_SystemError','BAD_EXCEPTION_MSG')
            .exc_goto_end()
            .set_stack(None)
        )


    if f.handler_blocks and f.handler_blocks[-1].type == BLOCK_WITH:
        # CLEANUP_WITH already called pop_handler_block (and pushed the block
        # back on) so we don't need to do it here
        hblock = f.handler_blocks.pop()
        with_cleanup = True
        r_tmp1 = f.r_pres[0]
    else:
        hblock = pop_handler_block(f,BLOCK_FINALLY)
        with_cleanup = False

    # find the next "finally" block and the next loop block if it comes before
    # the "finally" block
    nextl = None
    nextf = None
    for b in reversed(f.blockends):
        if b.type == BLOCK_LOOP:
            if not nextl: nextl = b
        elif is_finally(b):
            nextf = b
            break

    has_return = hblock.has_return
    has_continue = nextl and nextl.continue_offset is not None
    has_break = nextl and nextl.has_break

    not_long = JumpTarget()
    not_except = JumpTarget()
    err = JumpTarget()
    end = JumpTarget()

    n_addr = pyinternals.raw_addresses['Py_None']
    n_fits = f.fits_imm32(n_addr)


    r = f().mov(f.stack[0],r_tmp1)

    if not n_fits:
        r.mov(n_addr,f.r_pres[1])
        n_addr = f.r_pres[1]

    (r
        .mov(f.type_of(r_tmp1),r_tmp2)

        .testl(TPFLAGS_TYPE_SUBCLASS,f.type_flags_of(r_tmp2))
        .jz(not_except)
        .testl(TPFLAGS_BASE_EXC_SUBCLASS,f.type_flags_of(r_tmp1))
        .jz(not_except)
    )

    arg1 = r_tmp1
    if with_cleanup:
        exit_fail = JumpTarget()
        (r
            # call the stored __exit__ function
            .invoke('PyObject_CallFunctionObjArgs',f.stack[6],r_tmp1,f.stack[1],f.stack[2],0)
            .test(f.r_ret,f.r_ret)
            .jz(exit_fail)
            .mov(f.r_ret,f.r_pres[0])
            .invoke('PyObject_IsTrue',f.r_ret)
            .decref(f.r_pres[0],preserve_reg=f.r_ret)
            .test(f.r_ret,f.r_ret)
            .if_cond(f.test_L) [f()
                (exit_fail)
                .lea(f.stack[3],f.r_pres[0])
                .mov(-3,f.r_scratch[0])
                (InnerCall(f.op,f.abi,f.util_funcs.unwind_exc))
                .exc_goto_end()
            ]
            .if_cond(f.test_G) [f()
                .lea(f.stack[3],f.r_pres[0])
                .mov(-3,f.r_scratch[0])
                (InnerCall(f.op,f.abi,f.util_funcs.unwind_exc))
                .goto(end)
            ]
        )
        arg1 = f.stack[0]

    (r.branch()
        .add_to_stack(-1)
        .pop_stack(r_tmp2)
        .pop_stack(r_tmp3)

        # PyErr_Restore steals references
        .invoke('PyErr_Restore',arg1,r_tmp2,r_tmp3)
        .lea(r.stack[0],f.r_pres[0])
        .mov(0,f.r_scratch[0])
        (InnerCall(f.op,f.abi,f.util_funcs.unwind_exc))
        .add_to_stack(-3)
        .exc_goto_end(True)

    .end_branch())

    r(not_except)

    if with_cleanup:
        # call the stored __exit__ function
        r.invoke('PyObject_CallFunctionObjArgs',f.stack[6],n_addr,n_addr,n_addr,0)
        r.check_err()
        r.decref()

    if has_return or has_continue or has_break:
        if with_cleanup:
            r.mov(f.type_of(r_tmp1),r_tmp2) # r_tmp is r_pres[0]

        r.testl(TPFLAGS_LONG_SUBCLASS,f.type_flags_of(r_tmp2))
        r.jz(not_long)

        if nextf and f.stack.offset == nextf.stack:
            assert not (nextl or with_cleanup)
            # just go to the next finally block
            r.goto(nextf.target)
        else:
            check_break = JumpTarget()
            check_cont = JumpTarget()
            r.invoke('PyLong_AsLong',r_tmp1)

            if has_return:
                r.cmp(WHY_RETURN,f.r_ret)
                r.jne(check_break if has_break else (check_cont if has_continue else err))

                if nextf:
                    f_stack_diff = f.stack.offset - nextf.stack
                    assert f_stack_diff > 0

                    # free the extra values (including the __exit__ function)
                    # and shift our values down
                    for n in range(f_stack_diff):
                        r.mov(f.stack[n+EXCEPT_VALUES],f.r_ret).decref()

                    (r
                        .mov(f.stack[0],f.r_ret)
                        .mov(f.stack[1],f.r_scratch[0])
                        .mov(f.stack[2],f.r_scratch[1]) # Py_None
                        .mov(f.r_ret,f.stack[f_stack_diff])
                        .mov(f.r_scratch[0],f.stack[1+f_stack_diff])
                    )

                    # padding (if f_stack_diff is between 1 and 3 then that
                    # many stack items will already be Py_None)
                    for n in range(max(4-f_stack_diff,0),4):
                        r.mov(f.r_scratch[1],f.stack[2+n])

                    r.goto(nextf.target)
                else:
                    # the return value will stay on the stack for now
                    retaddr = r.stack[1]

                    (r.branch()
                        .mov(r.stack[0],f.r_scratch[0])
                        .decref(f.r_scratch[0])
                        .mov(r.stack[2],f.r_ret)
                        .decref(f.r_ret,amount=EXCEPT_VALUES-2) # all Py_None
                        .add_to_stack(-EXCEPT_VALUES)
                    )

                    # the __exit__ function, if present, will get popped off by
                    # either unwind_handler or end_func

                    if f.handler_blocks:
                        r.unwind_handler()

                    r.mov(retaddr,f.r_ret)
                    r.end_func()

                    r.end_branch()

            if has_break or has_continue:
                loop_jump_inner = (f().branch()
                    .pop_stack(f.r_scratch[0])
                    .decref(f.r_scratch[0])
                    .pop_stack(f.r_ret)
                    .decref(amount=EXCEPT_VALUES-1) # padding
                .end_branch())

                if with_cleanup:
                    # pop off __exit__ function
                    loop_jump_inner.pop_stack(f.r_ret).decref()

                if has_break:
                    r(check_break)
                    r.cmp(WHY_BREAK,f.r_ret)
                    r.jne(check_cont if has_continue else err)
                    r += loop_jump_inner
                    r.goto(nextl.target)

                if has_continue:
                    r(check_cont)
                    r.cmp(WHY_CONTINUE,f.r_ret)
                    r.jne(err)
                    r += loop_jump_inner
                    r(JumpRSource(
                            f.op.jmp,
                            f.abi,
                            f.JMP_DISP_MAX_LEN,
                            f.reverse_target(nextl.continue_offset)))

    (r
        (not_long)

        .cmp(n_addr,r_tmp1)
        .if_cond(f.test_NE) [f()
            (err)
            .invoke('PyErr_SetString','PyExc_SystemError','BAD_EXCEPTION_MSG')
            .exc_goto_end()
        ]
        .decref(r_tmp1,amount=EXCEPT_VALUES) # all values are None
    )

    r.add_to_stack(-EXCEPT_VALUES)
    r(end)

    if with_cleanup:
        # pop off __exit__ function
        r.del_stack_items(1)

    return r

def omitted(f):
    if not f.last_op_is_jump:
        if f.handler_blocks and f.handler_blocks[-1].type == BLOCK_WITH:
            # CLEANUP_WITH already called pop_handler_block (and pushed the
            # block back on) so we don't need to do it here
            f.handler_blocks.pop()
        else:
            pop_handler_block(f,BLOCK_FINALLY)

_op_END_FINALLY.omitted = omitted

@handler
def _op_NOP(f):
    return []

@handler
def _op_DUP_TOP(f):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .mov(tos,f.r_ret)
        .set_tos()
        .incref()
    )

@handler
def _op_DUP_TOP_TWO(f):
    tos = f.stack.tos()
    return (f()
        .mov(f.stack[not f.stack.tos_in_r_ret],f.r_scratch[0])
        .push_tos()
        .mov(tos,f.r_ret)
        .set_tos()
        .push_stack(f.r_scratch[0])
        .incref(f.r_scratch[0])
        .incref(f.r_ret)
    )

def _unary_op(f,func):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke(func,tos)
        .check_err()
        .pop_stack(f.r_scratch[0])
        .push_stack(f.r_ret)
        .decref(f.r_scratch[0])
    )

@handler
def _op_UNARY_POSITIVE(f):
    return _unary_op(f,'PyNumber_Positive');

@handler
def _op_UNARY_NEGATIVE(f):
    return _unary_op(f,'PyNumber_Negative');

@handler
def _op_UNARY_INVERT(f):
    return _unary_op(f,'PyNumber_Invert');

@handler
def _op_UNARY_NOT(f):
    elif_ = JumpTarget()
    else_ = JumpTarget()
    endif = JumpTarget()

    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_IsTrue',tos)
        .pop_stack(f.r_scratch[0])
        .decref(f.r_scratch[0],preserve_reg=f.r_ret)
        .test(f.r_ret,f.r_ret)
        .jnz(elif_)
            .mov('Py_True',f.r_ret)
            .goto(endif)
        (elif_)
        .jl(else_)
            .mov('Py_False',f.r_ret)
            .goto(endif)
        (else_)
            .exc_goto_end()
        (endif)
        .set_tos()
        .incref()
    )

def _loop_jump(f,fblock,jmp_source,typecode):
    # .unwind_handler() alters StackManager.protected_items
    r = f().push_tos().branch()

    if f.handler_blocks:
        limit = r.stack.offset
        if fblock:
            assert (fblock.stack - EXCEPT_VALUES) >= limit
            limit = fblock.stack - EXCEPT_VALUES
        r.unwind_handler(limit)

    if fblock:
        r.mov('Py_None',f.r_scratch[0])
        r.add_to_stack(EXCEPT_VALUES-1)

        for i in range(1,EXCEPT_VALUES-1):
            r.mov(f.r_scratch[0],r.stack[i])

        r.incref(f.r_scratch[0],EXCEPT_VALUES-1)

        # CPython keeps an array of pre-allocated integer objects between -5
        # and 256. For values in that range, this function never raises an
        # exception, thus no error checking is done here.
        r.invoke('PyLong_FromLong',typecode)
        r.push_stack(f.r_ret)

        r.goto(fblock.target)
    else:
        r(jmp_source)

    return r.end_branch()

@handler
def _op_CONTINUE_LOOP(f,to):
    fblock = None
    for b in reversed(f.blockends):
        if is_finally(b) and not fblock:
            fblock = b
        elif b.type == BLOCK_LOOP:
            if b.continue_offset is None:
                b.continue_offset = to
                b.stack = f.stack.offset
            else:
                if b.continue_offset != to:
                    raise NCCompileError('Loop has multiple "continue" statements with disagreeing targets')
                if b.stack != f.stack.offset:
                    raise NCCompileError('Loop has multiple "continue" statements with disagreeing stack offsets')
            break
    else:
        raise NCCompileError('"continue" op-code found not inside loop')

    return _loop_jump(
        f,
        fblock,
        JumpRSource(f.op.jmp,f.abi,f.JMP_DISP_MAX_LEN,f.reverse_target(to)),
        WHY_CONTINUE)

@handler
def _op_BREAK_LOOP(f):
    fblock = None
    target = None
    for b in reversed(f.blockends):
        if is_finally(b) and not fblock:
            fblock = b
        elif b.type == BLOCK_LOOP:
            b.has_break = True
            target = b.target
            break
    else:
        raise NCCompileError('"break" op-code found not inside loop')

    return _loop_jump(f,fblock,f.goto(target),WHY_BREAK)

@handler
def _op_DELETE_SUBSCR(f):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_DelItem',f.stack[1],tos)
        .check_err(True)
        .del_stack_items(2)
    )

@handler
@hasname
def _op_DELETE_ATTR(f,name):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_SetAttr',tos,address_of(name),0)
        .check_err(True)
        .del_stack_items(1)
    )

@handler
@hasname
def _op_DELETE_GLOBAL(f,name):
    return (f()
        .push_tos()
        .invoke('PyDict_DelItem',f.GLOBALS,address_of(name))
        .if_eax_is_not_zero(f()
            .invoke('format_exc_check_arg',
                'PyExc_NameError',
                'GLOBAL_NAME_ERROR_MSG',
                address_of(name))
            .exc_goto_end()
        )
    )

@handler
def _op_DELETE_FAST(f,arg):
    item = f.Address(arg*f.ptr_size,f.r_scratch[0])
    return (f()
        .push_tos()
        .mov(f.FAST_LOCALS,f.r_scratch[0])
        .if_(signed(item) == 0) [f()
            .invoke('format_exc_check_arg',
                'PyExc_UnboundLocalError',
                'UNBOUNDLOCAL_ERROR_MSG',
                address_of(f.code.co_varnames[arg]))
            .exc_goto_end()
        ]
        .mov(0,item)
    )

@handler
def _op_BINARY_MODULO(f):
    tos = f.stack.tos()
    r = f()

    uaddr = pyinternals.raw_addresses['PyUnicode_Type']
    if not f.fits_imm32(uaddr):
        r.mov(uaddr,f.r_scratch[0])
        uaddr = f.r_scratch[0]

    return (r
        .push_tos()
        .mov(tos,f.r_ret)
        .mov('PyUnicode_Format',f.r_scratch[1])
        .if_(signed(uaddr) != f.type_of(f.r_ret)) [
            join(f().mov('PyNumber_Remainder',f.r_scratch[1]).code)
        ]
        .invoke(f.r_scratch[1],f.stack[1],f.r_ret)
        .check_err()
        .mov(f.stack[1],f.r_scratch[0])
        .mov(f.r_ret,f.stack[1])
        .decref(f.r_scratch[0])
        .del_stack_items(1)
    )

def binary_power(func,f):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke(func,f.stack[1],tos,'Py_None')
        .check_err()
        .mov(f.stack[1],f.r_scratch[1])
        .mov(f.r_ret,f.stack[1])
        .decref(f.r_scratch[1])
        .del_stack_items(1)
    )

@handler
def _op_BINARY_POWER(f):
    return binary_power('PyNumber_Power',f)

@handler
def _op_INPLACE_POWER(f):
    return binary_power('PyNumber_InPlacePower',f)


def hasfree(func):
    return update_wrapper(
        (lambda f,arg: func(f,arg,f.Address(
            (arg + f.code.co_nlocals) * f.ptr_size,
            f.r_scratch[0]))),
        func)

@handler
@hasfree
def _op_LOAD_CLOSURE(f,arg,item):
    return (f()
        .mov(f.FAST_LOCALS,f.r_scratch[0])
        .push_tos()
        .mov(item,f.r_ret)
        .set_tos()
        .incref()
    )

@handler
@hasfree
def _op_LOAD_DEREF(f,arg,item):
    return (f()
        .mov(f.FAST_LOCALS,f.r_scratch[0])
        .push_tos()
        .invoke('PyCell_Get',item)
        .if_eax_is_zero(f()
            .invoke('format_exc_unbound',address_of(f.code),arg)
            .exc_goto_end()
        )
        .set_tos()
    )

@handler
@hasfree
def _op_STORE_DEREF(f,arg,item):
     tos = f.stack.tos()

     return (f()
        .mov(f.FAST_LOCALS,f.r_scratch[0])
        .push_tos()
        .invoke('PyCell_Set',item,tos)
        .del_stack_items(1)
    )

@handler
@hasfree
def _op_DELETE_DEREF(f,arg,item):
    return (f()
        .mov(f.FAST_LOCALS,f.r_scratch[0])
        .push_tos()
        .mov(item,f.r_ret)
        .if_(signed(f.ctype('PyCellObject',f.r_ret).ob_ref) == 0) [f()
            .invoke('format_exc_unbound',address_of(f.code),arg)
            .exc_goto_end()
        ]
        .invoke('PyCell_Set',f.r_ret,0)
    )

@handler
def _op_BUILD_SLICE(f,arg):
    if arg != 3 and arg != 2:
        raise NCCompileError('Invalid argument in BUILD_SLICE instruction')

    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PySlice_New',
            *((f.stack[2],f.stack[1],tos) if arg == 3 else (f.stack[1],tos,0)))
        .check_err()
        .mov(f.stack[arg-1],f.r_scratch[0])
        .mov(f.r_ret,f.stack[arg-1])
        .decref(f.r_scratch[0])
        .del_stack_items(arg-1)
    )

@handler
def _op_BUILD_SET(f,arg):
    err = JumpTarget()
    r = (f()
        .push_tos()
        .invoke('PySet_New',0)
        .set_tos()
    )
    if not arg:
        return r.check_err()

    r.test(f.r_ret,f.r_ret)

    if arg < f.tuning.build_set_loop_threshhold:
        r.jz(err)
        r.mov(f.r_ret,f.r_pres[0])
        for i in range(arg-1):
            r.invoke('PySet_Add',f.r_pres[0],f.stack[i])
            r.test(f.r_ret,f.r_ret)
            r.jnz(err)

        r.invoke('PySet_Add',f.stack[arg-1])
        r.if_eax_is_not_zero(f()
            (err)
            .exc_goto_end()
        )
        r.del_stack_items(arg)
        r.mov(f.r_pres[0],f.r_ret)
    else:
        addr = f.stack[arg]
        addr.index = f.r_pres[0]
        addr.scale = f.ptr_size

        (r
            .mov(f.r_ret,f.r_pres[1])
            .mov(-arg,f.r_pres[0])
            .do [join(f()
                .if_eax_is_not_zero(f
                    .invoke('PySet_Add',f.r_pres[1],addr)
                    .not_(f.r_ret)
                )
                .mov(addr,f.r_scratch[0])
                .decref(f.r_scratch[0],preserve_reg=f.r_ret)
                .add(1,f.r_pres[0])
            .code)] .while_cond(f.test_NZ)
            .add_to_stack(-arg)
            .check_err()
            .mov(f.r_pres[1],f.r_ret)
        )

    return r


def _sequence_add(func,f,arg):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke(func,f.stack[arg],tos)
        .check_err(True)
        .del_stack_items(1)
    )

@handler
def _op_LIST_APPEND(f,arg):
    return _sequence_add('PyList_Append',f,arg)

@handler
def _op_SET_ADD(f,arg):
    return _sequence_add('PySet_Add',f,arg)

def call_function_var_kw(f,arg,flags):
    assert f.stack.tos_in_r_ret == 1 or f.stack.tos_in_r_ret == 0

    na = arg & 0xff
    nk = (arg >> 8) & 0xff
    n = na + 2 * nk + bool(flags & CALL_FLAG_VAR) + bool(flags & CALL_FLAG_KW)

    not_method = JumpTarget()
    endif = JumpTarget()

    r_sp = f.stack.arg_reg(n=1)
    m_obj = f.ctype('PyMethodObject',f.r_pres[0])

    return (f()
        .mov('PyMethod_Type',f.r_scratch[0])
        .mov(f.stack[n - f.stack.tos_in_r_ret],f.r_pres[0])
        .push_tos()
        .cmp(f.r_scratch[0],f.type_of(f.r_pres[0]))
        .jne(not_method)
        .mov(m_obj.im_self,f.r_scratch[0])
        .test(f.r_scratch[0],f.r_scratch[0])
        .jz(not_method)
            .mov(m_obj.im_func,f.r_pres[1])
            .incref(f.r_scratch[0])
            .mov(f.r_scratch[0],f.stack[n])
            .incref(f.r_pres[1])
            .decref(f.r_pres[0])
            .push_arg(na+1,n=3)
            .mov(f.r_pres[1],f.r_pres[0])
            .goto(endif)
        (not_method)
            .push_arg(na,n=3)
        (endif)
        .lea(f.stack[0],r_sp)
        .push_arg(nk,n=4)
        .push_arg(r_sp,n=1)
        .push_arg(f.r_pres[0],n=0)
        .push_arg(flags,n=2)
        .call('ext_do_call')
        .decref(f.r_pres[0],preserve_reg=f.r_ret)
        .add_to_stack(-n-1)
        .check_err()
        .set_tos()
    )

@handler
def _op_CALL_FUNCTION_VAR(f,arg):
    return call_function_var_kw(f,arg,CALL_FLAG_VAR)

@handler
def _op_CALL_FUNCTION_KW(f,arg):
    return call_function_var_kw(f,arg,CALL_FLAG_KW)

@handler
def _op_CALL_FUNCTION_VAR_KW(f,arg):
    return call_function_var_kw(f,arg,CALL_FLAG_VAR|CALL_FLAG_KW)

@handler
def _op_ROT_TWO(f):
    r = f()
    if f.stack.tos_in_r_ret:
        r.mov(f.stack[0],f.r_scratch[0])
        r.mov(f.r_ret,f.stack[0])
        r.mov(f.r_scratch[0],f.r_ret)
    else:
        r.pop_stack(f.r_scratch[0])
        r.mov(f.stack[0],f.r_ret)
        r.set_tos()
        r.mov(f.r_scratch[0],f.stack[0])

    return r

@handler
def _op_ROT_THREE(f):
    r = f()
    if f.stack.tos_in_r_ret:
        r.mov(f.stack[1],f.r_scratch[1])
        r.mov(f.stack[0],f.r_scratch[0])
        r.mov(f.r_ret,f.stack[1])
        r.mov(f.r_scratch[1],f.stack[0])
        r.mov(f.r_scratch[0],f.r_ret)
    else:
        r.pop_stack(f.r_scratch[0])
        r.mov(f.stack[1],f.r_scratch[1])
        r.mov(f.stack[0],f.r_ret)
        r.set_tos()
        r.mov(f.r_scratch[0],f.stack[1])
        r.mov(f.r_scratch[1],f.stack[0])

    return r

def _op_jump_or_pop_if(f,to,jop1,jop2):
    tos = f.stack.tos()
    dont_jump = JumpTarget()

    r = f().push_tos()

    if to > f.byte_offset:
        f.stack.conditional_jump(to)

    return (r
        .invoke('PyObject_IsTrue',tos)
        .test(f.r_ret,f.r_ret)
        (jop1(dont_jump))
        (f.jump_to(jop2,f.JCC_MAX_LEN,to))

        .exc_goto_end()

        (dont_jump)
        .del_stack_items(1)
    )

@handler
def _op_JUMP_IF_FALSE_OR_POP(f,to):
    return _op_jump_or_pop_if(f,to,f.jg,f.jz)

@handler
def _op_JUMP_IF_TRUE_OR_POP(f,to):
    return _op_jump_or_pop_if(f,to,f.jz,f.jg)

def switch_stack(f,reverse):
    """Copy the stack to the frame's (f.r_pres[0]) f_valuestack member

    The order of the items is reversed to match the order Python stores its
    stack values.

    """
    valstack = f.stack.offset - f.stack_prolog
    r = f().mov(f.ctype('PyFrameObject',f.r_pres[0]).f_valuestack,f.r_scratch[0])

    if valstack >= f.tuning.mem_copy_loop_threshhold:
        r.mov(0,f.r_pres[1])
        src = f.stack[valstack - 1]
        src.index = f.r_pres[1]
        src.scale = f.ptr_size

        dest = f.Address(base=f.r_scratch[0])
        if reverse: src,dest = dest,src

        r.do [join(f()
            .mov(src,f.r_scratch[1])
            .sub(1,f.r_pres[1])
            .mov(f.r_scratch[1],dest)
            .add(f.ptr_size,f.r_scratch[0])
        .code)] .while_(signed(f.r_pres[1]) > -valstack)
    else:
        for i in range(valstack):
            src = f.stack[i]
            dest = f.Address((valstack - 1 - i) * f.ptr_size,f.r_scratch[0])
            if reverse: src,dest = dest,src

            r.mov(src,f.r_pres[1])
            r.mov(f.r_pres[1],dest)
        r.add(valstack * f.ptr_size,f.r_scratch[0])

    return r

@handler
def _op_YIELD_VALUE(f):
    # Currently, all the values that are part of the Python stack are copied
    # back and forth between our stack and the frame's f_valuestack member
    # (where the interpreter would store them) when suspending and resuming
    # from a generator. It might be better to have those values stored in
    # f_valuestack in the first place and everything else on the actual stack
    # (ideally we would just store everything in f_valuestack, but there isn't
    # enough room).

    assert f.stack_prolog is not None and f.stack_prolog <= f.stack.offset

    if not f.code.co_flags & CO_GENERATOR:
        raise NCCompileError(
            'The YIELD_VALUE opcode cannot be used if the code is not marked as a generator')

    r = f()
    if not r.use_tos():
        r.pop_stack(f.r_ret)

    r.mov(f.FRAME,f.r_pres[0])
    f_obj = f.ctype('PyFrameObject',f.r_pres[0])

    valstack = f.stack.offset - f.stack_prolog
    if valstack:
        r += switch_stack(f,False)
    else:
        r.mov(f_obj.f_valuestack,f.r_scratch[0])
    r.mov(f.r_scratch[0],f_obj.f_stacktop)

    yield_point = JumpTarget()
    r(YieldMov(f.op,f.yield_start,yield_point,f_obj.f_lasti))

    if f.handler_blocks:
        r.get_threadstate(f.r_scratch[0])
        r(InnerCall(f.op,f.abi,f.util_funcs.swap_exc_state))

    r.mov(0,f.r_pres[0])
    r.goto(f.end)
    r(yield_point)


    #*** the generator will resume here: ***

    # Note: don't use f.TEMP here because if the throw flag was passed to us
    # using a register, it was moved to f.TEMP (ie: f.throw_flag_store will
    # equal f.TEMP).

    # an item gets pushed onto the stack when a generator is resumed
    r.add_to_stack(1)

    r.mov(f.FRAME,f.r_pres[0])
    r += switch_stack(f,True)
    r.movl(-1,f_obj.f_lasti)
    r.mov(0,f_obj.f_stacktop)
    r.cmpl(0,f.throw_flag_store)
    r.if_cond(f.test_NE) [f()
        .exc_goto_end()
    ]
    return r

@handler
def _op_PRINT_EXPR(f):
    tos = f.stack.tos()

    r = f().invoke('_print_expr',tos)

    # _print_expr steals a reference
    if not r.use_tos():
        r.add_to_stack(-1)

    return r.check_err(True)



def join(x):
    # for lists of bytes objects, b''.join is about 30 times faster
    return b''.join(x) if isinstance(x[0],bytes) else reduce(operator.add,x)


class Function:
    def __init__(self,code,padding,address=0,entry_point=None,annotation=None,annot_max=None):
        self.code = code

        # the size, in bytes, of the trailing padding (nop instructions) in
        # "code"
        self.padding = padding

        self.address = address

        # This is basically a modified Python code object. See
        # CodeObjectWithCCode and create_compiled_entry_point in pyinternals.c
        # for information about what this is exactly.
        self.entry_point = entry_point

        self.annotation = annotation or []
        self.annot_max = annot_max

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
    annot_max = None

    for i in range(len(chunks)-1,-1,-1):
        if isinstance(chunks[i],(debug.InnerAnnotation,debug.DescribedAnnot)):
            # since appending to a list is O(1) while prepending is O(n), we
            # add the items backwards and reverse the list afterwards
            debug.append_annot(annots,debug.Annotation(chunks[i],annot_size))
            annot_size = 0
            annot_max = debug.annot_max(annot_max,chunks[i])
            chunks[i] = None

        elif isinstance(chunks[i],JumpTarget):
            chunks[i].displacement = displacement
            chunks[i] = None

        else:
            if isinstance(chunks[i],JumpSource):
                chunks[i] = chunks[i].compile(displacement)
                if not chunks[i]:
                    chunks[i] = None
                    continue
            elif isinstance(chunks[i],DelayedCompile):
                chunks[i].displacement = displacement

            displacement += len(chunks[i])
            annot_size += len(chunks[i])

    assert annot_size == 0 or not annots, "if there are any annotations, there should be one at the start"

    annots.reverse()

    # add padding for alignment
    if CALL_ALIGN_MASK:
        pad_size = aligned_size(displacement) - displacement
        chunks += [op.nop()] * pad_size
        for et in end_targets:
            et.displacement += pad_size

    code = join([(c.compile() if isinstance(c,DelayedCompile) else c) for c in chunks if c is not None])

    for et in end_targets:
        et.displacement += displacement

    return Function(code,pad_size,annotation=annots,annot_max=annot_max)


def simple_frame(op,abi,tuning,func_name=None):
    local_stack_size = aligned_size((MAX_ARGS + 1) * abi.ptr_size)
    f = Frame(op,abi,tuning,local_stack_size)
    r = f()
    if func_name: r.comment(func_name + ' function:')
    r.add_to_stack(1,debug.RETURN_ADDRESS)
    return f, r

def local_name_func(op,abi,tuning):
    # TODO: have just one copy shared by all compiled modules

    f,r = simple_frame(op,abi,tuning,'local_name')

    else1 = JumpTarget()
    endif1 = JumpTarget()
    else2 = JumpTarget()
    ret = JumpTarget()
    inc_ret = JumpTarget()

    d_addr = pyinternals.raw_addresses['PyDict_Type']
    fits = f.fits_imm32(d_addr)

    if not fits:
        r.mov(d_addr,f.r_pres[1])
        d_addr = f.r_pres[1]

    return (r
        .reserve_stack()
        .mov(f.LOCALS,f.r_ret)
        .if_eax_is_zero(f()
            .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_LOAD_MSG',f.r_pres[0])
            .goto(ret)
        )

        .cmp(d_addr,f.type_of(f.r_ret))
        .je(else1)
            .invoke('PyObject_GetItem',f.r_ret,f.r_pres[0])
            .test(f.r_ret,f.r_ret)
            .jnz(ret)

            .invoke('PyErr_ExceptionMatches','PyExc_KeyError')
            .test(f.r_ret,f.r_ret)
            .jz(ret)
            .call('PyErr_Clear')

            .goto(endif1)
        (else1)
            .invoke('PyDict_GetItem',f.r_ret,f.r_pres[0])
            .test(f.r_ret,f.r_ret)
            .jnz(inc_ret)
        (endif1)

        .invoke('PyDict_GetItem',f.GLOBALS,f.r_pres[0])
        .if_eax_is_zero(f()
            .mov(f.BUILTINS,f.r_ret)
            .cmp(d_addr,f.type_of(f.r_ret))
            .je(else2)
                .invoke('PyObject_GetItem',f.r_ret,f.r_pres[0])
                .test(f.r_ret,f.r_ret)
                .jnz(ret)

                .invoke('PyErr_ExceptionMatches','PyExc_KeyError')
                .test(f.r_ret,f.r_ret)
                .jz(ret)
                .call('PyErr_Clear')
                .goto(ret)
            (else2)
                .invoke('PyDict_GetItem',f.r_ret,f.r_pres[0])
                .test(f.r_ret,f.r_ret)
                .jnz(inc_ret)
                .invoke('format_exc_check_arg',
                    'PyExc_NameError',
                    'NAME_ERROR_MSG',
                    f.r_pres[0])
                .goto(ret)
        )

        (inc_ret)
        .incref(f.r_ret)
        (ret)
        .release_stack()
        .ret()
    )

def global_name_func(op,abi,tuning):
    # TODO: have just one copy shared by all compiled modules

    # r_pres[0] is expected to have the address of the name to load

    f,r = simple_frame(op,abi,tuning,'global_name')

    not_dict = JumpTarget()
    name_err = JumpTarget()
    ret = JumpTarget()

    d_addr = pyinternals.raw_addresses['PyDict_Type']
    fits = f.fits_imm32(d_addr)

    if not fits:
        r.mov(d_addr,f.r_scratch[0])
        d_addr = f.r_scratch[0]

    return (r
        .reserve_stack()
        .mov(f.GLOBALS,f.r_scratch[1])
        .mov(f.BUILTINS,f.r_pres[1])
        .cmp(d_addr,f.type_of(f.r_scratch[1]))
        .jne(not_dict)
        .cmp(d_addr,f.type_of(f.r_pres[1]))
        .jne(not_dict)
            .invoke('_PyDict_LoadGlobal',f.r_scratch[1],f.r_pres[1],f.r_pres[0])
            .if_eax_is_zero(f()
                .call('PyErr_Occurred')
                .test(f.r_ret,f.r_ret)
                .jz(name_err)
                .mov(0,f.r_ret)
                .goto(ret)
            )
            .incref()
            .goto(ret)
        (not_dict)
            .invoke('PyObject_GetItem',f.r_scratch[1],f.r_pres[0])
            .if_eax_is_zero(f()
                .invoke('PyObject_GetItem',f.r_pres[1],f.r_pres[0])
                .if_eax_is_zero(f()
                    .invoke('PyErr_ExceptionMatches','PyExc_KeyError')
                    .if_eax_is_not_zero(f()
                        (name_err)
                        .invoke('format_exc_check_arg',
                            'PyExc_NameError',
                            'GLOBAL_NAME_ERROR_MSG',
                            f.r_pres[0])
                        .mov(0,f.r_ret)
                    )
                )
            )
        (ret)
        .release_stack()
        .ret())

def prepare_exc_handler_func_head(op,abi,tuning):
    # TODO: have just one copy shared by all compiled modules

    # r_scratch[0] is expected to have the number of items that need to be
    # popped from the stack

    # r_pres[0] is expected to be what the address of the top of the stack will
    # be once the items have been popped

    # This function is broken into two parts: prepare_exc_handler_func_head and
    # prepare_exc_handler_func_tail. prepare_exc_handler_func_tail is made into
    # a seperate part so reraise_exc_handler_func can use tail-call
    # optimization with this function.

    f,r = simple_frame(op,abi,tuning,'exc_handler')

    r_tmp1 = f.stack.arg_reg(f.r_scratch[0],0)
    r_tmp2 = f.stack.arg_reg(f.r_scratch[1],1)
    tstate = f.ctype('PyThreadState',f.r_pres[1])
    return (r
        .reserve_stack()
        .push_stack(f.r_scratch[0])
        .get_threadstate(f.r_pres[1])
        .invoke('PyTraceBack_Here',f.FRAME)
        .mov(tstate.c_tracefunc,r_tmp1)
        .mov(tstate.c_traceobj,r_tmp2)
        .if_(r_tmp1) [
            join(f.invoke('call_exc_trace',r_tmp1,r_tmp2))
        ]
        .pop_stack(f.r_scratch[0]))

def prepare_exc_handler_func_tail(op,abi,tuning):
    # MUST follow prepare_exc_handler_func_head

    # Normally, a finally block will have 1 to 6 items on the stack for it to
    # pop, but our stack management uses absolute offsets instead of push and
    # pop instructions so we don't have this flexibility, so we pad the stack
    # with Nones if needed (here, it's not).

    # if this value changes, the code here (and the comment above) will need
    # updating
    assert EXCEPT_VALUES == 6

    f,r = simple_frame(op,abi,tuning)

    exc = f.Address(-abi.ptr_size*6,f.r_pres[0])
    val = f.Address(-abi.ptr_size*5,f.r_pres[0])
    tb = f.Address(-abi.ptr_size*4,f.r_pres[0])
    exc_old = f.Address(-abi.ptr_size*3,f.r_pres[0])
    val_old = f.Address(-abi.ptr_size*2,f.r_pres[0])
    tb_old = f.Address(-abi.ptr_size,f.r_pres[0])

    tstate = f.ctype('PyThreadState',f.r_pres[1])

    (r
        .reserve_stack(False)
        .clean_stack(addr=f.Address(base=f.r_pres[0]),index=f.r_scratch[0])

        .get_threadstate(f.r_pres[1])
        .mov(tstate.exc_traceback,f.r_scratch[0])
        .mov(tstate.exc_value,f.r_scratch[1])
        .mov(tstate.exc_type,f.r_ret)
        .mov(f.r_scratch[0],tb_old)
        .mov(f.r_scratch[1],val_old)
        .if_eax_is_zero(f()
            .mov('Py_None',f.r_ret)
            .incref()
        )
        .mov(f.r_ret,exc_old))

    load_args = []
    for n,item in enumerate((exc,val,tb)):
        dest = f.stack.arg_dest(n)
        reg = (f.r_scratch[0],f.r_scratch[1],f.r_ret)[n]
        if isinstance(dest,f.Address):
            load_args.insert(n,f.op.lea(item,reg))
            load_args.append(f.op.mov(reg,dest))
        else:
            load_args.append(f.op.lea(item,dest))
    load_args = join(load_args)

    return (r
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
        .if_(signed(f.r_scratch[1]) == 0) [f()
            .mov('Py_None',f.r_scratch[1])
        ]
        .mov(f.r_ret,tstate.exc_type)
        .mov(f.r_scratch[0],tstate.exc_value)
        .incref(f.r_scratch[1])
        .mov(f.r_scratch[1],tstate.exc_traceback)
        .mov(0,f.r_ret)
        .release_stack()
        .ret())

def reraise_exc_handler_func(op,abi,tuning,prepare_exc_handler_tail):
    # TODO: have just one copy shared by all compiled modules

    # r_scratch[0] is expected to have the number of items that need to be
    # popped from the stack

    # r_pres[0] is expected to be what the address of the top of the stack will
    # be once the items have been popped

    f,r = simple_frame(op,abi,tuning,'reraise_exc_handler')

    return (r
        .reserve_stack()
        (InnerCall(op,abi,prepare_exc_handler_tail,True)))

def unwind_exc_func_head(op,abi,tuning):
    # TODO: have just one copy shared by all compiled modules

    # r_scratch[0] is expected to have the number of items that need to be
    # popped from the stack

    # r_pres[0] is expected to be what the address of the top of the stack will
    # be once the items have been popped

    # This function must be called if an exception handler exits before
    # reaching POP_EXCEPT.

    # This function is broken into two parts: unwind_exc_func_head and
    # unwind_exc_func_tail. unwind_exc_func_tail is made into a seperate part
    # so unwind_finally_func can use tail-call optimization with this function.

    f,r = simple_frame(op,abi,tuning,'unwind_exc')

    return r.reserve_stack()

def unwind_exc_func_tail(op,abi,tuning):
    # MUST follow unwind_exc_func_head

    f,r = simple_frame(op,abi,tuning)

    exc = f.Address(0,f.r_pres[0])
    val = f.Address(abi.ptr_size,f.r_pres[0])
    tb = f.Address(abi.ptr_size*2,f.r_pres[0])
    tstate = f.ctype('PyThreadState',f.r_pres[1])

    (r
        .reserve_stack(False)
        .clean_stack(addr=f.Address(base=f.r_pres[0]),index=f.r_scratch[0])
        .get_threadstate(f.r_pres[1])
        .mov(exc,f.r_scratch[0])
        .mov(val,f.r_scratch[1])
        .mov(tb,f.r_pres[0]))

    del exc, val, tb # no longer valid
    (r
        # swap the values
        .xor(f.r_scratch[0],tstate.exc_type)
        .xor(f.r_scratch[1],tstate.exc_value)
        .xor(f.r_pres[0],tstate.exc_traceback)
        .xor(tstate.exc_type,f.r_scratch[0])
        .xor(tstate.exc_value,f.r_scratch[1])
        .xor(tstate.exc_traceback,f.r_pres[0])
        .xor(f.r_scratch[0],tstate.exc_type)
        .xor(f.r_scratch[1],tstate.exc_value)
        .xor(f.r_pres[0],tstate.exc_traceback))

    del tstate
    return (r
        .decref(f.r_scratch[0],preserve_reg=f.r_scratch[1]) # set to Py_None if NULL by prepare_exc_handler_func
        .if_(f.r_scratch[1])[f.decref(f.r_scratch[1])]
        .if_(f.r_pres[0])[f.decref(f.r_pres[0])]
        .release_stack()
        .ret())

def unwind_finally_func(op,abi,tuning,unwind_exc_tail):
    # this function uses an ad hoc calling convention and is only callable from
    # the code generated by this module

    # TODO: have just one copy shared by all compiled modules

    # r_scratch[0] is expected to have the number of items that need to be
    # popped from the stack

    # r_pres[0] is expected to be what the address of the top of the stack will
    # be once the items have been popped

    # This function must be called if a finally block exits before reaching
    # END_FINALLY.

    # The code here assumes that only exceptions use more than two out of
    # EXCEPT_VALUES spaces, and the rest are set to Py_None. If EXCEPT_VALUES
    # changes, the assumption should be rechecked.
    assert EXCEPT_VALUES == 6

    f,r = simple_frame(op,abi,tuning,'unwind_finally')
    not_exc = JumpTarget()

    return (r
        .reserve_stack()
        .clean_stack(addr=f.Address(base=f.r_pres[0]),index=f.r_scratch[0])
        .mov(f.Address(base=f.r_pres[0]),f.r_scratch[1])
        .mov(f.type_of(f.r_scratch[1]),f.r_ret)
        .testl(TPFLAGS_TYPE_SUBCLASS,f.type_flags_of(f.r_ret))
        .jz(not_exc)
        .testl(TPFLAGS_BASE_EXC_SUBCLASS,f.type_flags_of(f.r_scratch[1]))
        .jz(not_exc)
        .add(3*abi.ptr_size,f.r_pres[0])
        .sub(3,f.r_scratch[0])
        (InnerCall(op,abi,unwind_exc_tail,True))
        (not_exc)
        .mov(f.Address(base=f.r_pres[0]),f.r_ret)
        .decref(f.r_ret)
        .mov(f.Address(abi.ptr_size,f.r_pres[0]),f.r_ret)
        .decref(f.r_ret)
        .mov(f.Address(abi.ptr_size*2,f.r_pres[0]),f.r_ret)
        .decref(f.r_ret,amount=EXCEPT_VALUES-2)
        .release_stack()
        .ret())

def swap_exc_state_func(op,abi,tuning):
    # this function uses an ad hoc calling convention and is only callable from
    # the code generated by this module

    # TODO: have just one copy shared by all compiled modules

    # r_pres[0] is expected to be the frame object
    # r_scratch[0] is expected to be the thread state object

    f_obj = CType(abi,'PyFrameObject',abi.r_pres[0])
    tstate = CType(abi,'PyThreadState',abi.r_scratch[0])
    r = []
    for m in ['exc_type','exc_value','exc_traceback']:
        a = f_obj.__getattr__('f_'+m)
        b = tstate.__getattr__(m)

        r.append(op.mov(a,abi.r_scratch[1]))
        r.append(op.mov(b,abi.r_pres[1]))
        r.append(op.mov(abi.r_scratch[1],b))
        r.append(op.mov(abi.r_pres[1],a))
    r.append(op.ret())

    return r

def compile_eval(code,op,abi,tuning,util_funcs,entry_points):
    """Generate a function equivalent to PyEval_EvalFrameEx called with f.code"""

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
    #     - miscellaneous temp value

    # the first 2 will already be on the stack by the time %esp is adjusted
    pre_stack = 2
    stack_first = 10

    if pyinternals.REF_DEBUG:
        # a place to store r_ret, r_scratch[0] and r_scratch[1] when increasing
        # reference counts (which calls a function when ref_debug is True)
        stack_first += DEBUG_TEMPS
    elif pyinternals.COUNT_ALLOCS:
        # when COUNT_ALLOCS is True and REF_DEBUG is not, an extra space is
        # needed to save a temporary value
        stack_first += 1

    local_stack_size = aligned_size(
        (code.co_stacksize +
         max(MAX_ARGS-len(abi.r_arg),0) +
         stack_first) * abi.ptr_size + abi.shadow)

    assert local_stack_size % abi.ptr_size == 0

    f = Frame(
        op,
        abi,
        tuning,
        local_stack_size,
        code,
        util_funcs,
        entry_points)

    f.throw_flag_store = f.stack.func_arg(1)
    move_throw_flag = (f.code.co_flags & CO_GENERATOR and
                       isinstance(f.throw_flag_store,f.Register))

    if move_throw_flag:
        # we use f.TEMP to use the throw flag and temporarily use another
        # address (the address where the first Python stack value will go) as
        # our temporary store
        f.throw_flag_store = f.TEMP
        f.TEMP = f.stack[-stack_first-1]

    # at the epilogue, GLOBALS is no longer used and we use its space as a
    # temporary store for the return value
    ret_temp_store = f.GLOBALS

    fast_end = JumpTarget()

    opcodes = (f()
        .comment('prologue')
        .add_to_stack(1,debug.RETURN_ADDRESS)
        .save_reg(abi.r_bp)
        .mov(abi.r_sp,abi.r_bp)
        .save_reg(f.r_pres[0])
        .save_reg(f.r_pres[1])
        .reserve_stack())

    f_obj = f.ctype('PyFrameObject',f.r_pres[0])
    tstate = f.ctype('PyThreadState',f.r_scratch[0])

    if move_throw_flag:
        opcodes.mov(f.dword(f.stack.func_arg(1)),f.throw_flag_store)

    argreg = f.stack.arg_reg(n=0)
    (opcodes
        .mov(f.stack.func_arg(0),f.r_pres[0])
        .call('_EnterRecursiveCall')
        .check_err(True)

        .get_threadstate(f.r_scratch[0])

        .mov(f_obj.f_globals,f.r_scratch[1])
        .mov(f_obj.f_builtins,f.r_ret)

        .push_stack(f.r_pres[0],debug.PushVariable('f'))
        .push_stack(f.r_scratch[1])
        .push_stack(f.r_ret)

        .mov(f_obj.f_locals,f.r_scratch[1])
        .lea(f_obj.f_localsplus,f.r_ret)

        .push_stack(f.r_scratch[1])
        .push_stack(f.r_ret)

        .mov(f.r_pres[0],tstate.frame)
    )

    if f.code.co_flags & CO_GENERATOR:
        not_in_handler = JumpTarget()
        endif = JumpTarget()

        naddr = pyinternals.raw_addresses['Py_None']
        if not f.fits_imm32(naddr):
            opcodes.mov(naddr,f.r_ret)
            naddr = f.r_ret

        jdispatch = f()
        rip = getattr(f.abi.ops,'rip',None)
        if rip:
            jdispatch.lea(f.Address(0,rip),f.r_ret)(f.yield_start)
        else:
            jdispatch.call(f.Displacement(0))(f.yield_start).pop(f.r_ret)

        jdispatch.add(f.r_ret,f.r_scratch[0]).jmp(f.r_scratch[0])

        (opcodes
            .mov(f_obj.f_exc_type,f.r_scratch[1])
            .test(f.r_scratch[1],f.r_scratch[1])
            .jz(not_in_handler)
            .cmp(naddr,f.r_scratch[1])
            .je(not_in_handler)
                (InnerCall(f.op,f.abi,f.util_funcs.swap_exc_state))
                .goto(endif)
            (not_in_handler)
                .mov(tstate.exc_type,f.r_ret)
                .mov(tstate.exc_value,f.r_pres[1])
                .mov(tstate.exc_traceback,f.r_scratch[1])

                .if_(f.r_ret) [ join(f.incref(f.r_ret)) ]
                .if_(f.r_pres[1]) [ join(f.incref(f.r_pres[1])) ]
                .if_(f.r_scratch[1]) [ join(f.incref(f.r_scratch[1])) ]

                .mov(f_obj.f_exc_type,f.r_scratch[0])
                .mov(f.r_ret,f_obj.f_exc_type)
                .mov(f_obj.f_exc_value,f.r_ret)
                .mov(f.r_pres[1],f_obj.f_exc_value)
                .mov(f_obj.f_exc_traceback,f.r_pres[1])
                .mov(f.r_scratch[1],f_obj.f_exc_traceback)

                .if_(f.r_scratch[0]) [ join(f.decref(f.r_scratch[0],preserve_reg=f.r_ret)) ]
                .if_(f.r_ret) [ join(f.decref(f.r_ret)) ]
                .if_(f.r_pres[1]) [ join(f.decref(f.r_pres[1])) ]
            (endif)
            .mov(f_obj.f_lasti,f.dword(f.r_scratch[0]))
            .if_(signed(f.dword(f.r_scratch[0])) != -1) [ jdispatch ]
            .cmpl(0,f.throw_flag_store)
            .if_cond(f.test_NE) [f()
                .mov(0,ret_temp_store)
                .goto(fast_end)
            ]
        )

    del f_obj
    del tstate

    if move_throw_flag:
        # use the original address for TEMP again
        f.TEMP = f.throw_flag_store

    opcodes.set_stack(stack_first,debug.PYTHON_STACK_START)
    f.stack_prolog = stack_first
    f.stack.protected_items.append(stack_first)

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


    if f.stack.offset is not None and f.stack.offset != f.stack_prolog:
        raise NCCompileError(
            'stack.offset should be {0}, but is {1}'
            .format(f.stack_prolog,f.stack.offset))
    opcodes.set_stack(f.stack_prolog)
    f.stack.protected_items.pop()
    assert len(f.stack.protected_items) == 1

    if f.blockends:
        raise NCCompileError('there is an unclosed block statement')

    tstate = f.ctype('PyThreadState',f.r_scratch[0])
    f_obj = f.ctype('PyFrameObject',f.r_scratch[1])

    # call Py_DECREF on anything left on the stack and return %eax
    (opcodes
        (f.end)
        .comment('epilogue')
        .mov(f.r_ret,ret_temp_store)
        .clean_stack()
        (fast_end)
        .call('_LeaveRecursiveCall')
        .mov(ret_temp_store,f.r_ret)
        .get_threadstate(f.r_scratch[0])
        .mov(f.FRAME,f.r_scratch[1])
        .release_stack()
        .restore_reg(f.r_pres[1])
        .restore_reg(f.r_pres[0])
        .mov(f_obj.f_back,f.r_scratch[1])
        .restore_reg(abi.r_bp)
        .mov(f.r_scratch[1],tstate.frame)
        .ret())

    assert f.stack.offset == 1

    return opcodes


def compile_raw(_code,abi,tuning=Tuning(),op=None):
    assert len(abi.r_scratch) >= 2 and len(abi.r_pres) >= 2

    if isinstance(_code,types.CodeType):
        _code = (_code,)

    ufuncs = UtilityFunctions()
    prepare_exc_handler_tail = JumpTarget()
    unwind_exc_tail = JumpTarget()

    entry_points = collections.OrderedDict()
    if not op: op = abi.ops

    ceval = partial(compile_eval,
        op=op,
        abi=abi,
        tuning=tuning,
        util_funcs=ufuncs,
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

    def add_util_func(target,body,*extra_args):
        target.displacement = 0
        functions.insert(
            0,
            resolve_jumps(op,destitch(body(op,abi,tuning,*extra_args)),end_targets))
        end_targets.append(target)

    if ufuncs.swap_exc_state.used:
        add_util_func(ufuncs.swap_exc_state,swap_exc_state_func)

    if ufuncs.unwind_exc.used or ufuncs.unwind_finally.used:
        add_util_func(unwind_exc_tail,unwind_exc_func_tail)

    if ufuncs.unwind_exc.used:
        add_util_func(ufuncs.unwind_exc,unwind_exc_func_head)

    if ufuncs.unwind_finally.used:
        add_util_func(ufuncs.unwind_finally,unwind_finally_func,unwind_exc_tail)

    if ufuncs.prepare_exc_handler.used or ufuncs.reraise_exc_handler.used:
        add_util_func(prepare_exc_handler_tail,prepare_exc_handler_func_tail)

    if ufuncs.prepare_exc_handler.used:
        add_util_func(ufuncs.prepare_exc_handler,prepare_exc_handler_func_head)

    if ufuncs.reraise_exc_handler.used:
        add_util_func(
            ufuncs.reraise_exc_handler,
            reraise_exc_handler_func,
            prepare_exc_handler_tail)

    if ufuncs.global_name.used:
        add_util_func(ufuncs.global_name,global_name_func)

    if ufuncs.local_name.used:
        add_util_func(ufuncs.local_name,local_name_func)


    entry_points = list(entry_points.values())
    for ep,func in reversed(entry_points):
        functions.insert(0,resolve_jumps(op,func.code,end_targets))

    offset = 0
    for epf,func in itertools.zip_longest(entry_points,functions):
        if epf:
            pyinternals.cep_set_offset(epf[0],offset)
            func.entry_point = epf[0]
        func.address = offset
        offset += len(func)

    return CompilationUnit(functions),[ep for ep,func in entry_points]



