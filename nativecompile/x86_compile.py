
import sys
import dis
import weakref
import operator
import types
import itertools
from functools import partial, reduce

from . import x86_ops as ops
from . import pyinternals


PRINT_STACK_OFFSET = False


STACK_ITEM_SIZE = 4
CALL_ALIGN_MASK = 0xf
MAX_ARGS = 4
PRE_STACK = 4
DEBUG_TEMPS = 3


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
    def __init__(self,op,local_mem_size):
        assert local_mem_size % STACK_ITEM_SIZE == 0

        self.op = op
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
        if (self.__offset + self.args) * STACK_ITEM_SIZE > self.local_mem_size:
            raise NCSystemError("Not enough stack space was reserved. This is either a bug or the code object being compiled has an incorrect value for co_stack_size.")

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

    def push_arg(self,x,tempreg = ops.ecx,n = None):
        if n is None:
            n = self.args
        else:
            self.args = max(self.args,n)

        dest = ops.Address(n * STACK_ITEM_SIZE,ops.esp)
        if x == dest:
            # this case will make sense once calling conventions that pass
            # arguments in registers are implemented
            r = []
        elif isinstance(x,ops.Address):
            r = [self.op.mov(x,tempreg),self.op.mov(tempreg,dest)]
        elif isinstance(x,int):
            r = [self.op.movl(x,dest)]
        else:
            r = [self.op.mov(x,dest)]

        self.args += 1
        self.check_stack_space()
        return r
    
    def push_tos(self,set_again = False):
        """%eax is needed right now so if the TOS item hasn't been pushed onto 
        the stack, do it now."""
        r = [self.push_stack(ops.eax)] if self.tos_in_eax else []
        self.tos_in_eax = set_again
        return r
    
    def use_tos(self,set_again = False):
        r = self.tos_in_eax
        self.tos_in_eax = set_again
        return r

    def tos(self):
        return ops.eax if self.tos_in_eax else self[0]

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
        offset = self.local_mem_size - (self.offset - n) * STACK_ITEM_SIZE
        assert offset >= 0
        return ops.Address(offset,ops.esp)

    def func_arg(self,n):
        """Return the address or register where argument n of the current
        function is stored.

        This should not be confused with push_arg and arg_reg, which operate on
        the arguments of the function about to be called.

        """
        return ops.Address(n * STACK_ITEM_SIZE + self.local_mem_size,ops.esp)

    def call(self,func):
        self.args = 0
        if isinstance(func,str):
            return [
                self.op.mov(pyinternals.raw_addresses[func],ops.eax),
                self.op.call(ops.eax)]

        return [self.op.call(func)]

    def arg_reg(self,tempreg=ops.ecx,n=None):
        """If the nth argument is stored in a register, return that register.
        Otherwise, return tempreg.

        Since push_arg will emit nothing when the source and destination are the
        same, this can be used to eliminate an extra push with opcodes that
        require a register destination. If the given function argument is stored
        in a register, arg_reg will return that register and when passed to
        push_arg, push_arg will emit nothing. If not, tempreg will be returned
        and push_arg will emit the appropriate MOV instruction.

        """
        # until other calling conventions are implemented, the return value will
        # always be tempreg
        return tempreg



class JumpTarget:
    used = False
    displacement = None


class JumpSource:
    def __init__(self,op,target):
        self.op = op
        self.target = target
        target.used = True
    
    def compile(self,displacement):
        dis = displacement - self.target.displacement
        return self.op(ops.Displacement(dis)) if dis else b'' #optimize away useless jumps


class DelayedCompile:
    pass


class InnerCall(DelayedCompile):
    """A function call with a relative target

    This is just like JumpSource, except the target is a different function and
    the exact offset depends on how much padding is needed between this source's
    function and the target function, which cannot be determined until the
    length of the entire source function is determined.

    """
    def __init__(self,opset,target):
        self.opset = opset
        self.target = target
        target.used = True
        self.displacement = None

    def compile(self):
        return self.opset.call(ops.Displacement(self.displacement - self.target.displacement))

    def __len__(self):
        return ops.CALL_DISP_LEN


class JumpRSource(DelayedCompile):
    def __init__(self,op,size,target):
        self.op = op
        self.size = size
        self.target = target
        target.used = True
        self.displacement = None

    def compile(self):
        c = self.op(ops.Displacement(self.displacement - self.target.displacement,True))
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
    operations are only interleaved when pyinternals.ref_debug is false.
    Otherwise the operations are arranged sequentially.  This is to allow the
    use of the code from Frame.incref, which cannot be interleaved when
    ref_debug is true.

    """
    items = []
    pending = []

    try:
        if pyinternals.ref_debug:
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
        if isinstance(f.op,ops.Assembly):
            r.comment(opname)
        if f.forward_targets and f.forward_targets[0][0] <= f.byte_offset:
            pos,t,pop = f.forward_targets.pop(0)
            assert pos == f.byte_offset
            r.push_tos()(t)
            if pop:
                r.pop_stack(ops.edx).decref(ops.edx)

        f.stack.current_pos(f.byte_offset)

        if PRINT_STACK_OFFSET:
            print('stack: {}  stack items: {}  opcode: {}'.format(
                f.stack.offset,
                f.stack.offset//STACK_ITEM_SIZE + f.stack.tos_in_eax,
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


a = itertools.count(STACK_ITEM_SIZE * (1 - PRE_STACK),-STACK_ITEM_SIZE)
GLOBALS = ops.Address(next(a),ops.ebp)
BUILTINS = ops.Address(next(a),ops.ebp)
LOCALS = ops.Address(next(a),ops.ebp)
FAST_LOCALS = ops.Address(next(a),ops.ebp)

# these are only used when pyinternals.ref_debug is True
TEMP_EAX = ops.Address(next(a),ops.ebp)
TEMP_ECX = ops.Address(next(a),ops.ebp)
TEMP_EDX = ops.Address(next(a),ops.ebp)
del a


def raw_addr_if_str(x):
    return pyinternals.raw_addresses[x] if isinstance(x,str) else x

class Frame:
    def __init__(self,op,tuning,local_mem_size,code=None,local_name=None,entry_points=None):
        self.code = code
        self.op = op
        self.tuning = tuning
        self.stack = StackManager(op,local_mem_size)
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
    
    def check_err(self,inverted=False):
        if inverted:
            return self.if_eax_is_not_zero(
                [self.op.xor(ops.eax,ops.eax)] + 
                self.goto_end())

        return self.if_eax_is_zero(self.goto_end())

    def invoke(self,func,*args):
        return reduce(operator.concat,(self.stack.push_arg(raw_addr_if_str(a)) for a in args)) + self.stack.call(func)

    def _if_eax_is(self,test,opcodes):
        if isinstance(opcodes,(bytes,ops.AsmSequence)):
            return [
                self.op.test(ops.eax,ops.eax),
                self.op.jcc(~test,ops.Displacement(len(opcodes))),
                opcodes]
                
        after = JumpTarget()
        return [
            self.op.test(ops.eax,ops.eax),
            JumpSource(partial(self.op.jcc,~test),after)
        ] + opcodes + [
            after
        ]

    def if_eax_is_zero(self,opcodes): return self._if_eax_is(ops.test_Z,opcodes)
    def if_eax_is_not_zero(self,opcodes): return self._if_eax_is(ops.test_NZ,opcodes)

    def goto(self,target):
        return JumpSource(self.op.jmp,target)

    def goto_end(self):
        return [self.op.lea(self.stack[0],ops.ebx),JumpSource(self.op.jmp,self._end)]

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

    def incref(self,reg = ops.eax):
        if pyinternals.ref_debug:
            # the registers that would otherwise be undisturbed, must be preserved
            return ([
                self.op.mov(ops.eax,TEMP_EAX),
                self.op.mov(ops.ecx,TEMP_ECX),
                self.op.mov(ops.edx,TEMP_EDX)
            ] + self.invoke('Py_IncRef',reg) + [
                self.op.mov(TEMP_EDX,ops.edx),
                self.op.mov(TEMP_ECX,ops.ecx),
                self.op.mov(TEMP_EAX,ops.eax)])

        return [self.incl_or_addl(ops.Address(pyinternals.refcnt_offset,reg))]

    def decref(self,reg = ops.eax,preserve_eax = False):
        if pyinternals.ref_debug:
            inv = self.invoke('Py_DecRef',reg)
            return [self.op.mov(ops.eax,TEMP_EAX)] + inv + [self.op.mov(TEMP_EAX,ops.eax)] if preserve_eax else inv

        assert reg.reg != ops.ecx.reg
        
        mid = []
        
        if preserve_eax:
            mid.append(self.op.mov(ops.eax,self.stack[-1]))

        mid += [
            self.op.mov(ops.Address(pyinternals.type_offset,reg),ops.esi),
        ]

        mid += self.invoke(
            ops.Address(pyinternals.type_dealloc_offset,ops.esi),
            reg)

        if pyinternals.count_allocs:
            mid += self.invoke('inc_count',ops.esi)
        
        if preserve_eax:
            mid.append(self.op.mov(self.stack[-1],ops.eax))
        
        mid = join(mid)
        
        return [
            self.decl_or_subl(ops.Address(pyinternals.refcnt_offset,reg)),
            self.op.jnz(ops.Displacement(len(mid))),
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
        return JumpSource(op,self.forward_target(to)) if to > self.byte_offset else JumpRSource(op,max_size,self.reverse_target(to))

    def __call__(self):
        return Stitch(self)


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


def destitch(x):
    return x.code if isinstance(x,Stitch) else x

class Stitch:
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

    def inc_or_add(self,x):
        self.code.append(self.f.inc_or_add(x))
        return self

    def incb_or_addb(self,x):
        self.code.append(self.f.incb_or_addb(x))
        return self

    def incl_or_addl(self,x):
        self.code.append(self.f.incl_or_addl(x))
        return self

    def dec_or_sub(self,x):
        self.code.append(self.f.dec_or_sub(x))
        return self

    def decb_or_subb(self,x):
        self.code.append(self.f.decb_or_subb(x))
        return self

    def decl_or_subl(self,x):
        self.code.append(self.f.decl_or_subl(x))
        return self

    def call(self,x):
        self.code += self.f.stack.call(x)
        return self

    def if_eax_is_zero(self,opcodes):
        self.code += self.f.if_eax_is_zero(destitch(opcodes))
        return self

    def if_eax_is_not_zero(self,opcodes):
        self.code += self.f.if_eax_is_not_zero(destitch(opcodes))
        return self

    @arg1_as_subscr
    def if_cond(self,test,opcodes):
        if isinstance(opcodes,(bytes,ops.AsmSequence)):
            self.code += [
                self.f.op.jcc(~test,ops.Displacement(len(opcodes))),
                opcodes]
        else:
            after = JumpTarget()
            self.code += [
                JumpSource(partial(self.f.op.jcc,~test),after)
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
        def inner(*args):
            self.code.append(func(*map(raw_addr_if_str,args)))
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
        .mov(f.stack[1],ops.edx)
        .mov(ops.eax,f.stack[1])
        .decref(ops.edx)
        .pop_stack(ops.edx)
        .decref(ops.edx)
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
        r.insert(0,f.stack.pop_stack(ops.eax))
    return r

@hasname
def _op_LOAD_NAME(f,name):
    return (f()
        .push_tos(True)
        .mov(address_of(name),ops.ebx)
        (InnerCall(f.op,f.local_name))
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
        .mov(LOCALS,ops.eax)
        .if_eax_is_zero(f()
            .clear_args()
            .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_STORE_MSG')
            .goto_end()
        )
        .mov('PyObject_SetItem',ops.ecx)
        .cmpl('PyDict_Type',ops.Address(pyinternals.type_offset,ops.eax))
        .push_arg(ops.eax,n=0)
        .if_cond[ops.test_E](
            f.op.mov(pyinternals.raw_addresses['PyDict_SetItem'],ops.ecx)
        )
        .call(ops.ecx)
        .check_err(True)
        .pop_stack(ops.eax)
        .decref()
    )

@hasname
def _op_DELETE_NAME(f,name):
    return (f()
        .push_tos()
        .mov(LOCALS,ops.eax)
        .if_eax_is_zero(f()
            .invoke('PyErr_Format',
                'PyExc_SystemError',
                'NO_LOCALS_DELETE_MSG',
                address_of(name))
            .goto_end()
        )
        .invoke('PyObject_DelItem',ops.eax,address_of(name))
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
        .invoke('PyDict_GetItem',GLOBALS,address_of(name))
        .if_eax_is_zero(f()
            .invoke('PyDict_GetItem',BUILTINS,address_of(name))
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
        .invoke('PyDict_SetItem',GLOBALS,address_of(name),tos)
        .check_err(True)
        .pop_stack(ops.eax)
        .decref()
    )

@hasconst
def _op_LOAD_CONST(f,const):
    if isinstance(const,types.CodeType):
        const = f.entry_points[id(const)][0]

    return (f()
        .push_tos(True)
        .mov(address_of(const),ops.eax)
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
    return f().goto_end() if f.stack.use_tos() else f().pop_stack(ops.eax).goto_end()

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
        .pop_stack(ops.edx)
        .push_stack(ops.eax)
        .decref(ops.edx)
    )

@handler
def _op_FOR_ITER(f,to):
    argreg = f.stack.arg_reg(n=0)
    return (f()
        .push_tos(True)
        (f.rtarget())
        .mov(f.stack[0],argreg)
        .mov(ops.Address(pyinternals.type_offset,argreg),ops.eax)
        .mov(ops.Address(pyinternals.type_iternext_offset,ops.eax),ops.eax)
        .invoke(ops.eax,argreg)
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
        JumpRSource(f.op.jmp,ops.JMP_DISP_MAX_LEN,f.reverse_target(to)))

@hasname
def _op_LOAD_ATTR(f,name):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_GetAttr',tos,address_of(name))
        .check_err()
        .pop_stack(ops.edx)
        .push_stack(ops.eax)
        .decref(ops.edx)
    )

def _op_pop_jump_if_(f,to,state):
    dont_jump = JumpTarget()
    jop1,jop2 = (f.op.jz,f.op.jg) if state else (f.op.jg,f.op.jz)
    tos = f.stack.tos()
    r = (f()
        .push_tos()
        .invoke('PyObject_IsTrue',tos)
        .pop_stack(ops.edx)
        .decref(ops.edx,True)
        .test(ops.eax,ops.eax)
        (JumpSource(jop1,dont_jump))
        (f.jump_to(jop2,ops.JCC_MAX_LEN,to))
        .xor(ops.eax,ops.eax)
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
                    .mov(ops.Address(item_offset,ops.eax),ops.ebx)
                    .push_stack(ops.eax)
                    .xor(ops.eax,ops.eax)
                )

                top.index = ops.eax

                lbody = (
                    f.op.mov(top,ops.edx) +
                    f.op.mov(ops.edx,ops.Address(-STACK_ITEM_SIZE,ops.ebx,ops.ecx,STACK_ITEM_SIZE)) +
                    f.inc_or_add(ops.eax))
            else:
                r.xor(ops.ebx,ops.ebx)

                top.index = ops.ebx

                lbody = (
                    f.op.mov(top,ops.edx) +
                    f.op.mov(ops.edx,ops.Address(item_offset-STACK_ITEM_SIZE,ops.eax,ops.ecx,STACK_ITEM_SIZE)) +
                    f.inc_or_add(ops.ebx))

            (r
                .mov(items,ops.ecx)
                (lbody)
                .loop(ops.Displacement(-len(lbody) - ops.LOOP_LEN)))

            f.stack.offset -= items
        else:
            if deref:
                r.mov(ops.Address(item_offset,ops.eax),ops.ebx)

            for i in reversed(range(items)):
                addr = ops.Address(STACK_ITEM_SIZE*i,ops.ebx) if deref else ops.Address(item_offset+STACK_ITEM_SIZE*i,ops.eax)
                r.pop_stack(ops.edx).mov(ops.edx,addr)


    return r

@handler
def _op_BUILD_LIST(f,items):
    return _op_BUILD_(f,items,'PyList_New',pyinternals.list_item_offset,True)

@handler
def _op_BUILD_TUPLE(f,items):
    return _op_BUILD_(f,items,'PyTuple_New',pyinternals.tuple_item_offset,False)

@handler
def _op_STORE_SUBSCR(f):
    tos = f.stack.tos()
    return (f()
        .push_tos()
        .invoke('PyObject_SetItem',f.stack[1],tos,f.stack[2])
        .check_err(True)
        .pop_stack(ops.eax)
        .decref()
        .pop_stack(ops.eax)
        .decref()
        .pop_stack(ops.eax)
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
        .mov(FAST_LOCALS,ops.ecx)
        .mov(ops.Address(STACK_ITEM_SIZE*arg,ops.ecx),ops.eax)
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
        r.pop_stack(ops.eax)

    item = ops.Address(STACK_ITEM_SIZE*arg,ops.ecx)
    return (r
        .mov(FAST_LOCALS,ops.ecx)
        .mov(item,ops.edx)
        .mov(ops.eax,item)
        .test(ops.edx,ops.edx)
        .if_cond[ops.test_NZ](
            join(f.decref(ops.edx))
        )
    )

@handler
def _op_UNPACK_SEQUENCE(f,arg):
    assert arg > 0

    r = f()
    if f.stack.use_tos():
        r.mov(ops.eax,ops.ebx)
    else:
        r.pop_stack(ops.ebx)

    check_list = JumpTarget()
    else_ = JumpTarget()
    done = JumpTarget()

    s_top = f.stack[-1]
    s_top.scale = STACK_ITEM_SIZE

    # a place to temporarily store the sequence
    seq_store = f.stack[-1-arg]

    (r
        .mov(ops.Address(pyinternals.type_offset,ops.ebx),ops.edx)
        .cmp('PyTuple_Type',ops.edx)
        (JumpSource(f.op.jne,check_list))
            .cmpl(arg,ops.Address(pyinternals.var_size_offset,ops.ebx))
            (JumpSource(f.op.jne,else_)))

    if arg >= f.tuning.unpack_seq_loop_threshhold:
        s_top.index = ops.eax
        body = join(f()
            .mov(ops.Address(pyinternals.tuple_item_offset-STACK_ITEM_SIZE,ops.ebx,ops.exc,STACK_ITEM_SIZE),ops.edx)
            .incref(ops.edx)
            .mov(ops.edx,s_top)
            .inc_or_add(ops.eax).code)
        (r
            .xor(ops.eax,ops.eax)
            .mov(arg,ops.ecx)
            (body)
            .loop(ops.Displacement(-len(body) - ops.LOOP_LEN)))
    else:
        itr = iter(reversed(range(arg)))
        def unpack_one(reg):
            i = next(itr)
            return (f()
                .mov(ops.Address(pyinternals.tuple_item_offset + STACK_ITEM_SIZE * i,ops.ebx),reg)
                .incref(reg)
                .mov(reg,f.stack[i-arg])).code

        r += interleave_ops([ops.eax,ops.ecx,ops.edx],unpack_one)


    (r
            .goto(done)
        (check_list)
        .cmp('PyList_Type',ops.edx)
        (JumpSource(f.op.jne,else_))
            .cmpl(arg,ops.Address(pyinternals.var_size_offset,ops.ebx))
            (JumpSource(f.op.jne,else_))
                .mov(ops.Address(pyinternals.list_item_offset,ops.ebx),ops.edx))
    
    if arg >= f.tuning.unpack_seq_loop_threshhold:
        s_top.index = ops.ebx
        body = join(f()
            .mov(ops.Address(-STACK_ITEM_SIZE,ops.edx,ops.ecx,STACK_ITEM_SIZE),ops.eax)
            .incref(ops.eax)
            .mov(ops.eax,s_top)
            .inc_or_add(ops.ebx).code)
        (r
            .mov(ops.ebx,seq_store)
            .mov(arg,ops.ecx)
            .xor(ops.ebx,ops.ebx)
            (body)
            .loop(ops.Displacement(-(len(body) + ops.LOOP_LEN)))
            .mov(seq_store,ops.ebx))
    else:
        itr = iter(reversed(range(arg)))
        def unpack_one(reg):
            i = next(itr)
            return (f()
                .mov(ops.Address(STACK_ITEM_SIZE * i,ops.edx),reg)
                .incref(reg)
                .mov(reg,f.stack[i-arg])).code

        r += interleave_ops([ops.eax,ops.ecx],unpack_one)


    f.stack.offset += arg

    p3 = f.stack.arg_reg(n=3)

    return (r
            .goto(done)
        (else_)
            .lea(f.stack[0],p3)
            .invoke('_unpack_iterable',ops.ebx,arg,-1,p3)
            .if_eax_is_zero(f()
                .decref(ops.ebx)
                .goto_end()
            )
        (done)
        .decref(ops.ebx))

@handler
def _op_UNPACK_EX(f,arg):
    totalargs = 1 + (arg & 0xff) + (arg >> 8)

    r = f()
    if f.stack.use_tos():
        r.mov(ops.eax,ops.ebx)
    else:
        r.pop(ops.ebx)

    f.stack.offset += totalargs
    argreg = f.stack.arg_reg(n=3)

    return (r
        .lea(f.stack[0],argreg)
        .invoke('_unpack_iterable',ops.ebx,arg & 0xff,arg >> 8,argreg)
        .decref(ops.ebx,True)
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
            .mov(f.stack[1],ops.edx)
            .mov(ops.eax,f.stack[1])
            .decref(ops.edx)
            .pop_stack(ops.edx)
            .decref(ops.edx)
        )

    if op == 'is' or op == 'is not':
        outcome_a,outcome_b = false_true_addr(op == 'is not')

        r = f()
        if f.stack.use_tos(True):
            r.mov(ops.eax,ops.edx)
        else:
            r.pop_stack(ops.edx)

        temp = f.stack[-1]
        return (r
            .cmp(ops.edx,f.stack[0])
            .movl(outcome_a,temp)
            .if_cond[ops.test_E](
                f.op.movl(outcome_b,temp)
            )
            .decref(ops.edx)
            .pop_stack(ops.edx)
            .decref(ops.edx)
            .mov(temp,ops.eax)
            .incref()
        )

    if op == 'in' or op == 'not in':
        outcome_a,outcome_b = false_true_addr(op == 'not in')

        tos = f.stack.tos()
        return (f()
            .push_tos()
            .invoke('PySequence_Contains',tos,f.stack[1])
            .test(ops.eax,ops.eax)
            .if_cond[ops.test_L](f()
                .xor(ops.eax,ops.eax)
                .goto_end()
            )
            .mov(outcome_a,ops.eax)
            .if_cond[ops.test_NZ](
                f.op.mov(outcome_b,ops.eax)
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
        p0 = f.stack.arg_reg(tempreg=ops.edx,n=0)
        p1 = ops.eax
        if not f.stack.use_tos():
            p1 = f.stack.arg_reg(tempreg=ops.eax,n=1)
            r.pop_stack(p1)

        (r
            .pop_stack(p0)
            .push_arg(p1,n=1)
            .push_arg(p0,n=0)
        )
    elif arg == 1:
        p0 = ops.eax
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
        .pop_stack(ops.edx)
        .decref(ops.edx)
        .pop_stack(ops.edx)
        .decref(ops.edx)
    )

@handler
def _op_STORE_MAP(f):
    return _op_map_store_add(0,f)

@handler
def _op_MAP_ADD(f,arg):
    return _op_map_store_add(arg-1,f)



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


def local_name_func(op,tuning):
    # this function uses an ad hoc calling convention and is only callable from
    # the code generated by this module

    local_stack_size = aligned_size((MAX_ARGS + 1) * STACK_ITEM_SIZE)
    stack_ptr_shift = local_stack_size - STACK_ITEM_SIZE

    f = Frame(op,tuning,local_stack_size)

    
    else_ = JumpTarget()
    endif = JumpTarget()
    ret = JumpTarget()
    
    return (f()
        .sub(stack_ptr_shift,ops.esp)
        .mov(LOCALS,ops.eax)
        .if_eax_is_zero(f()
            .invoke('PyErr_Format','PyExc_SystemError','NO_LOCALS_LOAD_MSG',ops.ebx)
            .add(stack_ptr_shift,ops.esp)
            .ret()
        )
        
        # if (%eax)->ob_type != PyDict_Type:
        .cmpl('PyDict_Type',ops.Address(pyinternals.type_offset,ops.eax))
        (JumpSource(f.op.je,else_))
            .invoke('PyObject_GetItem',ops.eax,ops.ebx)
            .test(ops.eax,ops.eax)
            (JumpSource(f.op.jnz,ret))
            
            .invoke('PyErr_ExceptionMatches','PyExc_KeyError')
            .test(ops.eax,ops.eax)
            (JumpSource(f.op.jz,ret))
            .call('PyErr_Clear')

            .goto(endif)
        (else_)
            .invoke('PyDict_GetItem',ops.eax,ops.ebx)
            .test(ops.eax,ops.eax)
            (JumpSource(f.op.jnz,ret))
        (endif)
    
        .invoke('PyDict_GetItem',GLOBALS,ops.ebx)
        .if_eax_is_zero(f()
            .invoke('PyDict_GetItem',BUILTINS,ops.ebx)
            .if_eax_is_zero(f()
                .invoke('format_exc_check_arg',
                    'NAME_ERROR_MSG',
                    'PyExc_NameError')
            )
        )
        
        (ret)
        .add(stack_ptr_shift,ops.esp)
        .ret()
    )



def compile_eval(code,op,tuning,local_name,entry_points):
    """Generate a function equivalent to PyEval_EvalFrame called with f.code"""

    # the stack will have following items:
    #     - return address
    #     - old value of ebp
    #     - old value of ebx
    #     - GLOBALS
    #     - BUILTINS
    #     - LOCALS
    #     - FAST_LOCALS
    #
    # the first 3 will already be on the stack by the time %esp is adjusted

    stack_first = 4

    if pyinternals.ref_debug:
        # a place to store %eax,%ecx and %edx when increasing reference counts
        # (which calls a function when ref_debug is True)
        stack_first += DEBUG_TEMPS

    local_stack_size = aligned_size(
        (code.co_stacksize + MAX_ARGS + PRE_STACK + stack_first) * STACK_ITEM_SIZE)

    stack_ptr_shift = local_stack_size - PRE_STACK * STACK_ITEM_SIZE

    f = Frame(op,tuning,local_stack_size,code,local_name,entry_points)

    opcodes = (f()
        .push(ops.ebp)
        .mov(ops.esp,ops.ebp)
        .push(ops.ebx)
        .push(ops.esi)
        .sub(stack_ptr_shift,ops.esp)
    )
    f.stack.offset = PRE_STACK

    argreg = f.stack.arg_reg(n=0)
    (opcodes
        .lea(f.stack[-1],argreg)
        .movl(0,f.stack[-1])
        .invoke('_EnterRecursiveCall',argreg)
        .check_err(True)

        .mov(f.stack.func_arg(0),ops.eax)

        # as far as I can tell, after expanding the macros and removing the
        # "dynamic annotations" (see dynamic_annotations.h in the CPython
        # headers), this is all that PyThreadState_GET boils down to:
        .mov(ops.Address(pyinternals.raw_addresses['_PyThreadState_Current']),ops.ecx)

        .mov(ops.Address(pyinternals.frame_globals_offset,ops.eax),ops.edx)
        .mov(ops.Address(pyinternals.frame_builtins_offset,ops.eax),ops.ebx)

        .push_stack(ops.edx)
        .push_stack(ops.ebx)

        .mov(ops.Address(pyinternals.frame_locals_offset,ops.eax),ops.edx)
        .lea(ops.Address(pyinternals.frame_localsplus_offset,ops.eax),ops.ebx)

        .push_stack(ops.edx)
        .push_stack(ops.ebx)

        .mov(ops.eax,ops.Address(pyinternals.threadstate_frame_offset,ops.ecx))
    )

    if pyinternals.ref_debug:
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
        .mov(ops.Address(base=ops.ebx),ops.edx)
        .decref(ops.edx)
        .add(STACK_ITEM_SIZE,ops.ebx)
        .code)
    
    cmpjl = f.op.cmp(ops.ebp,ops.ebx)
    jlen = -(len(dr) + len(cmpjl) + ops.JCC_MIN_LEN)
    assert ops.fits_in_sbyte(jlen)
    cmpjl += f.op.jb(ops.Displacement(jlen))
    
    # call Py_DECREF on anything left on the stack and return %eax
    (opcodes
        (f._end)
        .mov(ops.eax,ops.Address(STACK_ITEM_SIZE*-stack_prolog,ops.ebp))
        .sub(STACK_ITEM_SIZE*stack_prolog,ops.ebp)
        .jmp(ops.Displacement(len(dr)))
        (dr)
        (cmpjl)
        .call('_LeaveRecursiveCall')
        .mov(ops.Address(base=ops.ebp),ops.eax)
        .mov(ops.Address(pyinternals.raw_addresses['_PyThreadState_Current']),ops.ecx)
        .mov(f.stack.func_arg(0),ops.edx)
        .add(stack_ptr_shift,ops.esp)
        .pop(ops.esi)
        .pop(ops.ebx)
        .mov(ops.Address(pyinternals.frame_back_offset,ops.edx),ops.edx)
        .pop(ops.ebp)
        .mov(ops.edx,ops.Address(pyinternals.threadstate_frame_offset,ops.ecx))
        .ret()
    )

    return opcodes


def compile_raw(_code,binary = True,tuning=Tuning()):
    local_name = JumpTarget()
    entry_points = {}
    op = ops if binary else ops.Assembly()

    ceval = partial(compile_eval,
        op=op,
        tuning=tuning,
        local_name=local_name,
        entry_points=entry_points)

    def compile_code_constants(code):
        for c in code.co_consts:
            if isinstance(c,types.CodeType) and id(c) not in entry_points:
                compile_code_constants(c)
                entry_points[id(c)] = (
                    pyinternals.create_compiled_entry_point(c),
                    ceval(c))
    
    compile_code_constants(_code)
    main_entry = ceval(_code)

    functions = []
    end_targets = []
    
    if local_name.used:
        local_name.displacement = 0
        end_targets.append(local_name)
        functions.append(resolve_jumps(op,local_name_func(op,tuning).code))

    entry_points = list(entry_points.values())
    for ep,func in reversed(entry_points):
        functions.insert(0,resolve_jumps(op,func.code,end_targets))

    main_entry = resolve_jumps(op,main_entry.code,end_targets)
    offset = len(main_entry)
    for epf,func in zip(entry_points,functions):
        pyinternals.cep_set_offset(epf[0],offset)
        offset += len(func)

    functions.insert(0,main_entry)

    if not binary:
        functions = join(functions)
    
    return functions,[ep for ep,func in entry_points]


