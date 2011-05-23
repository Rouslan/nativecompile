
import sys
import dis
import weakref
import operator
import types
from functools import partial, reduce

from . import x86_ops as ops
from . import pyinternals


PRINT_STACK_OFFSET = False


STACK_ITEM_SIZE = 4
CALL_ALIGN_MASK = 0xf


TPFLAGS_INT_SUBCLASS = 1<<23
TPFLAGS_LONG_SUBCLASS = 1<<24
TPFLAGS_LIST_SUBCLASS = 1<<25
TPFLAGS_TUPLE_SUBCLASS = 1<<26
TPFLAGS_BYTES_SUBCLASS = 1<<27
TPFLAGS_UNICODE_SUBCLASS = 1<<28
TPFLAGS_DICT_SUBCLASS = 1<<29
TPFLAGS_BASE_EXC_SUBCLASS = 1<<30
TPFLAGS_TYPE_SUBCLASS = 1<<31



class StackManager:
    def __init__(self,op):
        self.op = op

        # The number of bytes the stack has moved. When this is None, we don't
        # know what the offset is because the previous opcode is a jump past the
        # current opcode. 'resets' is expected to indicate what the offset
        # should be now.
        self.offset = 0

         # if True, the TOS item is in %eax and has not actually been pushed
         # onto the stack
        self.tos_in_eax = False

        self.resets = []
    
    def push(self,x):
        self.offset += STACK_ITEM_SIZE
        return self.op.push(x)
    
    def pop(self,x):
        self.offset -= STACK_ITEM_SIZE
        return self.op.pop(x)
    
    def add_to(self,amount):
        self.offset -= amount
        return self.op.add(amount,ops.esp)
    
    def sub_from(self,amount):
        self.offset += amount
        return self.op.sub(amount,ops.esp)
    
    def push_tos(self,set_again = False,extra_push = False):
        """%eax is needed right now so if the TOS item hasn't been pushed onto 
        the stack, do it now."""
        r = [self.push(ops.eax)] if self.tos_in_eax else []
        if extra_push:
            r.append(self.op.push(ops.eax if self.tos_in_eax else ops.Address(base=ops.esp)))
        self.tos_in_eax = set_again
        return r
    
    def use_tos(self,set_again = False):
        r = self.tos_in_eax
        self.tos_in_eax = set_again
        return r

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
                r.counted_pop(ops.edx).decref(ops.edx)

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


GLOBALS = ops.Address(-STACK_ITEM_SIZE,ops.ebp)
BUILTINS = ops.Address(STACK_ITEM_SIZE * -2,ops.ebp)
LOCALS = ops.Address(STACK_ITEM_SIZE * -3,ops.ebp)
FAST_LOCALS = ops.Address(STACK_ITEM_SIZE * -4,ops.ebp)


def raw_addr_if_str(x):
    return pyinternals.raw_addresses[x] if isinstance(x,str) else x

class Frame:
    def __init__(self,code,op,tuning,local_name,entry_points):
        self.code = code
        self.op = op
        self.tuning = tuning
        self.stack = StackManager(op)
        self.end = JumpTarget()
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
        return [
            self.op.test(ops.eax,ops.eax),
            JumpSource(self.op.jnz if inverted else self.op.jz,self.end)]
    
    def call(self,func):
        return [
            self.op.mov(pyinternals.raw_addresses[func],ops.eax),
            self.op.call(ops.eax)]

    def invoke(self,func,*args):
        if not args: return self.call(func)
        return [self.op.push(raw_addr_if_str(a)) for a in reversed(args)] + self.call(func) + [self.op.add(len(args)*STACK_ITEM_SIZE,ops.esp)]

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

    def discard_stack_items(self,n):
        return self.op.add(STACK_ITEM_SIZE * n,ops.esp)

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
            return [self.op.push(ops.eax)] + self.invoke('Py_IncRef',reg) + [self.op.pop(ops.eax)]

        return [self.incl_or_addl(ops.Address(pyinternals.refcnt_offset,reg))]

    def decref(self,reg = ops.eax,preserve_eax = False):
        if pyinternals.ref_debug:
            inv = self.invoke('Py_DecRef',reg)
            return [self.op.push(ops.eax)] + inv + [self.op.pop(ops.eax)] if preserve_eax else inv

        assert reg.reg != ops.ecx.reg
        
        mid = []
        
        if preserve_eax:
            mid.append(self.op.push(ops.eax))

        mid += [
            self.op.mov(ops.Address(pyinternals.type_offset,reg),ops.ecx),
            self.op.push(reg)
        ]

        if pyinternals.count_allocs:
            mid += self.invoke('inc_count',ops.ecx)

        mid += [
            self.op.call(ops.Address(pyinternals.type_dealloc_offset,ops.ecx)),
            self.discard_stack_items(1)
        ]
        
        if preserve_eax:
            mid.append(self.op.pop(ops.eax))
        
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
            raise Exception('unexpected jump target')

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

    def discard_stack_items(self,n):
        self.code.append(self.f.discard_stack_items(n))
        return self

    def push_tos(self,set_again = False,extra_push = False):
        self.code += self.f.stack.push_tos(set_again,extra_push)
        return self

    def counted_push(self,x):
        self.code.append(self.f.stack.push(x))
        return self

    def counted_pop(self,x):
        self.code.append(self.f.stack.pop(x))
        return self

    def counted_discard_stack_items(self,n):
        self.code.append(self.f.stack.add_to(STACK_ITEM_SIZE*n))
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
        if isinstance(x,str):
            self.code += self.f.call(x)
        else:
            self.code.append(self.f.op.call(x))
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
    'decref']:
    setattr(Stitch,func,_forward_list_func(getattr(Frame,func)))



def _binary_op(f,func):
    return (f()
        .push_tos(True,True)
        .push(ops.Address(STACK_ITEM_SIZE*2,ops.esp))
        .call(func)
        .discard_stack_items(2)
        .check_err()
        .counted_pop(ops.edx)
        .decref(ops.edx,True)
        .counted_pop(ops.edx)
        .decref(ops.edx,True)
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
        r.insert(0,f.stack.pop(ops.eax))
    return r

@hasname
def _op_LOAD_NAME(f,name):
    return (f()
        .push_tos(True)
        .push(address_of(name))
        (InnerCall(f.op,f.local_name))
        .test(ops.eax,ops.eax)
        (JumpSource(f.op.jz,f.end))
        .incref()
    )

@hasname
def _op_STORE_NAME(f,name):
    return (f()
        .push_tos(extra_push=True)
        .push(address_of(name))
        .mov(LOCALS,ops.eax)
        .if_eax_is_zero(f()
            .push('NO_LOCALS_STORE_MSG')
            .push('PyExc_SystemError')
            .call('PyErr_Format')
            .discard_stack_items(4)
            .goto(f.end)
        )
        .mov('PyObject_SetItem',ops.ecx)
        .cmpl('PyDict_Type',ops.Address(pyinternals.type_offset,ops.eax))
        .push(ops.eax)
        .if_cond[ops.test_E](
            f.op.mov(pyinternals.raw_addresses['PyDict_SetItem'],ops.ecx)
        )
        .call(ops.ecx)
        .discard_stack_items(3)
        .check_err(True)
        .counted_pop(ops.eax)
        .decref()
    )

@hasname
def _op_DELETE_NAME(f,name):
    return (f()
        .push_tos()
        .push(address_of(name))
        .mov(LOCALS,ops.eax)
        .if_eax_is_zero(f()
            .push('NO_LOCALS_DELETE_MSG')
            .push('PyExc_SystemError')
            .call('PyErr_Format')
            .discard_stack_items(3)
            .goto(f.end)
        )
        .push(ops.eax)
        .call('PyObject_DelItem')
        .discard_stack_items(2)
        .if_eax_is_zero(f()
            .invoke('format_exc_check_arg',
                'PyExc_NameError',
                'NAME_ERROR_MSG',
                address_of(name))
            .goto(f.end)
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
                .goto(f.end)
            )
        )
        .incref()
    )

@hasname
def _op_STORE_GLOBAL(f,name):
    return (f()
        .push_tos(extra_push=True)
        .push(address_of(name))
        .push(GLOBALS)
        .call('PyDict_SetItem')
        .discard_stack_items(3)
        .check_err(True)
        .counted_pop(ops.eax)
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
    ret = f.op.ret()
    return (f()
        .push_tos(True)
        .mov(ops.esp,ops.eax)
        .counted_push(arg)
        .counted_push(ops.eax)
        .call('call_function')

        # +3 for arg, the function object and the address of the stack
        .counted_discard_stack_items((arg & 0xFF) + ((arg >> 8) & 0xFF) + 3)

        .check_err()
    )

@handler
def _op_RETURN_VALUE(f):
    return f().goto(f.end) if f.stack.use_tos() else f().counted_pop(ops.eax).goto(f.end)

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
    return (f()
        .push_tos(extra_push=True)
        .call('PyObject_GetIter')
        .discard_stack_items(1)
        .check_err()
        .pop(ops.edx)
        .push(ops.eax)
        .decref(ops.edx)
    )

@handler
def _op_FOR_ITER(f,to):
    return (f()
        .push_tos(True)
        (f.rtarget())
        .push(ops.Address(base=ops.esp))
        .mov(ops.Address(base=ops.esp),ops.eax)
        .mov(ops.Address(pyinternals.type_offset,ops.eax),ops.eax)
        .mov(ops.Address(pyinternals.type_iternext_offset,ops.eax),ops.eax)
        .call(ops.eax)
        .discard_stack_items(1)
        .if_eax_is_zero(f()
            .call('PyErr_Occurred')
            .if_eax_is_not_zero(f()
                .invoke('PyErr_ExceptionMatches','PyExc_StopIteration')
                .test(ops.eax,ops.eax)
                (JumpSource(f.op.jz,f.end))
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
    in_eax = f.stack.tos_in_eax
    return (f()
        .push_tos()
        .push(address_of(name))
        .push(ops.eax if in_eax else ops.Address(STACK_ITEM_SIZE,ops.esp))
        .call('PyObject_GetAttr')
        .discard_stack_items(2)
        .check_err()
        .pop(ops.edx)
        .push(ops.eax)
        .decref(ops.edx)
    )

def _op_pop_jump_if_(f,to,state):
    dont_jump = JumpTarget()
    jop1,jop2 = (f.op.jz,f.op.jg) if state else (f.op.jg,f.op.jz)
    r = (f()
        .push_tos(extra_push=True)
        .call('PyObject_IsTrue')
        .discard_stack_items(1)
        .counted_pop(ops.edx)
        .decref(ops.edx,True)
        .test(ops.eax,ops.eax)
        (JumpSource(jop1,dont_jump))
        (f.jump_to(jop2,ops.JCC_MAX_LEN,to))
        .xor(ops.eax,ops.eax)
        .goto(f.end)
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
        f.stack.offset -= STACK_ITEM_SIZE * items

        if items >= f.tuning.build_seq_loop_threshhold:
            if deref:
                r.mov(ops.Address(item_offset,ops.eax),ops.ebx)

                lbody = (
                    f.op.pop(ops.edx) +
                    f.op.mov(ops.edx,ops.Address(-STACK_ITEM_SIZE,ops.ebx,ops.ecx,STACK_ITEM_SIZE)))
            else:
                lbody = (
                    f.op.pop(ops.edx) +
                    f.op.mov(ops.edx,ops.Address(item_offset-STACK_ITEM_SIZE,ops.eax,ops.ecx,STACK_ITEM_SIZE)))

            (r
                .mov(items,ops.ecx)
                (lbody)
                .loop(ops.Displacement(-(len(lbody) + ops.LOOP_LEN))))
        else:
            if deref:
                r.mov(ops.Address(item_offset,ops.eax),ops.ebx)

            for i in reversed(range(items)):
                addr = ops.Address(STACK_ITEM_SIZE*i,ops.ebx) if deref else ops.Address(item_offset+STACK_ITEM_SIZE*i,ops.eax)
                r.pop(ops.edx).mov(ops.edx,addr)

    return r

@handler
def _op_BUILD_LIST(f,items):
    return _op_BUILD_(f,items,'PyList_New',pyinternals.list_item_offset,True)

@handler
def _op_BUILD_TUPLE(f,items):
    return _op_BUILD_(f,items,'PyTuple_New',pyinternals.tuple_item_offset,False)

@handler
def _op_STORE_SUBSCR(f):
    in_eax = f.stack.tos_in_eax
    return (f()
        .push_tos()
        .push(ops.Address(STACK_ITEM_SIZE*2,ops.esp))
        .push(ops.eax if in_eax else ops.Address(STACK_ITEM_SIZE,ops.esp))
        .push(ops.Address(STACK_ITEM_SIZE*3,ops.esp))
        .call('PyObject_SetItem')
        .discard_stack_items(3)
        .check_err(True)
        .counted_pop(ops.eax)
        .decref()
        .counted_pop(ops.eax)
        .decref()
        .counted_pop(ops.eax)
        .decref()
    )

def _op_make_callable(f,arg,closure):
    annotations = (arg >> 16) & 0x7fff

    # +3 for the code object, arg and closure
    sitems = (arg & 0xff) + ((arg >> 8) & 0xff) * 2 + annotations + 3

    if closure: sitems += 1
    if annotations: sitems += 1

    return (f()
        .push_tos(True)
        .counted_push(arg)
        .counted_push(int(bool(closure)))
        .call('_make_function')
        .counted_discard_stack_items(sitems)
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
            .goto(f.end)
        )
        .incref()
    )

@handler
def _op_STORE_FAST(f,arg):
    r = f()
    if not f.stack.use_tos():
        r.counted_pop(ops.eax)

    item = ops.Address(STACK_ITEM_SIZE*arg,ops.ecx)
    decref = join(f.decref(ops.edx))
    return (r
        .mov(FAST_LOCALS,ops.ecx)
        .mov(item,ops.edx)
        .mov(ops.eax,item)
        .test(ops.edx,ops.edx)
        .jz(ops.Displacement(len(decref)))
        (decref)
    )

@handler
def _op_UNPACK_SEQUENCE(f,arg):
    assert arg > 0

    f.stack.offset += STACK_ITEM_SIZE * arg
    r = f()
    if f.stack.use_tos():
        r.mov(ops.eax,ops.ebx)
    else:
        r.counted_pop(ops.ebx)

    check_list = JumpTarget()
    else_ = JumpTarget()
    done = JumpTarget()

    (r
        .mov(ops.Address(pyinternals.type_offset,ops.ebx),ops.edx)
        .cmp('PyTuple_Type',ops.edx)
        (JumpSource(f.op.jne,check_list))
            .cmpl(arg,ops.Address(pyinternals.var_size_offset,ops.ebx))
            (JumpSource(f.op.jne,else_)))

    if arg >= f.tuning.unpack_seq_loop_threshhold:
        body = join(f()
            .mov(ops.Address(pyinternals.tuple_item_offset-STACK_ITEM_SIZE,ops.ebx,ops.exc,STACK_ITEM_SIZE),ops.edx)
            .incref(ops.edx)
            .push(ops.edx).code)
        (r
            .mov(arg,ops.ecx)
            (body)
            .loop(ops.Displacement(-(len(body) + ops.LOOP_LEN))))
    else:
        itr = iter(reversed(range(arg)))
        def unpack_one(reg):
            i = next(itr)
            return (f()
                .mov(ops.Address(pyinternals.tuple_item_offset + STACK_ITEM_SIZE * i,ops.ebx),reg)
                .incref(reg)
                .push(reg)).code

        r += interleave_ops([ops.eax,ops.ecx,ops.edx],unpack_one)
        r.goto(done)

    (r
        (check_list)
        .cmp('PyList_Type',ops.edx)
        (JumpSource(f.op.jne,else_))
            .cmpl(arg,ops.Address(pyinternals.var_size_offset,ops.ebx))
            (JumpSource(f.op.jne,else_))
                .mov(ops.Address(pyinternals.list_item_offset,ops.ebx),ops.edx))
    
    if arg >= f.tuning.unpack_seq_loop_threshhold:
        body = join(f()
            .mov(ops.Address(-STACK_ITEM_SIZE,ops.edx,ops.exc,STACK_ITEM_SIZE),ops.eax)
            .incref(ops.eax)
            .push(ops.eax).code)
        (r
            .mov(arg,ops.ecx)
            (body)
            .loop(ops.Displacement(-(len(body) + ops.LOOP_LEN))))
    else:
        itr = iter(reversed(range(arg)))
        def unpack_one(reg):
            i = next(itr)
            return (f()
                .mov(ops.Address(STACK_ITEM_SIZE * i,ops.edx),reg)
                .incref(reg)
                .push(reg)).code

        r += interleave_ops([ops.eax,ops.ecx],unpack_one)
        r.goto(done)

    return (r
        (else_)
            .sub(arg * STACK_ITEM_SIZE,ops.esp)
            .invoke('_unpack_iterable',ops.ebx,arg,-1,ops.esp)
            .if_eax_is_zero(f()
                .add(arg * STACK_ITEM_SIZE,ops.esp)
                .decref(ops.ebx)
                .goto(f.end)
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

    f.stack.offset += totalargs * STACK_ITEM_SIZE

    return (r
        .sub(totalargs * STACK_ITEM_SIZE,ops.esp)
        .invoke('_unpack_iterable',ops.ebx,arg & 0xff,arg >> 8,ops.esp)
        .decref(ops.ebx,True)
        .if_eax_is_zero(f()
            .add(arg * STACK_ITEM_SIZE,ops.esp)
            .goto(f.end)
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
            .counted_pop(ops.edx)
            .decref(ops.edx,True)
            .counted_pop(ops.edx)
            .decref(ops.edx,True)
        )

    if op == 'is' or op == 'is not':
        outcome_a,outcome_b = false_true_addr(op == 'is not')

        r = f()
        if f.stack.use_tos(True):
            r.mov(ops.eax,ops.edx)
        else:
            r.counted_pop(ops.edx)

        return (r
            .cmp(ops.edx,ops.Address(base=ops.esp))
            .mov(outcome_a,ops.eax)
            .if_cond[ops.test_E](
                f.op.mov(outcome_b,ops.eax)
            )
            .decref(ops.edx,True)
            .counted_pop(ops.edx)
            .decref(ops.edx,True)
            .incref()
        )

    if op == 'in' or op == 'not in':
        outcome_a,outcome_b = false_true_addr(op == 'not in')

        return (f()
            .push_tos(True)
            .push(ops.Address(STACK_ITEM_SIZE,ops.esp))
            .push(ops.Address(STACK_ITEM_SIZE,ops.esp))
            .call('PySequence_Contains')
            .discard_stack_items(2)
            .test(ops.eax,ops.eax)
            .if_cond[ops.test_L](f()
                .xor(ops.eax,ops.eax)
                .goto(f.end)
            )
            .mov(outcome_a,ops.eax)
            .if_cond[ops.test_NZ](
                f.op.mov(outcome_b,ops.eax)
            )
            .incref()
        ) + pop_args()

    if op == 'exception match':
        return (f()
            .push_tos(True,True)
            .push(ops.Address(STACK_ITEM_SIZE*2,ops.esp))
            .call('_exception_cmp')
            .discard_stack_items(2)
            .check_err()
        ) + pop_args()

    return (f()
        .push_tos(True)
        .push(arg)
        .push(ops.Address(STACK_ITEM_SIZE,ops.esp))
        .push(ops.Address(STACK_ITEM_SIZE*3,ops.esp))
        .call('PyObject_RichCompare')
        .discard_stack_items(3)
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
        if not f.stack.use_tos():
            r.counted_pop(ops.eax)
        (r
            .counted_pop(ops.ecx)
            .push(ops.eax)
            .push(ops.ecx)
        )
    elif arg == 1:
        if not f.stack.use_tos():
            r.counted_pop(ops.eax)
        (r
            .push(0)
            .push(ops.eax)
        )
    elif arg == 0:
        (r
            .push_tos()
            .push(0)
            .push(0)
        )
    else:
        raise SystemError("bad RAISE_VARARGS oparg")

    # We don't have to worry about decrementing the reference counts. _do_raise
    # does that for us.
    return (r
        .call('_do_raise')
        .discard_stack_items(2)
        .goto(f.end)
    )

@handler
def _op_BUILD_MAP(f,arg):
    return (f()
        .push_tos(True)
        .invoke('_PyDict_NewPresized',arg)
        .check_err()
    )

def _op_map_store_add(offset,f):
    in_eax = f.stack.tos_in_eax
    return (f()
        .push_tos()
        .push(ops.Address(STACK_ITEM_SIZE,ops.esp))
        .push(ops.eax if in_eax else ops.Address(STACK_ITEM_SIZE,ops.esp))
        .push(ops.Address(STACK_ITEM_SIZE*(4+offset),ops.esp))
        .call('PyDict_SetItem')
        .discard_stack_items(3)
        .check_err(True)
        .counted_pop(ops.edx)
        .decref(ops.edx)
        .counted_pop(ops.edx)
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

    # add padding so that the total length is a multiple of CALL_ALIGN_MASK + 1
    if CALL_ALIGN_MASK:
        pad_size = ((displacement + CALL_ALIGN_MASK) & ~CALL_ALIGN_MASK) - displacement
        chunks += [op.nop()] * pad_size
        for et in end_targets:
            et.displacement -= pad_size
    
    code = join([(c.compile() if isinstance(c,DelayedCompile) else c) for c in chunks])

    for et in end_targets:
        et.displacement -= displacement

    return code


def local_name_func(f):
    # this function uses an ad hoc calling convention and is only callable from
    # the code generated by this module
    
    else_ = JumpTarget()
    endif = JumpTarget()
    found = JumpTarget()
    
    return (f()
        .mov(LOCALS,ops.eax)
        .push(ops.Address(STACK_ITEM_SIZE,ops.esp))
        .if_eax_is_zero(f()
            .push('NO_LOCALS_LOAD_MSG')
            .push('PyExc_SystemError')
            .call('PyErr_Format')
            .discard_stack_items(3)
            .ret(STACK_ITEM_SIZE)
        )
        .push(ops.eax)
        
        # if (%eax)->ob_type != PyDict_Type:
        .cmpl('PyDict_Type',ops.Address(pyinternals.type_offset,ops.eax))
        (JumpSource(f.op.je,else_))
            .call('PyObject_GetItem')
            .discard_stack_items(2)
            .test(ops.eax,ops.eax)
            (JumpSource(f.op.jnz,found))
            
            .invoke('PyErr_ExceptionMatches','PyExc_KeyError')
            .if_eax_is_zero(
                f.op.ret(STACK_ITEM_SIZE)
            )
            .call('PyErr_Clear')

            .goto(endif)
        (else_)
            .call('PyDict_GetItem')
            .discard_stack_items(2)
            .test(ops.eax,ops.eax)
            (JumpSource(f.op.jnz,found))
        (endif)
    
        .invoke('PyDict_GetItem',GLOBALS,ops.Address(STACK_ITEM_SIZE,ops.esp))
        .if_eax_is_zero(f()
            .invoke('PyDict_GetItem',BUILTINS,ops.Address(STACK_ITEM_SIZE,ops.esp))
            .if_eax_is_zero(f()
                .invoke('format_exc_check_arg',
                    'NAME_ERROR_MSG',
                    'PyExc_NameError')
            )
        )
        
        (found)
        .ret(STACK_ITEM_SIZE)
    )



def compile_eval(f):
    """Generate a function equivalent to PyEval_EvalFrame, called with f.code"""

    opcodes = (f()
        .push(0)
        .push(ops.esp)
        .call('_EnterRecursiveCall')
        .discard_stack_items(2)
        .if_eax_is_not_zero(
            f.op.ret()
        )
        .counted_push(ops.ebp)
        .counted_push(ops.ebx)
        .mov(ops.esp,ops.ebp)
        .mov(ops.Address(STACK_ITEM_SIZE*3,ops.esp),ops.eax)
        .lea(ops.Address(pyinternals.frame_localsplus_offset,ops.eax),ops.edx)

        # as far as I can tell, after expanding the macros and removing the
        # "dynamic annotations" (see dynamic_annotations.h in the CPython
        # headers), this is all that PyThreadState_GET boils down to:
        .mov(ops.Address(pyinternals.raw_addresses['_PyThreadState_Current']),ops.ecx)

        .counted_push(ops.Address(pyinternals.frame_globals_offset,ops.eax))
        .counted_push(ops.Address(pyinternals.frame_builtins_offset,ops.eax))
        .counted_push(ops.Address(pyinternals.frame_locals_offset,ops.eax))
        .counted_push(ops.edx)

        .mov(ops.eax,ops.Address(pyinternals.threadstate_frame_offset,ops.ecx))
    )
    
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
    
    
    assert f.stack.offset == stack_prolog, 'stack.offset should be {0}, but is {1}'.format(stack_prolog,f.stack.offset)
    assert not f.blockends
    
    dr = join([f.op.pop(ops.edx)] + f.decref(ops.edx,True))
    
    cmpjl = f.op.cmp(ops.ebp,ops.esp)
    jlen = -(len(dr) + len(cmpjl) + ops.JCC_MIN_LEN)
    assert ops.fits_in_sbyte(jlen)
    cmpjl += f.op.jb(ops.Displacement(jlen))
    
    # call Py_DECREF on anything left on the stack and return %eax
    (opcodes
        (f.end)
        .sub(stack_prolog - STACK_ITEM_SIZE*2,ops.ebp)
        .jmp(ops.Displacement(len(dr)))
        (dr)
        (cmpjl)
        .mov(ops.Address(pyinternals.raw_addresses['_PyThreadState_Current']),ops.ecx)
        .add(stack_prolog - STACK_ITEM_SIZE*2,ops.esp)
        .pop(ops.ebx)
        .mov(ops.Address(STACK_ITEM_SIZE*2,ops.esp),ops.edx)
        .pop(ops.ebp)
        .mov(ops.Address(pyinternals.frame_back_offset,ops.edx),ops.edx)
        .push(ops.eax)
        .mov(ops.edx,ops.Address(pyinternals.threadstate_frame_offset,ops.ecx))
        .call('_LeaveRecursiveCall')
        .pop(ops.eax)
        .ret()
    )

    return opcodes


def compile_raw(_code,binary = True,tuning=Tuning()):
    local_name = JumpTarget()
    entry_points = {}
    op = ops if binary else ops.Assembly()

    F = partial(Frame,
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
                    compile_eval(F(c)))
    
    compile_code_constants(_code)
    main_entry = compile_eval(F(_code))

    functions = []
    end_targets = []
    
    if local_name.used:
        local_name.displacement = 0
        end_targets.append(local_name)
        functions.append(resolve_jumps(op,local_name_func(F(None)).code))

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


