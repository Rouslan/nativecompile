
import dis
import weakref
import operator
from functools import partial, reduce

from . import x86_ops as ops
from . import pyinternals


STACK_ITEM_SIZE = 4




class StackManager:
    def __init__(self,op):
        self.op = op
        self.offset = 0 # the number of bytes the stack has moved
        self.tos_in_eax = False # if True, the TOS item is in %eax and has not actually been pushed onto the stack
    
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
    
    def use_tos(self):
        r = self.tos_in_eax
        self.tos_in_eax = False
        return r


# displacement is deliberately not set so it cannot be used until it is first
# encountered in the resolve_jumps loop
class JumpTarget:
    def displace(self,amount):
        self.displacement += amount
    
    def resolve(self):
        return ops.Displacement(self.displacement)


class JumpSource:
    def __init__(self,op,target):
        self.op = op
        self.target = target
    
    def compile(self):
        dis = self.target.resolve()
        return self.op(dis) if dis.val else b'' #optimize away useless jumps


class JumpRSource:
    def __init__(self,opset,op,max_size,target):
        self.op = op
        self.size = max_size
        self.opset = opset
        target.sources.append(self)

    def displace(self,amount):
        self.displacement += amount

    def compile(self):
        c = self.op(ops.Displacement(-self.displacement))
        assert len(c) <= self.size
        # pad the instruction with NOPs if it's too short
        return c + self.opset.nop() * (self.size - len(c))

    def __len__(self):
        return self.size

class JumpRTarget:
    def __init__(self):
        self.sources = []


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


handlers = [None] * 0xFF

def handler(func,name = None):
    def inner(f,*extra):
        r = f()
        if f.forward_targets and f.forward_targets[0][0] <= f.byte_offset:
            ft = f.forward_targets.pop(0)
            assert ft[0] == f.byte_offset
            r.push_tos()(ft[1])

        return r + func(f,*extra)

    handlers[dis.opmap[(name or func.__name__)[len('_op_'):]]] = inner
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


GLOBALS = ops.Address(-4,ops.ebp)
BUILTINS = ops.Address(-8,ops.ebp)
LOCALS = ops.Address(-12,ops.ebp)


class Frame:
    def __init__(self,code,op):
        self.code = code
        self.op = op
        self.stack = StackManager(op)
        self.end = JumpTarget()
        self._local_name = None
        self.blockends = []
        self.byte_offset = None
        self.forward_targets = []

        # Although JUMP_ABSOLUTE could jump to any instruction, we assume
        # compiled Python code only uses it in certain cases. Thus, we can make
        # small optimizations between successive instructions without needing an
        # extra pass over the byte code to determine all the jump targets.
        self.rtargets = {}
    
    def check_err(self,inverted=False):
        return [
            self.op.test(ops.eax,ops.eax),
            JumpSource(self.op.jnz if inverted else self.op.jz,self.end)]
    
    def local_name(self):
        if self._local_name is None:
            self._local_name = JumpTarget()
        return self._local_name
    
    def call(self,func):
        return [
            self.op.mov(pyinternals.raw_addresses[func],ops.eax),
            self.op.call(ops.eax)]

    def invoke(self,func,*args):
        if not args: return self.call(func)
        return list(map(self.op.push,reversed(args))) + self.call(func) + [self.op.add(len(args)*STACK_ITEM_SIZE,ops.esp)]

    def _if_eax_is(self,test,opcodes):
        if not isinstance(opcodes,list):
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

    def incref(self,reg = ops.eax):
        if pyinternals.ref_debug:
            return [self.op.push(ops.eax)] + self.invoke('Py_IncRef',reg) + [self.op.pop(ops.eax)]

        return [self.op.incl(ops.Address(pyinternals.refcnt_offset,reg))]

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
            self.op.decl(ops.Address(pyinternals.refcnt_offset,reg)),
            self.op.jnz(ops.Displacement(len(mid))),
            mid
        ]

    def rtarget(self):
        t = JumpRTarget()
        self.rtargets[self.byte_offset] = t
        return t

    def rtarget_at(self,offset):
        try:
            return self.rtargets[offset]
        except KeyError:
            raise Exception('unexpected jump target')

    def forward_target(self,at):
        assert at > self.byte_offset

        # there will rarely be more than two targets at any given time
        for i,ft in enumerate(self.forward_targets):
            if ft[0] == at:
                return ft[1]
            if ft[0] > at:
                t = JumpTarget()
                self.foward_targets.insert(i,(at,t))
                return t

        t = JumpTarget()
        self.forward_targets.append((at,t))
        return t

    def __call__(self):
        return Stitch(self)


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
    'call',
    'invoke',
    'if_eax_is_zero',
    'if_eax_is_not_zero',
    'incref',
    'decref']:
    setattr(Stitch,func,_forward_list_func(getattr(Frame,func)))



def _binary_op(f,func):
    return (
        f.stack.push_tos(True,True) + [
        f.op.push(ops.Address(STACK_ITEM_SIZE*2,ops.esp))
    ] + f.call(func) + [
        f.discard_stack_items(2)
    ] + f.check_err() + [
        f.stack.pop(ops.edx)
    ] + f.decref(ops.edx,True) + [
        f.stack.pop(ops.edx)
    ] + f.decref(ops.edx,True)
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
    return (
        f.stack.push_tos(True) + [
        f.op.push(address_of(name)),
        JumpSource(f.op.call,f.local_name()),
        f.op.test(ops.eax,ops.eax),
        JumpSource(f.op.jz,f.end)
    ] + f.incref()
    )

@hasname
def _op_STORE_NAME(f,name):
    mid = f.op.mov(pyinternals.raw_addresses['PyDict_SetItem'],ops.ecx)
    return (
        f.stack.push_tos(extra_push=True) + [
        f.op.push(address_of(name)),
        f.op.mov(pyinternals.raw_addresses['PyObject_SetItem'],ops.ecx),
        f.op.mov(LOCALS,ops.eax),
        f.op.push(ops.eax),
        f.op.cmpl(pyinternals.raw_addresses['PyDict_Type'],ops.Address(pyinternals.type_offset,ops.eax)),
        f.op.jne(ops.Displacement(len(mid))),
        mid,
        f.op.call(ops.ecx),
        f.discard_stack_items(3)
    ] + f.check_err(True) + [
        f.stack.pop(ops.eax)
    ] + f.decref()
    )

@hasname
def _op_LOAD_GLOBAL(f,name):
    return (
        f.stack.push_tos(True) + 
        f.invoke('PyDict_GetItem',GLOBALS,address_of(name)) + 
        f.if_eax_is_zero(
            call('PyDict_GetItem',BUILTINS,address_of(name)) + 
            f.if_eax_is_zero(
                f.invoke('format_exc_check_arg',
                     pyinternals.raw_addresses['PyExc_NameError'],
                     pyinternals.raw_addresses['GLOBAL_NAME_ERROR_MSG'],
                     address_of(name)) + [
                f.goto(f.end)
            ])
        ) +
        f.incref()
    )

@hasname
def _op_STORE_GLOBAL(f,name):
    return (
        f.stack.push_tos(extra_push=True) + [
        f.op.push(address_of(name)),
        f.op.push(GLOBALS)
    ] + f.call('PyDict_SetItem') + [
        f.discard_stack_items(3)
    ] + f.check_err(True) + [
        f.stack.pop(ops.eax)
    ] + f.decref()
    )

@hasconst
def _op_LOAD_CONST(f,const):
    return (
        f.stack.push_tos(True) + [
        f.op.mov(address_of(const),ops.eax),
    ] + f.incref()
    )

@handler
def _op_CALL_FUNCTION(f,arg):
    ret = f.op.ret()
    return (
        f.stack.push_tos(True) + [f.stack.push(arg)] + f.call('_call_function') +
        [f.stack.add_to(((arg & 0xFF) + ((arg >> 8) & 0xFF) + 2) * STACK_ITEM_SIZE)] +  # +2 for arg and the function object
        f.check_err()
    )

@handler
def _op_RETURN_VALUE(f):
    return [f.goto(f.end)] if f.stack.use_tos() else [f.stack.pop(ops.eax),f.goto(f.end)]

@handler
def _op_SETUP_LOOP(f,to):
    f.blockends.append(JumpTarget())
    return []

@handler
def _op_POP_BLOCK(f):
    assert not f.stack.tos_in_eax
    return ([
        f.blockends.pop(),
        f.stack.pop(ops.eax)
    ] + f.decref()
    )

@handler
def _op_GET_ITER(f):
    return (
        f.stack.push_tos(extra_push=True) +
        f.call('PyObject_GetIter') + [
        f.discard_stack_items(1)
    ] + f.check_err() + [
        f.op.pop(ops.edx),
        f.op.push(ops.eax)
    ] + f.decref(ops.edx)
    )

@handler
def _op_FOR_ITER(f,to):
    return (
        f.stack.push_tos(True) + [
        f.rtarget(),
        f.op.push(ops.Address(base=ops.esp)),
        f.op.mov(ops.Address(base=ops.esp),ops.eax),
        f.op.mov(ops.Address(pyinternals.type_offset,ops.eax),ops.eax),
        f.op.mov(ops.Address(pyinternals.type_iternext_offset,ops.eax),ops.eax),
        f.op.call(ops.eax),
        f.discard_stack_items(1)
    ] + f.if_eax_is_zero(
            f.call('PyErr_Occurred') +
            f.if_eax_is_not_zero(
                f.invoke('PyErr_ExceptionMatches',pyinternals.raw_addresses['PyExc_StopIteration']) + [
                f.op.test(ops.eax,ops.eax),
                JumpSource(f.op.jz,f.end)
            ] + f.call('PyErr_Clear')
            ) + [
            f.goto(f.blockends[-1])
        ])
    )

@handler
def _op_JUMP_ABSOLUTE(f,to):
    assert to < f.byte_offset
    return f.stack.push_tos() + [
        JumpRSource(f.op,f.op.jmp,ops.JMP_DISP_MAX_LEN,f.rtarget_at(to))]

@hasname
def _op_LOAD_ATTR(f,name):
    pn = f.op.push(address_of(name))
    return (
        (f().counted_push(ops.eax)(pn).push(ops.eax)
        if f.stack.use_tos() else
        f()(pn).push(ops.Address(STACK_ITEM_SIZE,ops.esp))) + f()
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
    return (f()
        .push_tos(extra_push=True)
        .call('PyObject_IsTrue')
        .discard_stack_items(1)
        .counted_pop(ops.edx)
        .decref(ops.edx,True)
        .test(ops.eax,ops.eax)
        (JumpSource(jop1,dont_jump))
        (JumpSource(jop2,f.forward_target(to)))
        .xor(ops.eax,ops.eax)
        .goto(f.end)
        (dont_jump)
    )

@handler
def _op_POP_JUMP_IF_FALSE(f,to):
    return _op_pop_jump_if_(f,to,False)

@handler
def _op_POP_JUMP_IF_TRUE(f,to):
    return _op_pop_jump_if_(f,to,True)



def join(x):
    try:
        return b''.join(x)
    except TypeError:
        return reduce(operator.add,x)

def resolve_jumps(chunks):
    targets = weakref.WeakSet()
    rsources = set()
    
    for i in range(len(chunks)-1,-1,-1):
        if isinstance(chunks[i],JumpTarget):
            chunks[i].displacement = 0
            targets.add(chunks[i])
            del chunks[i]
        elif isinstance(chunks[i],JumpRTarget):
            rsources.difference_update(chunks[i].sources)
            del chunks[i]
        else:
            if isinstance(chunks[i],JumpSource):
                chunks[i] = chunks[i].compile()
                if not chunks[i]:
                    del chunks[i]
                    continue
            elif isinstance(chunks[i],JumpRSource):
                chunks[i].displacement = 0
                rsources.add(chunks[i])
            
            l = len(chunks[i])
            for t in targets:
                t.displace(l)
            for s in rsources:
                s.displace(l)
    
    return join((c.compile() if isinstance(c,JumpRSource) else c) for c in chunks)


def local_name_func(f):
    # this function uses an ad hoc calling convention and is only callable from
    # the code generated by this module
    
    else_ = JumpTarget()
    endif = JumpTarget()
    found = JumpTarget()
    
    return ([
        f.op.push(ops.Address(STACK_ITEM_SIZE,ops.esp)),
        f.op.mov(LOCALS,ops.eax),
        f.op.push(ops.eax),
        
        # if (%eax)->ob_type != PyDict_Type:
        f.op.cmpl(pyinternals.raw_addresses['PyDict_Type'],ops.Address(pyinternals.type_offset,ops.eax)),
        JumpSource(f.op.je,else_)
    ] +     f.call('PyObject_GetItem') + [
            f.discard_stack_items(2),
            f.op.test(ops.eax,ops.eax),
            JumpSource(f.op.jnz,found),
            
        ] + f.invoke('PyErr_ExceptionMatches',pyinternals.raw_addresses['PyExc_KeyError']) +
            f.if_eax_is_zero(
                f.op.ret(STACK_ITEM_SIZE)
            ) +
            f.call('PyErr_Clear') + [

            f.goto(endif),
        else_
    ] +     f.call('PyDict_GetItem') + [
            f.discard_stack_items(2),
            f.op.test(ops.eax,ops.eax),
            JumpSource(f.op.jnz,found),
        endif
    
    ] + f.invoke('PyDict_GetItem',GLOBALS,ops.Address(STACK_ITEM_SIZE,ops.esp)) + 
        f.if_eax_is_zero(
            f.invoke('PyDict_GetItem',BUILTINS,ops.Address(STACK_ITEM_SIZE,ops.esp)) + 
            f.if_eax_is_zero(
                f.invoke('format_exc_check_arg',
                     pyinternals.raw_addresses['NAME_ERROR_MSG'],
                     pyinternals.raw_addresses['PyExc_NameError']) + [
                f.op.ret(STACK_ITEM_SIZE)
            ])
        ) + [
        
        found,
        f.op.ret(STACK_ITEM_SIZE),
    ])


def function_get(f,func):
    err = f.op.ret()
    if f.stack.offset: err = f.op.add(f.stack.offset,ops.esp) + err
    
    return (
        f.call(func) +
        f.if_eax_is_zero(err) + [
        f.stack.push(ops.eax)
    ])


def compile_raw(code,binary = True):
    f = Frame(code,ops if binary else ops.Assembly())
    
    opcodes = (f()
        .counted_push(ops.ebp)
        .mov(ops.esp,ops.ebp) +
        function_get(f,'PyEval_GetGlobals') +
        function_get(f,'PyEval_GetBuiltins') +
        function_get(f,'PyEval_GetLocals')
    )
    
    stack_prolog = f.stack.offset
    
    i = 0
    extended_arg = 0
    f.byte_offset = 0
    while i < len(code.co_code):
        bop = code.co_code[i]
        i += 1
        
        if bop >= dis.HAVE_ARGUMENT:
            boparg = code.co_code[i] + (code.co_code[i+1] << 8) + extended_arg
            i += 2
            
            if bop == dis.EXTENDED_ARG:
                extended_arg = bop << 16
            else:
                extended_arg = 0
                
                opcodes += get_handler(bop)(f,boparg)
                f.byte_offset = i
                #print('{}, {}'.format(dis.opname[bop],f.stack.offset))
        else:
            opcodes += get_handler(bop)(f)
            f.byte_offset = i
            #print('{}, {}'.format(dis.opname[bop],f.stack.offset))
    
    
    assert f.stack.offset == stack_prolog, 'stack.offset should be {0}, but is {1}'.format(stack_prolog,f.stack.offset)
    
    dr = join([f.op.pop(ops.edx)] + f.decref(ops.edx,True))
    
    cmpjl = f.op.cmp(ops.ebp,ops.esp)
    jlen = -(len(dr) + len(cmpjl) + ops.JCC_MIN_LEN)
    assert ops.fits_in_sbyte(jlen)
    cmpjl += f.op.jb(ops.Displacement(jlen))
    
    # call Py_DECREF on anything left on the stack and return %eax
    opcodes += [
        f.end,
        f.op.sub(stack_prolog - STACK_ITEM_SIZE,ops.ebp),
        f.op.jmp(ops.Displacement(len(dr))),
        dr,
        cmpjl,
        f.op.add(stack_prolog - STACK_ITEM_SIZE,ops.esp),
        f.op.pop(ops.ebp),
        f.op.ret()]
    
    if f._local_name:
        opcodes += [f._local_name] + local_name_func(f)
    
    return resolve_jumps(opcodes.code)


