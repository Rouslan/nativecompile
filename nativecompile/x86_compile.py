
import dis
import weakref
from functools import partial

from . import x86_ops as ops
from . import pyinternals


STACK_ITEM_SIZE = 4


def call(func):
    return [
        ops.mov(pyinternals.raw_addresses[func],ops.eax),
        ops.call(ops.eax)]


def invoke(func,*args):
    if not args: return call(func)
    return list(map(ops.push,reversed(args))) + call(func) + [ops.add(len(args)*STACK_ITEM_SIZE,ops.esp)]


def _if_eax_is(test,opcodes):
    if isinstance(opcodes,bytes):
        return [
            ops.test(ops.eax,ops.eax),
            ops.jcc(~test,ops.Displacement(len(opcodes))),
            opcodes]
            
    after = JumpTarget()
    return [
        ops.test(ops.eax,ops.eax),
        JumpSource(partial(ops.jcc,~test),after)
    ] + opcodes + [
        after
    ]

def if_eax_is_zero(opcodes): return _if_eax_is(ops.test_Z,opcodes)
def if_eax_is_not_zero(opcodes): return _if_eax_is(ops.test_NZ,opcodes)

def goto(target):
    return JumpSource(ops.jmp,target)

def discard_stack_items(n):
    return ops.add(STACK_ITEM_SIZE * n,ops.esp)


class StackManager:
    offset = 0 # the number of bytes the stack has moved
    tos_in_eax = False # if True, the TOS item is in %eax and has not actually been pushed onto the stack
    
    def push(self,x):
        self.offset += STACK_ITEM_SIZE
        return ops.push(x)
    
    def pop(self,x):
        self.offset -= STACK_ITEM_SIZE
        return ops.pop(x)
    
    def add_to(self,amount):
        self.offset -= amount
        return ops.add(amount,ops.esp)
    
    def sub_from(self,amount):
        self.offset += amount
        return ops.sub(amount,ops.esp)
    
    def push_tos(self,set_again = False):
        """%eax is needed right now so if the TOS item hasn't been pushed onto 
        the stack, do it now."""
        r = [self.push(ops.eax)] if self.tos_in_eax else []
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
        return self.op(dis) if dis.val or self.op is ops.call else b'' #optimize away useless jumps



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
    handlers[dis.opmap[(name or func.__name__)[len('_op_'):]]] = func
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
    def __init__(self,code):
        self.code = code
        self.stack = StackManager()
        self.end = JumpTarget()
        self._local_name = None
    
    def check_err(self):
        return [
            ops.test(ops.eax,ops.eax),
            JumpSource(ops.jz,self.end)]
    
    def local_name(self):
        if self._local_name is None:
            self._local_name = JumpTarget()
        return self._local_name



@handler
def _op_POP_TOP(f):
    return [] if f.stack.use_tos() else [f.stack.add_to(STACK_ITEM_SIZE)]

@hasname
def _op_LOAD_NAME(f,name):
    return (
        f.stack.push_tos(True) + [
        ops.push(address_of(name)),
        JumpSource(ops.call,f.local_name()),
        ops.test(ops.eax,ops.eax),
        JumpSource(ops.jz,f.end)
    ])

@hasname
def _op_LOAD_GLOBAL(f,name):
    return (
        f.stack.push_tos(True) + [
        ops.push(address_of(name)),
        ops.push(GLOBALS)
    ] + call('PyDict_GetItem') + 
        if_eax_is_zero([
            discard_stack_items(1),
            ops.push(BUILTINS)
        ] + call('PyDict_GetItem') + 
            if_eax_is_zero([
                discard_stack_items(1),
                ops.push(pyinternals.raw_addresses['GLOBAL_NAME_ERROR_MSG']),
                ops.push(pyinternals.raw_addresses['PyExc_NameError'])
            ] + call('format_exc_check_arg') + [
                goto(f.end)
            ])
        ) + [
        discard_stack_items(2)
    ])

@hasconst
def _op_LOAD_CONST(f,const):
    return f.stack.push_tos() + [f.stack.push(address_of(const))]

@handler
def _op_CALL_FUNCTION(f,arg):
    ret = ops.ret()
    return (
        f.stack.push_tos(True) + [f.stack.push(arg)] + call('_call_function') +
        [f.stack.add_to(((arg & 0xFF) + ((arg >> 8) & 0xFF) + 2) * STACK_ITEM_SIZE)] +  # +2 for arg and the function object
        f.check_err()
    )

@handler
def _op_RETURN_VALUE(f):
    return [goto(f.end)] if f.stack.use_tos() else [f.stack.pop(ops.eax),goto(f.end)]



def resolve_jumps(chunks):
    targets = weakref.WeakSet()
    
    for i in range(len(chunks)-1,-1,-1):
        if isinstance(chunks[i],JumpTarget):
            chunks[i].displacement = 0
            targets.add(chunks[i])
            del chunks[i]
        else:
            if isinstance(chunks[i],JumpSource):
                chunks[i] = chunks[i].compile()
            
            l = len(chunks[i])
            for t in targets:
                t.displace(l)
    
    return b''.join(chunks)


def local_name_func(f):
    # this function uses an ad hoc calling convention and is only callable from
    # the code generated by this module
    
    else_ = JumpTarget()
    endif = JumpTarget()
    found = JumpTarget()
    
    return ([
        ops.push(ops.Address(STACK_ITEM_SIZE,ops.esp)),
        ops.mov(LOCALS,ops.eax),
        ops.push(ops.eax),
        
        # if (%eax)->ob_type != PyDict_Type:
        ops.cmpl(pyinternals.raw_addresses['PyDict_Type'],ops.Address(pyinternals.type_offset,ops.eax)),
        JumpSource(ops.je,else_)
    ] +     call('PyObject_GetItem') +
            call('PyErr_Occurred') + [
            ops.test(ops.eax,ops.eax),
            JumpSource(ops.jnz,endif),
            
            ops.push(pyinternals.raw_addresses['PyExc_KeyError'])
        ] + call('PyErr_ExceptionMatches') +
            if_eax_is_zero(
                discard_stack_items(3) +
                ops.ret(STACK_ITEM_SIZE)
            ) + [
            discard_stack_items(1)
    ] +     call('PyErr_Clear') + [

            goto(endif),
        else_
    ] +     call('PyDict_GetItem') + [
            ops.test(ops.eax,ops.eax),
        endif,
    
        JumpSource(ops.jnz,found),
        discard_stack_items(1),
        ops.push(GLOBALS)
    ] + call('PyDict_GetItem') + 
        if_eax_is_zero([
            discard_stack_items(1),
            ops.push(BUILTINS)
        ] + call('PyDict_GetItem') + 
            if_eax_is_zero([
                discard_stack_items(1),
                ops.push(pyinternals.raw_addresses['NAME_ERROR_MSG']),
                ops.push(pyinternals.raw_addresses['PyExc_NameError'])
            ] + call('format_exc_check_arg') + [
                discard_stack_items(3),
                ops.ret(STACK_ITEM_SIZE)
            ])
        ) + [
        
        found,
        discard_stack_items(2),
        ops.ret(STACK_ITEM_SIZE),
    ])


def function_get(f,func):
    err = ops.ret()
    if f.stack.offset: err = ops.add(f.stack.offset,ops.esp) + err
    
    return (
        call(func) +
        if_eax_is_zero(err) + [
        f.stack.push(ops.eax)
    ])


def compile_raw(code):
    f = Frame(code)
    
    opcodes = ([
        f.stack.push(ops.ebp),
        ops.mov(ops.esp,ops.ebp)
    ] + function_get(f,'PyEval_GetGlobals') +
        function_get(f,'PyEval_GetBuiltins') +
        function_get(f,'PyEval_GetLocals')
    )
    
    stack_prolog = f.stack.offset
    
    i = 0
    extended_arg = 0
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
        else:
            opcodes += get_handler(bop)(f)
    
    
    assert f.stack.offset == stack_prolog, 'stack.offset should be {0}, but is {1}'.format(stack_prolog,f.stack.offset)
    
    opcodes += [
        f.end,
        #ops.add(f.stack.offset - STACK_ITEM_SIZE,ops.esp),
        ops.mov(ops.ebp,ops.esp), # get rid of whatever is on the stack
        ops.pop(ops.ebp),
        ops.ret()]
    
    if f._local_name:
        opcodes += [f._local_name] + local_name_func(f)
    
    return resolve_jumps(opcodes)


