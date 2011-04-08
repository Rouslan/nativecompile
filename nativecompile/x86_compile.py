
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
        return self.op(dis) if dis.val else b'' #optimize away useless jumps



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
    def __init__(self,code,op):
        self.code = code
        self.op = op
        self.stack = StackManager(op)
        self.end = JumpTarget()
        self._local_name = None
    
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




def _binary_op(f,func):
    return (
        f.stack.push_tos(True) + [
        
        # reverse the order of the top two items
        # This could be avoided by reordering the bytecode instructions. I'm not
        # sure if it would be worth the extra time and memory needed to do that.
        f.op.push(ops.Address(STACK_ITEM_SIZE,ops.esp))
        
    ] + f.call(func) + [
        f.op.add(STACK_ITEM_SIZE,ops.esp)
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
        f.stack.push_tos() + [
        f.stack.push(address_of(name)),
        
        f.op.mov(pyinternals.raw_addresses['PyObject_SetItem'],ops.ecx),
        
        f.op.mov(LOCALS,ops.eax),
        f.stack.push(ops.eax),
        f.op.cmpl(pyinternals.raw_addresses['PyDict_Type'],ops.Address(pyinternals.type_offset,ops.eax)),
        f.op.jne(ops.Displacement(len(mid))),
        mid,
        f.op.call(ops.ecx),
        f.stack.add_to(2 * STACK_ITEM_SIZE)
    ] + f.check_err(True) + [
        f.stack.pop(ops.eax)
    ] + f.decref()
    )

@hasname
def _op_LOAD_GLOBAL(f,name):
    return (
        f.stack.push_tos(True) + [
        f.op.push(address_of(name)),
        f.op.push(GLOBALS)
    ] + f.call('PyDict_GetItem') + 
        f.if_eax_is_zero([
            f.discard_stack_items(1),
            f.op.push(BUILTINS)
        ] + call('PyDict_GetItem') + 
            f.if_eax_is_zero([
                f.discard_stack_items(1),
                f.op.push(pyinternals.raw_addresses['GLOBAL_NAME_ERROR_MSG']),
                f.op.push(pyinternals.raw_addresses['PyExc_NameError'])
            ] + f.call('format_exc_check_arg') + [
                f.discard_stack_items(2),
                f.goto(f.end)
            ])
        ) + [
        f.discard_stack_items(2)
    ] + f.incref()
    )

@hasname
def _op_STORE_GLOBAL(f,name):
    return (
        f.stack.push_tos() + [
        f.stack.push(address_of(name)),
        f.stack.push(GLOBALS),
    ] + f.call('PyDict_SetItem') +
        f.check_err(True) + [
        f.stack.add_to(2 * STACK_ITEM_SIZE),
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



def join(x):
    try:
        return reduce(operator.concat,x)
    except TypeError:
        # because there is no way to overload the concat operator in Python code
        return reduce(operator.add,x)

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
                if not chunks[i]:
                    del chunks[i]
                    continue
            
            l = len(chunks[i])
            for t in targets:
                t.displace(l)
    
    return join(chunks)


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
    ] +     f.call('PyObject_GetItem') +
            f.call('PyErr_Occurred') + [
            f.op.test(ops.eax,ops.eax),
            JumpSource(f.op.jnz,endif),
            
            f.op.push(pyinternals.raw_addresses['PyExc_KeyError'])
        ] + f.call('PyErr_ExceptionMatches') +
            f.if_eax_is_zero(
                f.discard_stack_items(3) +
                f.op.ret(STACK_ITEM_SIZE)
            ) + [
            f.discard_stack_items(1)
    ] +     f.call('PyErr_Clear') + [

            f.goto(endif),
        else_
    ] +     f.call('PyDict_GetItem') + [
            f.op.test(ops.eax,ops.eax),
        endif,
    
        JumpSource(f.op.jnz,found),
        f.discard_stack_items(1),
        f.op.push(GLOBALS)
    ] + f.call('PyDict_GetItem') + 
        f.if_eax_is_zero([
            f.discard_stack_items(1),
            f.op.push(BUILTINS)
        ] + f.call('PyDict_GetItem') + 
            f.if_eax_is_zero([
                f.discard_stack_items(1),
                f.op.push(pyinternals.raw_addresses['NAME_ERROR_MSG']),
                f.op.push(pyinternals.raw_addresses['PyExc_NameError'])
            ] + f.call('format_exc_check_arg') + [
                f.discard_stack_items(3),
                f.op.ret(STACK_ITEM_SIZE)
            ])
        ) + [
        
        found,
        f.discard_stack_items(2),
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
    
    opcodes = ([
        f.stack.push(ops.ebp),
        f.op.mov(ops.esp,ops.ebp)
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
    
    dr = join([f.op.pop(ops.edx)] + f.decref(ops.edx,True))
    
    cmpjl = f.op.cmp(ops.esp,ops.ebp)
    jlen = -(len(dr) + len(cmpjl) + ops.jcc.min_len)
    #assert ops.fits_in_sbyte(jlen)
    cmpjl += f.op.jl(ops.Displacement(jlen))
    
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
    
    return resolve_jumps(opcodes)


