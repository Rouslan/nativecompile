
# To avoid repeating code, most of the functions and classes defined in this
# module handle both 32 and 64 bit machine code, but this module should only be
# used for writing 32 bit code. Use x86_64_ops for correct 64 bit code.


import binascii
import copy
from functools import partial

from .multimethod import mmtype,multimethod


SIZE_B = 0 # SIZE_B must evaluate to False
# SIZE_W not used
SIZE_D = 1
SIZE_Q = 2

TEST_MNEMONICS = [('o',),('no',),('b',),('nb',),('e','z'),('ne','nz'),('be',),('a',),('s',),('ns',),('p',),('np',),('l',),('ge',),('le',),('g',)]

def int_to_32(x):
    return x.to_bytes(4,byteorder='little',signed=True)

def int_to_16(x):
    return x.to_bytes(2,byteorder='little',signed=True)

def int_to_8(x):
    return x.to_bytes(1,byteorder='little',signed=True)

def immediate_data(w,data):
    w = 4 if w else 1
    if isinstance(data,bytes):
        assert len(data) == w
        return data
    return data.to_bytes(w,byteorder='little',signed=data<0)



class Register:
    def __init__(self,size,code):
        assert size in (SIZE_B,SIZE_D,SIZE_Q) and isinstance(code,int)
        self.size = size
        self.code = code

    @property
    def ext(self):
        return bool(self.code & 0b1000)

    @property
    def reg(self):
        return self.code & 0b111

    @property
    def w(self):
        return bool(self.size)
    
    def __eq__(self,b):
        if isinstance(b,Register):
            return self.size == b.size and self.code == b.code
        
        return NotImplemented
    
    def __ne__(self,b):
        if isinstance(b,Register):
            return self.size != b.size or self.code != b.code
        
        return NotImplemented
    
    def __hash__(self):
        return self.code | (self.size << 4)

    @property
    def name(self):
        assert (not self.ext) and self.size < SIZE_Q
        return [
            ['al','cl','dl','bl','ah','ch','dh','bh'],
            ['eax','ecx','edx','ebx','esp','ebp','esi','edi']
        ][self.size][self.reg]
    
    def __repr__(self):
        return 'Register({},{})'.format(self.size,self.code)

    def __str__(self):
        return '%' + self.name

Register.__mmtype__ = Register


class Address:
    def __init__(self,offset=0,base=None,index=None,scale=1):
        assert scale in (1,2,4,8)
        assert (base is None or base.w) and (index is None or index.w)
        assert base is None or index is None or base.size == index.size
        
        # %esp cannot be used as the index
        assert index is None or index.reg != 0b100
        
        self.offset = offset
        self.base = base
        self.index = index
        self.scale = scale
    
    def __eq__(self,b):
        if isinstance(b,Address):
            return (self.offset == b.offset
                and self.base == b.base
                and self.index == b.index
                and self.scale == b.scale)
        
        return NotImplemented
    
    def __ne__(self,b):
        if isinstance(b,Address):
            return (self.offset != b.offset
                or self.base != b.base
                or self.index != b.index
                or self.scale != b.scale)
        
        return NotImplemented
    
    def __hash__(self):
        return self.offset ^ hash(self.base) ^ hash(self.index) ^ self.scale

    @property
    def size(self):
        if self.base: return self.base.size
        if self.index: return self.index.size
        return None
    
    def _sib(self):
        r = (((self.index.reg if self.index else 0b100) << 3) | 
            (self.base.reg if self.base else 0b101))
        if self.index:
            r |= {1:0,2:1<<6,4:2<<6,8:3<<6}[self.scale]
        return bytes([r])
    
    def _mod_rm_sib_disp(self):
        if self.index or (self.base and self.base.reg == 0b100):
            # The opcode format is a little different when the base is 0b101 and 
            # mod is 0b00 so to use %ebp, we'll have to go with the next mod value.
            if self.offset == 0 and not (self.base and self.base.reg == 0b101):
                return 0b00, 0b100, self._sib()
                
            if fits_in_sbyte(self.offset):
                return 0b01, 0b100, self._sib() + int_to_8(self.offset)
                
            return 0b10, 0b100, self._sib() + int_to_32(self.offset)


        if self.base is None:
            return 0b00, 0b101, int_to_32(self.offset)
        
        # The opcode format is a little different when the base is 0b101 and 
        # mod is 0b00 so to use %ebp, we'll have to go with the next mod value.
        if self.offset == 0 and not (self.base and self.base.reg == 0b101):
            return 0b00, self.base.reg, b''
        
        if fits_in_sbyte(self.offset):
            return 0b01, self.base.reg, int_to_8(self.offset)
        
        return 0b10, self.base.reg, int_to_32(self.offset)
    
    def mod_rm_sib_disp(self,mid):
        """Get the mod field, the r/m field and the SIB and displacement bytes"""
    
        mod,rm,extra = self._mod_rm_sib_disp()
        return bytes([(mod << 6) | (mid << 3) | rm]) + extra
    
    def __str__(self):
        if self.index is not None:
            if self.scale != 1:
                return '{}({},{},{})'.format(hex(self.offset) if self.offset else '',self.base or '',self.index,self.scale)
 
            return '{}({},{})'.format(hex(self.offset) if self.offset else '',self.base or '',self.index)
        if self.base is not None:
            return '{}({})'.format(hex(self.offset) if self.offset else '',self.base)
        return hex(self.offset)

    def offset_only(self):
        return self.base is None and self.index is None and self.scale == 1

    def __add__(self,b):
        # this needs to work correctly for derived classes
        if isinstance(b,int):
            r = copy.copy(self)
            r.offset += b
            return r

        return NotImplemented

    def __sub__(self,b): return self.__add__(-b)

    def __iadd__(self,b):
        if isinstance(b,int):
            self.offset += b
            return self
        
        return NotImplemented

    def __isub__(self,b): return self.__iadd__(-b)

Address.__mmtype__ = Address


class Displacement:
    """A displacement relative to the next instruction.
    
    This class exists to make it clear that a given op code treats a value as
    a displacement and not an absolute address.
    
    """
    def __init__(self,value,force_full_size=False):
        self.val = value
        self.force_full_size = force_full_size
    
    def __eq__(self,b):
        if isinstance(b,Displacement):
            return self.val == b.val and self.force_full_size == b.force_full_size
    
    def __ne__(self,b):
        if isinstance(b,Displacement):
            return self.val != b.val or self.force_full_size != b.force_full_size
    
    def __hash__(self):
        return self.val ^ self.force_full_size


def fits_in_sbyte(x):
    if isinstance(x,Displacement):
        if x.force_full_size: return False
        x = x.val
    return -0x80 <= x <= 0x7f

def fits_in_sdword(x):
    return -0x80000000 <= x <= 0x7fffffff


def rex(reg,rm,need_w=True):
    """Return the REX prefix byte if needed.

    This is only needed for 64-bit code. If given only 32-bit compatible
    registers and addresses, this function will always return an empty byte
    string.

    """
    rxb = 0
    w = None

    if isinstance(reg,Register):
        w = reg.size == SIZE_Q
        rxb |= reg.ext << 2
    elif reg is not None:
        assert isinstance(reg,int) and isinstance(rm,Address)
        w = reg == SIZE_Q
    else:
        assert isinstance(rm,Register)

    if isinstance(rm,Address):
        if rm.index and rm.index.ext: rxb |= 0b10
        if rm.base and rm.base.ext: rxb |= 1
    elif rm is not None:
        assert isinstance(rm,Register)
        assert w is None or w == (rm.size == SIZE_Q)
        w = rm.size == SIZE_Q
        rxb |= rm.ext

    return bytes([0b01000000 | (w << 3) | rxb]) if ((w and need_w) or rxb) else b''



class Test:
    def __init__(self,val):
        self.val = val
    
    def __invert__(self):
        return Test(self.val ^ 1)
    
    def __int__(self):
        return self.val

    def __eq__(self,b):
        if isinstance(b,Test):
            return self.val == b.val
        
        return NotImplemented
    
    def __ne__(self,b):
        if isinstance(b,Test):
            return self.val != b.val
        
        return NotImplemented
    
    def __hash__(self):
        return self.val
    
    @property
    def mnemonic(self):
        return TEST_MNEMONICS[self.val][0]
    
    def __repr__(self):
        return 'Test({})'.format(self.val)




def asm_str(indirect,nextaddr,x):
    if isinstance(x,int):
        return '${:#x}'.format(x)
    if isinstance(x,Displacement):
        return '{:x}'.format(x.val + nextaddr)
    if indirect and isinstance(x,(Address,Register)):
        return '*'+str(x)
    return str(x)

class AsmOp:
    __slots__ = ('binary','name','args','annot')
    
    def __init__(self,binary,name,args,annot=''):
        self.binary = binary
        self.name = name
        self.args = args
        self.annot = annot

class AsmComment:
    __slots__ = ('message',)
    
    def __init__(self,message):
        self.message = message

class AsmSequence:
    def __init__(self,ops=None):
        self.ops = ops or []
    
    def __len__(self):
        return sum(len(op.binary) for op in self.ops if type(op) is AsmOp)
    
    def __add__(self,b):
        if isinstance(b,AsmSequence):
            return AsmSequence(self.ops+b.ops)
            
        return NotImplemented
    
    def __iadd__(self,b):
        if isinstance(b,AsmSequence):
            self.ops += b.ops
            return self
        
        return NotImplemented

    def __mul__(self,b):
        if isinstance(b,int):
            return AsmSequence(self.ops*b)

        return NotImplemented

    def __imul__(self,b):
        if isinstance(b,int):
            self.ops *= b
            return self

        return NotImplemented
    
    def dump(self,base=0):
        lines = []
        addr = base
        for op in self.ops:
            if isinstance(op,AsmComment):
                lines.append('; {}\n'.format(op.message))
            else:
                indirect = op.name == 'call' or op.name == 'jmp'
                nextaddr = addr + len(op.binary)
            
                lines.append('{:8x}: {:20}{:8} {}{}\n'.format(
                    addr,
                    binascii.hexlify(op.binary).decode(),
                    op.name,
                    ', '.join(asm_str(indirect,nextaddr,arg) for arg in op.args),
                    op.annot and '  ; '+op.annot))
                
                addr = nextaddr
        
        return ''.join(lines)


class Assembly:
    @staticmethod
    def namespace():
        return globals()
    
    def op(self,name,*args):
        return AsmSequence([AsmOp(self.namespace()[name](*args),name,args)])

    def __getattr__(self,name):
        b = self.namespace()[name]
        if not callable(b): return b
        return partial(self.op,name)
    
    def jcc(self,test,x):
        return AsmSequence([AsmOp(self.namespace()['jcc'](test,x),'j'+test.mnemonic,(x,))])

    def comment(self,comm):
        return AsmSequence([AsmComment(comm)])

    def with_new_imm_dword(self,op,imm):
        assert len(op.ops) == 1
        
        old = op.ops[0]
        
        assert isinstance(old,AsmOp) and len(old.args) == 2
        
        return AsmSequence([AsmOp(with_new_imm_dword(old.binary,imm),old.name,(imm,old.args[1]),old.annot)])



al = Register(SIZE_B,0b000)
cl = Register(SIZE_B,0b001)
dl = Register(SIZE_B,0b010)
bl = Register(SIZE_B,0b011)
ah = Register(SIZE_B,0b100)
ch = Register(SIZE_B,0b101)
dh = Register(SIZE_B,0b110)
bh = Register(SIZE_B,0b111)
eax = Register(SIZE_D,0b000)
ecx = Register(SIZE_D,0b001)
edx = Register(SIZE_D,0b010)
ebx = Register(SIZE_D,0b011)
esp = Register(SIZE_D,0b100)
ebp = Register(SIZE_D,0b101)
esi = Register(SIZE_D,0b110)
edi = Register(SIZE_D,0b111)



test_O = Test(0b0000)
test_NO = Test(0b0001)
test_B = Test(0b0010)
test_NB = Test(0b0011)
test_E = Test(0b0100)

test_Z = test_E

test_NE = Test(0b0101)

test_NZ = test_NE

test_BE = Test(0b0110)
test_A = Test(0b0111)
test_S = Test(0b1000)
test_NS = Test(0b1001)
test_P = Test(0b1010)
test_NP = Test(0b1011)
test_L = Test(0b1100)
test_GE = Test(0b1101)
test_LE = Test(0b1110)
test_G = Test(0b1111)



def with_new_imm_dword(op,imm):
    """Returns the same instruction as 'op' but with the immediate value 'imm'
    instead.

    The immediate value is assumed to be 4 bytes long. No checking is done to
    make sure 'op' has an immediate value or if it's the correct size.

    """
    return op[0:-4] + immediate_data(True,imm)



def _op_reg_reg(byte1,a,b):
    assert a.size == b.size
    return rex(a,b) + bytes([
        byte1 | a.w,
        0b11000000 | (a.reg << 3) | b.reg])

def _op_addr_reg(byte1,a,b,reverse):
    return rex(b,a) + bytes([byte1 | (reverse << 1) | b.w]) + a.mod_rm_sib_disp(b.reg)

def _op_imm_reg(byte1,mid,byte_alt,a,b):
    if b.code == 0:
        r = bytes([byte_alt | b.w]) + immediate_data(b.w,a)
        if b.size == SIZE_Q: r = b'\x48' + r
        return r

    fits = fits_in_sbyte(a)
    return rex(None,b) + bytes([
        byte1 | (fits << 1) | b.w,
        0b11000000 | (mid << 3) | b.reg]) + immediate_data(not fits,a)

def _op_imm_addr(byte1,mid,a,b,size):
    fits = fits_in_sbyte(a)
    return rex(size,b) + bytes([byte1 | (fits << 1) | bool(size)]) + b.mod_rm_sib_disp(mid) + immediate_data(not fits,a)



# Some functions are decorated with @multimethod even though they only have
# one version. This is to have consistent type-checking.


@multimethod
def add(a : Register,b : Register):
    return _op_reg_reg(0,a,b)

@multimethod
def add(a : Address,b : Register):
    return _op_addr_reg(0,a,b,True)

@multimethod
def add(a : Register,b : Address):
    return _op_addr_reg(0,b,a,False)

@multimethod
def add(a : int,b : Register):
    return _op_imm_reg(0b10000000,0,0b00000100,a,b)

def add_imm_addr(a,b,size):
    return _op_imm_addr(0b10000000,0,a,b,size)

@multimethod
def addb(a : int,b : Address):
    return add_imm_addr(a,b,SIZE_B)

@multimethod
def addl(a : int,b : Address):
    return add_imm_addr(a,b,SIZE_D)



@multimethod
def call(proc : Displacement):
    return b'\xE8' + int_to_32(proc.val)

CALL_DISP_LEN = 5

@multimethod
def call(proc : Register):
    assert proc.w
    return rex(None,proc,False) + bytes([
        0b11111111,
        0b11010000 | proc.reg])

@multimethod
def call(proc : Address):
    return rex(SIZE_D,proc,False) + b'\xFF' + proc.mod_rm_sib_disp(0b010)



@multimethod
def cmovcc(test : Test,a : Register,b : Register):
    assert a.w and b.w

    # unlike most, this instruction moves R/M to Reg instead of Reg to R/M
    # (lookup the Intel machine instruction format for the nomenclature)
    return rex(b,a) + bytes([
        0b00001111,
        0b01000000 | test.val,
        0b11000000 | (b.reg << 3) | a.reg])

@multimethod
def cmovcc(test : Test,a : Address,b : Register):
    assert b.w
    return rex(b,a) + bytes([0b00001111,0b01000000 | test.val]) + a.mod_rm_sib_disp(b.reg)



@multimethod
def cmp(a : Register,b : Register):
    return _op_reg_reg(0b00111000,a,b)

@multimethod
def cmp(a : Address,b : Register):
    return _op_addr_reg(0b00111000,a,b,True)

@multimethod
def cmp(a : Register,b : Address):
    return _op_addr_reg(0b00111000,b,a,False)

@multimethod
def cmp(a : int,b : Register):
    return _op_imm_reg(0b10000000,0b111,0b00111100,a,b)

def cmp_imm_addr(a,b,size):
    return _op_imm_addr(0b10000000,0b111,a,b,size)

@multimethod
def cmpb(a : int,b : Address):
    return cmp_imm_addr(a,b,SIZE_B)

@multimethod
def cmpl(a : int,b : Address):
    return cmp_imm_addr(a,b,SIZE_D)



def cpuid():
    return b'\x0F\xA2'



@multimethod
def dec(x : Register):
    # REX omitted; dec is redefined in x86_64_ops
    if x.w:
        return bytes([0b01001000 | x.reg])
    else:
        return bytes([
            0b11111110,
            0b11001000 | x.reg])

def dec_addr(x,size):
    return rex(size,x) + bytes([0b11111110 | bool(size)]) + x.mod_rm_sib_disp(0b001)

@multimethod
def decb(x : Address):
    return dec_addr(x,SIZE_B)

@multimethod
def decl(x : Address):
    return dec_addr(x,SIZE_D)



@multimethod
def inc(x : Register):
    # REX omitted; inc is redefined in x86_64_ops
    if x.w:
        return bytes([0b01000000 | x.reg])
    else:
        return bytes([
            0b11111110,
            0b11000000 | x.reg])

def inc_addr(x,size):
    return rex(size,x) + bytes([0b11111110 | bool(size)]) + x.mod_rm_sib_disp(0)

@multimethod
def incb(x : Address):
    return inc_addr(x,SIZE_B)

@multimethod
def incl(x : Address):
    return inc_addr(x,SIZE_D)



def jcc(test : Test,x : Displacement):
    if fits_in_sbyte(x):
        return bytes([0b01110000 | test.val]) + int_to_8(x.val)
    
    return bytes([
        0b00001111,
        0b10000000 | test.val]) + int_to_32(x.val)

JCC_MIN_LEN = 2
JCC_MAX_LEN = 6



@multimethod
def jmp(x : Displacement):
    fits = fits_in_sbyte(x)
    return bytes([0b11101001 | (fits << 1)]) + ([int_to_32,int_to_8][fits])(x.val)

JMP_DISP_MIN_LEN = 2
JMP_DISP_MAX_LEN = 5

@multimethod
def jmp(x : Register):
    assert x.w
    return rex(None,x) + bytes([
        0b11111111,
        0b11100000 | x.reg])

@multimethod
def jmp(x : Address):
    return rex(None,x) + b'\xFF' + x.mod_rm_sib_disp(0b100)



@multimethod
def lea(a : Address,b : Register):
    assert b.w
    return rex(b,a) + b'\x8D' + a.mod_rm_sib_disp(b.reg)



def leave():
    return b'\xC9'



@multimethod
def loop(x : Displacement):
    return b'\xE2' + int_to_8(x.val)

@multimethod
def loopz(x : Displacement):
    return b'\xE1' + int_to_8(x.val)

loope = loopz
    
@multimethod
def loopnz(x : Displacement):
    return b'\xE0' + int_to_8(x.val)

loopne = loopnz

LOOP_LEN = 2



@multimethod
def mov(a : Register,b : Register):
    return _op_reg_reg(0b10001000,a,b)

def mov_addr_reg(a,b,forward):
    if b.reg == 0b000 and a.offset_only():
        r = bytes([0b10100000 | (forward << 1) | b.w]) + int_to_32(a.offset)
        if b.size == SIZE_Q: r = b'\x48' + r
        return r

    return _op_addr_reg(0b10001000,a,b,forward)

@multimethod
def mov(a : Address,b : Register):
    return mov_addr_reg(a,b,True)

@multimethod
def mov(a : Register,b : Address):
    return mov_addr_reg(b,a,False)

@multimethod
def mov(a : int,b : Register):
    return rex(None,b) + bytes([0b10110000 | (b.w << 3) | b.reg]) + immediate_data(b.w,a)

def mov_imm_addr(a,b,size):
    return rex(size,b) + bytes([0b11000110 | bool(size)]) + b.mod_rm_sib_disp(0) + immediate_data(bool(size),a)

def movb(a,b):
    assert mmtype(b.__class__) is Address
    return mov_imm_addr(a,b,SIZE_B)

def movl(a,b):
    assert mmtype(b.__class__) is Address
    return mov_imm_addr(a,b,SIZE_D)



def nop():
    return b'\x90'



@multimethod
def not_(x : Register):
    return rex(None,x) + bytes([0b11110110 | x.w, 0b11010000 | x.reg])

def not_addr(x,size):
    return rex(size,x) + bytes([0b11110110 | bool(size)]) + x.mod_rm_sib_disp(0b010)

@multimethod
def notb(x : Address):
    return not_addr(x,SIZE_B)

@multimethod
def notl(x : Address):
    return not_addr(x,SIZE_D)



@multimethod
def pop(x : Register):
    return rex(None,x,False) + bytes([0b01011000 | x.reg])

@multimethod
def pop(x : Address):
    return rex(None,x,False) + b'\x8F' + x.mod_rm_sib_disp(0)



@multimethod
def push(x : Register):
    return rex(None,x,False) + bytes([0b01010000 | x.reg])

@multimethod
def push(x : Address):
    return rex(None,x,False) + b'\xFF' + x.mod_rm_sib_disp(0b110)

@multimethod
def push(x : int):
    byte = fits_in_sbyte(x)
    return bytes([0b01101000 | (byte << 1)]) + immediate_data(not byte,x)



@multimethod
def ret():
    return b'\xC3'

@multimethod
def ret(pop : int):
    return b'\xC2' + int_to_16(pop)



def shx_imm_reg(amount,x,shiftright):
    r = rex(None,x)

    if amount == 1:
        return r + bytes([
            0b11010000 | x.w,
            0b11100000 | (shiftright << 3) | x.reg])
    
    return r + bytes([
        0b11000000 | x.w,
        0b11100000 | (shiftright << 3) | x.reg]) + immediate_data(False,amount)

def shx_reg_reg(amount,x,shiftright):
    assert amount == cl
    
    return rex(None,x) + bytes([
        0b11010010 | x.w,
        0b11100000 | (shiftright << 3) | x.reg])

def shx_imm_addr(amount,x,size,shiftright):
    r = rex(size,x)
    rmsd = x.mod_rm_sib_disp(0b100 | shiftright)
    if amount == 1:
        return r + bytes([0b11010000 | bool(size)]) + rmsd
    
    return r + bytes([0b11000000 | bool(size)]) + rmsd + immediate_data(False,amount)

def shx_reg_addr(amount,x,size,shiftright):
    assert amount == cl
    
    return rex(size,x) + bytes([0b11010010 | bool(size)]) + x.mod_rm_sib_disp(0b100 | shiftright)


@multimethod
def shl(amount : int,x : Register):
    return shx_imm_reg(amount,x,False)

@multimethod
def shl(amount : Register,x : Register):
    return shx_reg_reg(amount,x,False)

@multimethod
def shlb(amount : int,x : Address):
    return shx_imm_addr(amount,x,SIZE_B,False)

@multimethod
def shll(amount : int,x : Address):
    return shx_imm_addr(amount,x,SIZE_D,False)

@multimethod
def shlb(amount : Register,x : Address):
    return shx_reg_addr(amount,x,SIZE_B,False)

@multimethod
def shll(amount : Register,x : Address):
    return shx_reg_addr(amount,x,SIZE_D,False)



@multimethod
def shr(amount : int,x : Register):
    return shx_imm_reg(amount,x,True)

@multimethod
def shr(amount : Register,x : Register):
    return shx_reg_reg(amount,x,True)

@multimethod
def shrb(amount : int,x : Address):
    return shx_imm_addr(amount,x,SIZE_B,True)

@multimethod
def shrl(amount : int,x : Address):
    return shx_imm_addr(amount,x,SIZE_D,True)

@multimethod
def shrb(amount : Register,x : Address):
    return shx_reg_addr(amount,x,SIZE_B,True)

@multimethod
def shrl(amount : Register,x : Address):
    return shx_reg_addr(amount,x,SIZE_D,True)



@multimethod
def sub(a : Register,b : Register):
    return _op_reg_reg(0b00101000,a,b)

@multimethod
def sub(a : Address,b : Register):
    return _op_addr_reg(0b00101000,a,b,True)

@multimethod
def sub(a : Register,b : Address):
    return _op_addr_reg(0b00101000,b,a,False)

@multimethod
def sub(a : int,b : Register):
    return _op_imm_reg(0b10000000,0b101,0b00101100,a,b)

def sub_imm_addr(a,b,size):
    return _op_imm_addr(0b10000000,0b101,a,b,size)

@multimethod
def subb(a : int,b : Address):
    return sub_imm_addr(a,b,SIZE_B)

@multimethod
def subl(a : int,b : Address):
    return sub_imm_addr(a,b,SIZE_D)



@multimethod
def test(a : Register,b : Register):
    return _op_reg_reg(0b10000100,a,b)

@multimethod
def test(a : Address,b : Register):
    return _op_addr_reg(0b10000100,a,b,True)

@multimethod
def test(a : int,b : Register):
    return _op_imm_reg(0b11110110,0,0b10101000,a,b)

def test_imm_addr(a,b,size):
    return rex(size,b) + bytes([0b11110110 | bool(size)]) + b.mod_rm_sib_disp(0) + immediate_data(bool(size),a)

@multimethod
def testb(a : int,b : Address):
    return test_imm_addr(a,b,SIZE_B)

@multimethod
def testl(a : int,b : Address):
    return test_imm_addr(a,b,SIZE_D)



@multimethod
def xor(a : Register,b : Register):
    return _op_reg_reg(0b00110000,a,b)

@multimethod
def xor(a : Register,b : Address):
    return _op_addr_reg(0b00110000,b,a,True)

@multimethod
def xor(a : Address,b : Register):
    return _op_addr_reg(0b00110000,a,b,False)

@multimethod
def xor(a : int,b : Register):
    return _op_imm_reg(0b10000000,0b110,0b00110100,a,b)

def xor_imm_addr(a,b,size):
    return _op_imm_addr(0b10000000,0b110,a,b,size)

@multimethod
def xorb(a : int,b : Address):
    return xor_imm_addr(a,b,SIZE_B)

@multimethod
def xorl(a : int,b : Address):
    return xor_imm_addr(a,b,SIZE_D)
