

from .multimethod import multimethod


def int_to_32(x):
    return x.to_bytes(4,byteorder='little',signed=True)

def int_to_16(x):
    return x.to_bytes(2,byteorder='little',signed=True)

def int_to_8(x):
    return x.to_bytes(1,byteorder='little',signed=True)

def immediate_data(w,data):
    return data.to_bytes(4 if w else 1,byteorder='little',signed=data<0)

def fits_in_sbyte(x):
    return -128 <= x <= 127


def opcode_assembly(op,*args):
    return '{:12} {}\n'.format(op,', '.join(map(str,args)))


class Register:
    def __init__(self,w,reg,name):
        self.w = w
        self.reg = reg
        self.name = name
    
    def __str__(self):
        return '%' + self.name

class Address:
    def __init__(self,offset=0,base=None,index=None,scale=1):
        assert scale == 1 or scale == 2 or scale == 4 or scale == 8
        assert (base is None or base.w) and (index is None or index.w)
        
        # %esp cannot be used as the index
        assert index is None or index.reg != 0b100
        
        self.offset = offset
        self.base = base
        self.index = index
        self.scale = scale
    
    def _sib(self):
        return bytes([
            ({1:0,2:1,4:2,8:3}[self.scale] << 6) | 
            ((self.index.reg if self.index else 0b100) << 3) | 
            (self.base.reg if self.base else 0b101)])
    
    def mod_rm_sib_disp(self):
        """Get the mod field, the r/m field and the SIB and displacement bytes"""
        
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
        
    
    def __str__(self):
        if self.scale != 1:
            return '{0}({1},{2},{3})'.format(self.offset or '',self.base or '',self.index,self.scale)
        if self.index is not None:
            return '{0}({1},{2})'.format(self.offset or '',self.base or '',self.index)
        if self.base is not None:
            return '{0}({1})'.format(self.offset or '',self.base)
        return str(self.offset)

    def offset_only(self):
        return self.base is None and self.index is None and self.scale == 1


class Displacement:
    """A displacement relative to the next instruction.
    
    This class exists to make it clear that a given op code treats a value as a
    displacement and not an absolute address.
    
    """
    def __init__(self,value):
        self.val = value


class Test:
    def __init__(self,val):
        self.val = val
    
    def __invert__(self):
        return Test(self.val ^ 1)
    
    def __int__(self):
        return self.val


al = Register(0,0b000,'al')
cl = Register(0,0b001,'cl')
dl = Register(0,0b010,'dl')
bl = Register(0,0b011,'bl')
ah = Register(0,0b100,'ah')
ch = Register(0,0b101,'ch')
dh = Register(0,0b110,'dh')
bh = Register(0,0b111,'bh')
eax = Register(1,0b000,'eax')
ecx = Register(1,0b001,'ecx')
edx = Register(1,0b010,'edx')
ebx = Register(1,0b011,'ebx')
esp = Register(1,0b100,'esp')
ebp = Register(1,0b101,'ebp')
esi = Register(1,0b110,'esi')
edi = Register(1,0b111,'edi')



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





def op(op,*args,assembly=False):
    return opcode_assembly(op,*args) if assembly else globals()[op](*args)


def _op_reg_reg(byte1,a,b):
    assert a.w == b.w
    return bytes([
        byte1 | a.w,
        0b11000000 | (a.reg << 3) | b.reg])

def _op_addr_reg(byte1,a,b,reverse):
    mod,rm,extra = a.mod_rm_sib_disp()
    return bytes([
        byte1 | (reverse << 1) | b.w,
        (mod << 6) | (b.reg << 3) | rm]) + extra

def _op_imm_reg(byte1,mid,byte_alt,a,b):
    if b.reg == 0b000:
        return bytes([byte_alt | b.w]) + immediate_data(b.w,a)

    fits = fits_in_sbyte(a)
    return bytes([
        byte1 | (fits << 1) | b.w,
        0b11000000 | (mid << 3) | b.reg]) + immediate_data(not fits,a)

def _op_imm_addr(byte1,mid,a,b,w):
    fits = fits_in_sbyte(a)
    mod,rm,extra = b.mod_rm_sib_disp()
    return bytes([
        byte1 | (fits << 1) | w,
        (mod << 6) | (mid << 3) | rm]) + extra + immediate_data(not fits,a)



# Some functions are decorated with @multimethod even though they only have one
# version. This is to have consistent type-checking.


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

@multimethod
def addb(a : int,b : Address):
    return _op_imm_addr(0b10000000,0,a,b,False)

@multimethod
def addl(a : int,b : Address):
    return _op_imm_addr(0b10000000,0,a,b,True)



@multimethod
def call(proc : Displacement):
    return bytes([0b11101000]) + int_to_32(proc.val)

@multimethod
def call(proc : Register):
    assert proc.w
    return bytes([
        0b11111111,
        0b11010000 | proc.reg])

@multimethod
def call(proc : Address):
    mod,rm,extra = proc.mod_rm_sib_disp()
    return bytes([
        0b11111111,
        0b00010000 | (mod << 6) | rm]) + extra


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

@multimethod
def cmpb(a : int,b : Address):
    return _op_imm_addr(0b10000000,0b111,a,b,False)

@multimethod
def cmpl(a : int,b : Address):
    return _op_imm_addr(0b10000000,0b111,a,b,True)



@multimethod
def dec(x : Register):
    if x.w:
        return bytes([0b01001000 | x.reg])
    else:
        return bytes([
            0b11111110,
            0b11001000 | x.reg])

def dec_addr(x,w):
    mod,rm,extra = x.mod_rm_sib_disp()
    return bytes([
        0b11111110 | w,
        0b00001000 | (mod << 6) | rm]) + extra

@multimethod
def decb(x : Address):
    return dec_addr(x,False)

@multimethod
def decl(x : Address):
    return dec_addr(x,True)



@multimethod
def inc(x : Register):
    if x.w:
        return bytes([0b01000000 | x.reg])
    else:
        return bytes([
            0b11111110,
            0b11000000 | x.reg])

def inc_addr(x,w):
    mod,rm,extra = x.mod_rm_sib_disp()
    return bytes([
        0b11111110 | w,
        (mod << 6) | rm]) + extra

@multimethod
def incb(x : Address):
    return inc_addr(x,False)

@multimethod
def incl(x : Address):
    return inc_addr(x,True)



def jcc(test : Test,x : Displacement):
    if fits_in_sbyte(x.val):
        return bytes([0b01110000 | test.val]) + int_to_8(x.val)
    
    return bytes([
        0b00001111,
        0b10000000 | test.val]) + int_to_32(x.val)

def jo(x): return jcc(test_O,x)
def jno(x): return jcc(test_NO,x)
def jb(x): return jcc(test_B,x)
def jnb(x): return jcc(test_NB,x)
def je(x): return jcc(test_E,x)
jz = je
def jne(x): return jcc(test_NE,x)
jnz = jne
def jbe(x): return jcc(test_BE,x)
def ja(x): return jcc(test_A,x)
def js(x): return jcc(test_S,x)
def jns(x): return jcc(test_NS,x)
def jp(x): return jcc(test_P,x)
def jnp(x): return jcc(test_NP,x)
def jl(x): return jcc(test_L,x)
def jge(x): return jcc(test_GE,x)
def jle(x): return jcc(test_LE,x)
def jg(x): return jcc(test_G,x)



@multimethod
def jmp(x : Displacement):
    fits = fits_in_sbyte(x.val)
    return bytes([0b11101001 | (fits << 1)]) + ([int_to_32,int_to_8][fits])(x.val)

@multimethod
def jmp(x : Register):
    assert x.w
    return bytes([
        0b11111111,
        0b11100000 | x.reg])

@multimethod
def jmp(x : Address):
    mod,rm,extra = x.mod_rm_sib_disp()
    return bytes([
        0b11111111,
        0b00100000 | (mod << 6) | rm]) + extra



@multimethod
def mov(a : Register,b : Register):
    return _op_reg_reg(0b10001000,a,b)

def mov_addr_reg(a,b,forward):
    if b.reg == 0b000 and a.offset_only():
        return bytes([0b10100000 | (forward << 1) | b.w]) + int_to_32(a.offset)

    return _op_addr_reg(0b10001000,a,b,forward)

@multimethod
def mov(a : Address,b : Register):
    return mov_addr_reg(a,b,True)

@multimethod
def mov(a : Register,b : Address):
    return mov_addr_reg(b,a,False)

@multimethod
def mov(a : int,b : Register):
    return bytes([0b10110000 | (b.w << 3) | b.reg]) + immediate_data(b.w,a)

def mov_imm_addr(a,b,w):
    mod,rm,extra = b.mod_rm_sib_disp()
    return bytes([
        0b11000110 | w,
        (mod << 6) | rm]) + extra + immediate_data(w,a)

@multimethod
def movb(a : int,b : Address):
    return mov_imm_addr(a,b,False)

@multimethod
def movl(a : int,b : Address):
    return mov_imm_addr(a,b,True)



@multimethod
def pop(x : Register):
    return bytes([0b01011000 | x.reg])

@multimethod
def pop(x : Address):
    mod,rm,extra = x.mod_rm_sib_disp()
    return bytes([
        0b10001111,
        (mod << 6) | rm]) + extra



@multimethod
def push(x : Register):
    return bytes([0b01010000 | x.reg])

@multimethod
def push(x : Address):
    mod,rm,extra = x.mod_rm_sib_disp()
    return bytes([
        0b11111111,
        0b00110000 | (mod << 6) | rm]) + extra

@multimethod
def push(x : int):
    byte = fits_in_sbyte(x)
    return bytes([0b01101000 | (byte << 1)]) + immediate_data(not byte,x)



@multimethod
def ret():
    return bytes([0b11000011])

@multimethod
def ret(pop : int):
    return bytes([0b11000010]) + int_to_16(pop)



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

@multimethod
def subb(a : int,b : Address):
    return _op_imm_addr(0b10000000,0b101,a,b,False)

@multimethod
def subl(a : int,b : Address):
    return _op_imm_addr(0b10000000,0b101,a,b,True)



@multimethod
def test(a : Register,b : Register):
    return _op_reg_reg(0b10000100,a,b)

@multimethod
def test(a : Address,b : Register):
    return _op_addr_reg(0b10000100,a,b,True)

@multimethod
def test(a : int,b : Register):
    return _op_imm_reg(0b11110110,0,0b10101000,a,b)

def test_imm_addr(a,b,w):
    mode,rm,extra = b.mod_rm_sib_disp()
    return bytes([
        0b11110110 | w,
        (mod << 6) | rm]) + extra + immedate_data(w,a)

@multimethod
def testb(a : int,b : Address):
    return test_imm_addr(a,b,False)

@multimethod
def testl(a : int,b : Address):
    return test_imm_addr(a,b,True)

