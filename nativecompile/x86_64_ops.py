
from . multimethod import multimethod
from . import x86_ops
from .x86_ops import (
    TEST_MNEMONICS,Displacement,Test,AsmSequence,al,cl,dl,bl,eax,ecx,edx,ebx,
    esp,ebp,esi,edi,test_O,test_NO,test_B,test_NB,test_E,test_Z,test_NE,test_NZ,
    test_BE,test_A,test_S,test_NS,test_P,test_NP,test_L,test_GE,test_LE,test_G,
    add,addb,addl,cmp,cmpb,cmpl,decb,decl,incb,incl,jcc,JCC_MIN_LEN,JCC_MAX_LEN,
    jmp,lea,leave,loop,loopz,loope,loopnz,loopne,mov,movb,movl,nop,pop,ret,shl,
    shlb,shll,shr,shrb,shrl,sub,subb,subl,test,testb,testl,xor,xorb,xorl,
    CALL_DISP_LEN,JMP_DISP_MIN_LEN,JMP_DISP_MAX_LEN,LOOP_LEN,SIZE_B,SIZE_D,
    SIZE_Q,with_new_imm_dword
)



def immediate_data(w,data):
    return data.to_bytes(8 if w else 1,byteorder='little',signed=data<0)


class Register(x86_ops.Register):
    def __str__(self):
        return '%' + [
            ['al','cl','dl','bl','spl','bpl','sil','dil',
             'r8b','r9b','r10b','r11b','r12b','r13b','r14b','r15b'],
            ['eax','ecx','edx','ebx','esp','ebp','esi','edi',
             'r8d','r9d','r10d','r11d','r12d','r13d','r14d','r15d'],
            ['rax','rcx','rdx','rbx','rsp','rbp','rsi','rdi',
             'r8','r9','r10','r11','r12','r13','r14','r15']
        ][self.size][self.code]


spl = x86_ops.ah
bpl = x86_ops.ch
sil = x86_ops.dh
dil = x86_ops.bh
r8b = Register(SIZE_B,0b1000)
r9b = Register(SIZE_B,0b1001)
r10b = Register(SIZE_B,0b1010)
r11b = Register(SIZE_B,0b1011)
r12b = Register(SIZE_B,0b1100)
r13b = Register(SIZE_B,0b1101)
r14b = Register(SIZE_B,0b1110)
r15b = Register(SIZE_B,0b1111)

r8d = Register(SIZE_D,0b1000)
r9d = Register(SIZE_D,0b1001)
r10d = Register(SIZE_D,0b1010)
r11d = Register(SIZE_D,0b1011)
r12d = Register(SIZE_D,0b1100)
r13d = Register(SIZE_D,0b1101)
r14d = Register(SIZE_D,0b1110)
r15d = Register(SIZE_D,0b1111)

rax = Register(SIZE_Q,0b0000)
rcx = Register(SIZE_Q,0b0001)
rdx = Register(SIZE_Q,0b0010)
rbx = Register(SIZE_Q,0b0011)
rsp = Register(SIZE_Q,0b0100)
rbp = Register(SIZE_Q,0b0101)
rsi = Register(SIZE_Q,0b0110)
rdi = Register(SIZE_Q,0b0111)
r8 = Register(SIZE_Q,0b1000)
r9 = Register(SIZE_Q,0b1001)
r10 = Register(SIZE_Q,0b1010)
r11 = Register(SIZE_Q,0b1011)
r12 = Register(SIZE_Q,0b1100)
r13 = Register(SIZE_Q,0b1101)
r14 = Register(SIZE_Q,0b1110)
r15 = Register(SIZE_Q,0b1111)

class _Rip:
    size = SIZE_Q

rip = _Rip()


class Address(x86_ops.Address):
    def __init__(self,offset=0,base=None,index=None,scale=1):
        self.rip = False
        if base is rip:
            assert index is None
            self.rip = True
            base = None

        super().__init__(offset,base,index,scale)

    def _mod_rm_sib_disp(self):
        if self.rip or self.base or self.index:
            return super()._mod_rm_sib_disp()

        return 0b00, 0b100, bytes([0b00100101]) + x86_ops.int_to_32(self.offset)


class Assembly(x86_ops.Assembly):
    def binary(self,name):
        return globals()[name]



@multimethod
def call(proc : Displacement):
    return x86_ops.call(proc)

@multimethod
def call(proc : Register):
    assert proc.size == SIZE_Q
    return x86_ops.call(proc)

@multimethod
def call(proc : Address):
    assert proc.size is None or proc.size == SIZE_Q
    return x86_ops.call(proc)



@multimethod
def dec(x : Register):
    return x86_ops.rex(None,x) + bytes([
        0b11111110 | x.w,
        0b11001000 | x.reg])



@multimethod
def inc(x : Register):
    return x86_ops.rex(None,x) + bytes([
        0b11111110 | x.w,
        0b11000000 | x.reg])



@multimethod
def mov(a : int,b : Register):
    if b.size == SIZE_Q:
        return x86_ops.rex(None,b) + bytes([0b10111000 | b.reg]) + immediate_data(True,a)

    return x86_ops.mov(a,b)

mov.inherit(x86_ops.mov)


@multimethod
def push(x : Register):
    assert x.size == SIZE_Q
    return x86_ops.push(x)

@multimethod
def push(x : Address):
    assert x.size is None or x.size == SIZE_Q
    return x86_ops.push(x)

@multimethod
def push(x : int):
    # note that this is code is different from x86_ops.push because
    # immediate_data is redefined

    byte = x86_ops.fits_in_sbyte(x)
    return bytes([0b01101000 | (byte << 1)]) + immediate_data(not byte,x)
