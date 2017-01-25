#  Copyright 2017 Rouslan Korneychuk
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
from functools import partial
from typing import cast,List,Optional,Sequence,Tuple,Type,Union

from . import abi
from . import debug
from . import dwarf
from .intermediate import *


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

def immediate_data64(w,data):
    w = 8 if w else 1
    if isinstance(data,bytes):
        assert len(data) == w
        return data
    return data.to_bytes(w,byteorder='little',signed=data<0)



class Register(abi.Register,debug.DwarfLocation):
    def __init__(self,size : int,code : int) -> None:
        assert size in (SIZE_B,SIZE_D,SIZE_Q)
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
        return self.size > 1

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
        assert not self.ext
        if self.size == SIZE_B:
            return ['al','cl','dl','bl','ah','ch','dh','bh'][self.code]

        assert self.size == SIZE_D
        return ['eax','ecx','edx','ebx','esp','ebp','esi','edi'][self.reg]

    def __repr__(self):
        return 'Register({},{})'.format(self.size,self.code)

    def __str__(self):
        return '%' + self.name

    def dwarf_reg(self,mode):
        assert self.size == mode.ptr_size
        return dwarf.reg(mode)._elements_[self.name]

    def dwarf_loc_expr(self,op):
        return op.reg(self.dwarf_reg(op.mode))


# this class doesn't offer any new functionality, it only exists so that
# registers can print their own name
class Register64(Register):
    @property
    def name(self):
        if self.size == SIZE_B:
            return [
                'al','cl','dl','bl','spl','bpl','sil','dil',
                'r8b','r9b','r10b','r11b','r12b','r13b','r14b','r15b'][self.code]
        if self.size == SIZE_D:
            return ['eax','ecx','edx','ebx','esp','ebp','esi','edi',
                'r8d','r9d','r10d','r11d','r12d','r13d','r14d','r15d'][self.code]

        assert self.size == SIZE_Q
        return ['rax','rcx','rdx','rbx','rsp','rbp','rsi','rdi',
            'r8','r9','r10','r11','r12','r13','r14','r15'][self.code]

    __hash__ = Register.__hash__


class Address(debug.DwarfLocation):
    default_size = SIZE_D

    def __init__(self,offset : int=0,base : Optional[Register]=None,index : Optional[Register]=None,scale=1,size : Optional[int]=None) -> None:
        assert scale in (1,2,4,8)
        assert (base is None or base.size == self.default_size) and (index is None or index.size == self.default_size)

        # %esp cannot be used as the index
        assert index is None or index.reg != 0b100

        self.offset = offset
        self.base = base
        self.index = index
        self.scale = scale
        self.size = self.default_size if size is None else size

    def __eq__(self,b):
        if isinstance(b,Address):
            return (self.offset == b.offset
                and self.base == b.base
                and self.index == b.index
                and (self.index is None or self.scale == b.scale))

        return NotImplemented

    def __ne__(self,b):
        if isinstance(b,Address):
            return (self.offset != b.offset
                or self.base != b.base
                or self.index != b.index
                or (self.index is not None and self.scale != b.scale))

        return NotImplemented

    def __hash__(self):
        r = self.offset ^ hash(self.base) ^ hash(self.index)
        if self.index is not None: r ^= self.scale
        return r

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

    def __repr__(self):
        args = [str(self.offset)]
        base = self.base
        if getattr(self,'rip',False): base = rip

        if base is not None or self.index is not None:
            args.append(repr(base))
            if self.index is not None:
                args.append(repr(self.index))
                args.append(str(self.index))
        return '{}({})'.format(self.__class__.__name__,','.join(args))

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

    def dwarf_loc_expr(self,op):
        assert self.index is None and not getattr(self,'rip',False),"this case isn't supported"

        if self.base is None:
            return op.const_int(self.offset) + op.deref_size(op.mode.ptr_size)

        if self.base.code == esp.code:
            return op.fbreg(self.offset)

        return op.breg(self.base.dwarf_reg(op.mode),self.offset)

class _Rip(abi.Register):
    size = SIZE_Q

    def __repr__(self):
        return 'rip'

rip = _Rip()


class Address64(Address):
    default_size = SIZE_Q

    def __init__(self,offset : int=0,base : Union[Register,_Rip,None]=None,index : Optional[Register]=None,scale=1,size : Optional[int]=None) -> None:
        self.rip = False
        if base is rip:
            assert index is None
            self.rip = True
            base = None

        assert (base is None or cast(Register,base).size == SIZE_Q) and (index is None or cast(Register,index).size == SIZE_Q)

        super().__init__(offset,cast(Optional[Register],base),index,scale,size)

    __hash__ = Address.__hash__

    def _mod_rm_sib_disp(self):
        if self.rip or self.base or self.index:
            return super()._mod_rm_sib_disp()

        return 0b00, 0b100, bytes([0b00100101]) + int_to_32(self.offset)


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

    if reg is not None:
        w = reg.size == SIZE_Q
        rxb |= reg.ext << 2
    elif isinstance(rm,Address):
        w = rm.size == SIZE_Q
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


spl = Register64(SIZE_B,0b100)
bpl = Register64(SIZE_B,0b101)
sil = Register64(SIZE_B,0b110)
dil = Register64(SIZE_B,0b111)
r8b = Register64(SIZE_B,0b1000)
r9b = Register64(SIZE_B,0b1001)
r10b = Register64(SIZE_B,0b1010)
r11b = Register64(SIZE_B,0b1011)
r12b = Register64(SIZE_B,0b1100)
r13b = Register64(SIZE_B,0b1101)
r14b = Register64(SIZE_B,0b1110)
r15b = Register64(SIZE_B,0b1111)

r8d = Register64(SIZE_D,0b1000)
r9d = Register64(SIZE_D,0b1001)
r10d = Register64(SIZE_D,0b1010)
r11d = Register64(SIZE_D,0b1011)
r12d = Register64(SIZE_D,0b1100)
r13d = Register64(SIZE_D,0b1101)
r14d = Register64(SIZE_D,0b1110)
r15d = Register64(SIZE_D,0b1111)

rax = Register64(SIZE_Q,0b0000)
rcx = Register64(SIZE_Q,0b0001)
rdx = Register64(SIZE_Q,0b0010)
rbx = Register64(SIZE_Q,0b0011)
rsp = Register64(SIZE_Q,0b0100)
rbp = Register64(SIZE_Q,0b0101)
rsi = Register64(SIZE_Q,0b0110)
rdi = Register64(SIZE_Q,0b0111)
r8 = Register64(SIZE_Q,0b1000)
r9 = Register64(SIZE_Q,0b1001)
r10 = Register64(SIZE_Q,0b1010)
r11 = Register64(SIZE_Q,0b1011)
r12 = Register64(SIZE_Q,0b1100)
r13 = Register64(SIZE_Q,0b1101)
r14 = Register64(SIZE_Q,0b1110)
r15 = Register64(SIZE_Q,0b1111)



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

def _op_imm_addr(byte1,mid,a,b):
    fits = fits_in_sbyte(a)
    return rex(None,b) + bytes([byte1 | (fits << 1) | (b.size > 1)]) + b.mod_rm_sib_disp(mid) + immediate_data(not fits,a)



#
#
#
# def cpuid():
#     return b'\x0F\xA2'
#
#
#
# def leave():
#     return b'\xC9'
#
#
#
# @multimethod
# def loop(x : Displacement):
#     return b'\xE2' + int_to_8(x.val)
#
# @multimethod
# def loopz(x : Displacement):
#     return b'\xE1' + int_to_8(x.val)
#
# loope = loopz
#
# @multimethod
# def loopnz(x : Displacement):
#     return b'\xE0' + int_to_8(x.val)
#
# loopne = loopnz
#
# LOOP_LEN = 2
#


JMP_DISP_MIN_LEN = 2
JMP_DISP_MAX_LEN = 5

class X86OpDescription(OpDescription):
    modifies_test_flags = False

    def assembly_name(self,args):
        r = self.name
        if r in {'jcc','cmovcc'}:
            assert isinstance(args[0],Test)
            r = r[0:-2] + args[0].mnemonic
            del args[0]
        elif r not in {'call','push','pop','ret','nop'}:
            m = None
            for a in args:
                if isinstance(a,Address):
                    m = a
                elif not isinstance(a,int):
                    break
            else:
                assert m is not None
                r += {SIZE_B : 'b',SIZE_W : 'w',SIZE_D : 'd',SIZE_Q : 'q'}[m.size]

        return r

    def _indirect(self):
        return self.name == 'call' or self.name == 'jmp'

    def assembly_arg(self,nextaddr,x):
        if isinstance(x,int):
            return '${:#x}'.format(x)
        if isinstance(x,Displacement):
            return '{:x}'.format(x.val + nextaddr)
        if self._indirect() and isinstance(x,(Address,Register)):
            return '*' + str(x)
        return str(x)

def splice(seq,start,size,replacement):
    return tuple(seq[0:start]) + replacement + tuple(seq[start+size:])

class IndirectionAdapter(RegAllocatorOverloads):
    Address = Address

    def __init__(self,base_op : X86OpDescription,param : int,offset : int,base : bool,index : bool,scale : int) -> None:
        regs = base + index
        super().__init__(splice(base_op.param_dirs,param,1,(ParamDir(True,False),) * regs))

        self.base_op = base_op
        self.param = param
        self.offset = offset
        self.base = base
        self.index = index
        self.scale = scale

    @property
    def name(self):
        return self.base_op.name

    @staticmethod
    def _func(addr_t,f,param,offset,base,index,scale,*args):
        addr = addr_t(
            offset,
            args[param] if base else None,
            args[param+base] if index else None,
            scale)

        regs = base + index
        return f(*splice(args,param,regs,(addr,)))

    def _to_base_args(self,args):
        return splice(args,self.param,self.base + self.index,(AddressType(),))

    def _wrap_func(self,f):
        return partial(self._func,self.Address,f,self.param,self.offset,self.base,self.index,self.scale)

    def wrap_overload(self,o):
        regs = self.base + self.index

        return o.variant(
            splice(o.params,self.param,1,(FixedRegister,) * regs),
            self._wrap_func(o.func))

    def best_match(self,args):
        return self.wrap_overload(self.base_op.best_match(self._to_base_args(args)))

    def to_ir2(self,args):
        o = self.base_op.exact_match(self._to_base_args(args))
        return Instr2(self.base_op,self.wrap_overload(o),args)

    def assembly(self,args : Sequence,addr : int,binary : bytes,annot : Optional[str]=None):
        m = self.Address(
            self.offset,
            args[self.param] if self.base else None,
            args[self.param+self.base] if self.index else None,
            self.scale)

        return self.base_op.assembly(splice(args,self.param,self.base+self.index,(m,)),addr,binary,annot)

class IndirectionAdapter64(IndirectionAdapter):
    Address = Address64

class BasicOps:
    # noinspection PyPep8Naming
    def __init__(self,abi : 'X86Abi') -> None:
        imm_min = -0x80000000
        imm_max = 0xffffffff
        immediate_data_full = immediate_data

        if abi.ptr_size == 8:
            # 32-bit immediate values get sign extended when used with 64-bit
            # values. Since we most often work with addresses, which are
            # represented as unsigned integers, we disallow immediate values
            # with the highest bit set.
            imm_max = 0x7fffffff

            immediate_data_full = immediate_data64

        R = ParamDir(True,False)
        W = ParamDir(False,True)
        RW = ParamDir(True,True)
        reg = (FixedRegister,Register)
        if abi.ptr_size == 8: reg += (Register64,)
        m = (AddressType,abi.Address)
        imm = Immediate[imm_min,imm_max]
        d = (Target,Displacement)
        test = Test

        self.add = X86OpDescription('add',[
            Overload([reg,reg],
                (lambda a,b: _op_reg_reg(0,a,b))),
            Overload([reg,m],
                (lambda a,b: _op_addr_reg(0,b,a,False))),
            Overload([m,reg],
                (lambda a,b: _op_addr_reg(0,a,b,True))),
            Overload([imm,reg],
                (lambda a,b: _op_imm_reg(0b10000000,0,0b00000100,a,b))),
            Overload([imm,m],
                (lambda a,b: _op_imm_addr(0b10000000,0,a,b)))
        ],[R,RW])
        self.add.modifies_test_flags = True

        def call_d(proc : Displacement) -> bytes:
            return b'\xE8' + int_to_32(proc.val)

        def call_reg(proc : Register) -> bytes:
            assert proc.size == abi.ptr_size
            return rex(None,proc,False) + bytes([
                0b11111111,
                0b11010000 | proc.reg])

        def call_m(proc : Address) -> bytes:
            assert proc.size == abi.ptr_size
            return rex(None,proc,False) + b'\xFF' + proc.mod_rm_sib_disp(0b010)

        self.call = X86OpDescription('call',[
            Overload([d],call_d,min_len=5,max_len=5),
            Overload([reg],call_reg),
            Overload([m],call_m)
        ],[R])

        def cmovcc_reg_reg(test : Test,a : Register,b : Register) -> bytes:
            assert a.w and b.w

            # unlike most, this instruction moves R/M to Reg instead of Reg to R/M
            # (look-up the Intel machine instruction format for the nomenclature)
            return rex(b,a) + bytes([
                0b00001111,
                0b01000000 | test.val,
                0b11000000 | (b.reg << 3) | a.reg])

        def cmovcc_m_reg(test : Test,a : Address,b : Register) -> bytes:
            assert b.w
            return rex(b,a) + bytes(
                [0b00001111,0b01000000 | test.val]) + a.mod_rm_sib_disp(b.reg)

        self.cmovcc = X86OpDescription('cmovcc',[
            Overload([test,reg,reg],cmovcc_reg_reg),
            Overload([test,m,reg],cmovcc_m_reg)
        ],[R,R,RW])

        self.cmp = X86OpDescription('cmp',[
            Overload([reg,reg],
                (lambda a,b: _op_reg_reg(0b00111000,a,b))),
            Overload([m,reg],
                (lambda a,b: _op_addr_reg(0b00111000,a,b,True))),
            Overload([reg,m],
                (lambda a,b: _op_addr_reg(0b00111000,b,a,False))),
            Overload([imm,reg],
                (lambda a,b: _op_imm_reg(0b10000000,0b111,0b00111100,a,b))),
            Overload([imm,m],
                (lambda a,b: _op_imm_addr(0b10000000,0b111,a,b)))
        ],[R,R])
        self.cmp.modifies_test_flags = True

        if abi.ptr_size == 8:
            def dec_reg(x : Register) -> bytes:
                return rex(None,x) + bytes([
                    0b11111110 | x.w,
                    0b11001000 | x.reg])
        else:
            def dec_reg(x : Register) -> bytes:
                if x.w:
                    return bytes([0b01001000 | x.reg])
                else:
                    return bytes([
                        0b11111110,
                        0b11001000 | x.reg])

        self.dec = X86OpDescription('dec',[
            Overload([reg],dec_reg),
            Overload([m],
                (lambda x: rex(None,x) + bytes([0b11111110 | (x.size > 1)]) + x.mod_rm_sib_disp(0b001)))
        ],[RW])
        self.cmp.modifies_test_flags = True

        def imul_reg_reg(a : Register,b : Register) -> bytes:
            assert a.size == b.size
            if b.code == 0:
                return rex(None,a) + bytes(
                    [0b11110110 | (a.size > 1),0b11101000 | a.reg])

            assert a.size > SIZE_B
            return rex(b,a) + bytes(
                [0b00001111,0b10101111,0b11000000 | (b.reg << 3) | a.reg])

        def imul_m_reg(a : Address,b : Register) -> bytes:
            assert a.size == b.size
            if b.code == 0:
                return rex(b.size,a) + bytes(
                    [0b11110110 | (a.size > 1)]) + a.mod_rm_sib_disp(0b101)

            assert a.size > SIZE_B
            return rex(b,a) + b'\x0F\xAF' + a.mod_rm_sib_disp(b.reg)

        self.imul = X86OpDescription('imul',[
            Overload([reg,reg],imul_reg_reg),
            Overload([m,reg],imul_m_reg)
        ],[R,RW])
        self.imul.modifies_test_flags = True

        def imul_reg_imm_reg(a : Register,b : int,dest : Register) -> bytes:
            assert a.size == dest.size and a.size > SIZE_B

            fits = fits_in_sbyte(b)
            return rex(dest,a) + bytes([0b01101001 | (fits << 1),
                0b11000000 | (dest.reg << 3) | a.reg]) + immediate_data(
                not fits,b)

        def imul_m_imm_reg(a : Address,b : int,dest : Register) -> bytes:
            assert a.size == dest.size and a.size > SIZE_B

            fits = fits_in_sbyte(b)
            return rex(dest,a) + bytes(
                [0b01101001 | (fits << 1)]) + a.mod_rm_sib_disp(
                dest.reg) + immediate_data(not fits,b)

        self.imul_imm = X86OpDescription('imul',[
            Overload([reg,imm,reg],imul_reg_imm_reg),
            Overload([m,imm,reg],imul_m_imm_reg)
        ],[R,R,W])
        self.imul_imm.modifies_test_flags = True

        if abi.ptr_size == 8:
            def inc_reg(x : Register) -> bytes:
                return rex(None,x) + bytes([
                    0b11111110 | x.w,
                    0b11000000 | x.reg])
        else:
            def inc_reg(x : Register) -> bytes:
                if x.w:
                    return bytes([0b01000000 | x.reg])
                else:
                    return bytes([
                        0b11111110,
                        0b11000000 | x.reg])

        self.inc = X86OpDescription('inc',[
            Overload([reg],inc_reg),
            Overload([m],(lambda x: rex(None,x) + bytes([0b11111110 | (x.size > 1)]) + x.mod_rm_sib_disp(0)))
        ],[RW])
        self.inc.modifies_test_flags = True

        def jcc(test : Test,x : Displacement) -> bytes:
            if fits_in_sbyte(x):
                return bytes([0b01110000 | test.val]) + int_to_8(x.val)

            return bytes([
                0b00001111,
                0b10000000 | test.val]) + int_to_32(x.val)

        self.jcc = X86OpDescription('jcc',[
            Overload([test,d],jcc,min_len=2,max_len=6)
        ],[R,R])

        def jmp_d(x : Displacement) -> bytes:
            fits = fits_in_sbyte(x)
            return bytes([0b11101001 | (fits << 1)]) + (
                [int_to_32,int_to_8][fits])(x.val)

        def jmp_reg(x : Register) -> bytes:
            assert x.w
            return rex(None,x) + bytes([
                0b11111111,
                0b11100000 | x.reg])

        def jmp_m(x : Address) -> bytes:
            assert x.size == abi.ptr_size
            return rex(None,x,False) + b'\xFF' + x.mod_rm_sib_disp(0b100)

        self.jmp = X86OpDescription('jmp',[
            Overload([d],jmp_d,min_len=JMP_DISP_MIN_LEN,max_len=JMP_DISP_MAX_LEN),
            Overload([reg],jmp_reg),
            Overload([m],jmp_m)
        ],[R])

        def lea(a: Address,b: Register) -> bytes:
            assert a.size == b.size and a.size == abi.ptr_size
            return rex(b,a) + b'\x8D' + a.mod_rm_sib_disp(b.reg)

        self.lea = X86OpDescription('lea',[
            Overload([m,reg],lea)
        ],[ParamDir(False,False),W])

        self.leave = X86OpDescription('leave',[
            Overload([],(lambda: b'\xC9'))
        ],[])

        def mov_addr_reg(a,b,forward) -> bytes:
            if b.reg == 0b000 and a.offset_only():
                r = bytes([0b10100000 | (forward << 1) | b.w]) + int_to_32(
                    a.offset)
                if b.size == SIZE_Q: r = b'\x48' + r
                return r

            return _op_addr_reg(0b10001000,a,b,forward)

        def mov_imm_reg(a : int,b : Register) -> bytes:
            return rex(None,b) + bytes([0b10110000 | (b.w << 3) | b.reg]) + immediate_data_full(b.w,a)

        def mov_imm_m(a : int,b : Address) -> bytes:
            return rex(None,b) + bytes([0b11000110 | (b.size > 1)]) + b.mod_rm_sib_disp(0) + immediate_data(b.size > 1,a)

        mov_imm = imm
        if abi.ptr_size == 8:
            mov_imm = Immediate[-0x8000000000000000,0xffffffffffffffff]

        self.mov = X86OpDescription('mov',[
            Overload([reg,reg],
                (lambda a,b: _op_reg_reg(0b10001000,a,b))),
            Overload([m,reg],
                (lambda a,b: mov_addr_reg(a,b,True))),
            Overload([reg,m],
                (lambda a,b: mov_addr_reg(b,a,False))),
            Overload([mov_imm,reg],mov_imm_reg),
            Overload([imm,m],mov_imm_m)
        ],[R,W])

        self.neg = X86OpDescription('neg',[
            Overload([reg],
                (lambda x: rex(None,x) + bytes([0b11110110 | x.w, 0b11011000 | x.reg]))),
            Overload([m],
                (lambda x: rex(None,x) + bytes([0b11110110 | (x.size > 1)]) + x.mod_rm_sib_disp(0b011)))
        ],[RW])
        self.neg.modifies_test_flags = True

        self.nop = X86OpDescription('nop',[
            Overload([],(lambda: b'\x90'))
        ],[])

        self.not_ = X86OpDescription('not',[
            Overload([reg],(lambda x: rex(None,x) + bytes([0b11110110 | x.w, 0b11010000 | x.reg]))),
            Overload([m],(lambda x: rex(None,x) + bytes([0b11110110 | (x.size > 1)]) + x.mod_rm_sib_disp(0b010)))
        ],[RW])

        self.or_ = X86OpDescription('or',[
            Overload([reg,reg],
                (lambda a,b: _op_reg_reg(0b00001000,a,b))),
            Overload([m,reg],
                (lambda a,b: _op_addr_reg(0b00001000,a,b,True))),
            Overload([reg,m],
                (lambda a,b: _op_addr_reg(0b00001000,b,a,False))),
            Overload([imm,reg],
                (lambda a,b: _op_imm_reg(0b10000000,0b001,0b00001100,a,b))),
            Overload([imm,m],
                (lambda a,b: _op_imm_addr(0b10000000,0b001,a,b)))
        ],[R,RW])
        self.or_.modifies_test_flags = True

        def pop_reg(x : Register):
            return rex(None,x,False) + bytes([0b01011000 | x.reg])

        self.pop = X86OpDescription('pop',[
            Overload([reg],pop_reg),
            Overload([m],
                (lambda x: rex(None,x,False) + b'\x8F' + x.mod_rm_sib_disp(0)))
        ],[W])

        def push_imm(x : int) -> bytes:
            byte = fits_in_sbyte(x)
            return bytes([0b01101000 | (byte << 1)]) + immediate_data_full(not byte,x)

        self.push = X86OpDescription('push',[
            Overload([reg],
                (lambda x: rex(None,x,False) + bytes([0b01010000 | x.reg]))),
            Overload([m],
                (lambda x: rex(None,x,False) + b'\xFF' + x.mod_rm_sib_disp(0b110))),
            Overload([mov_imm],push_imm)
        ],[R])

        self.ret = X86OpDescription('ret',[
            Overload([],
                (lambda: b'\xC3'))
        ],[])

        self.ret_pop = X86OpDescription('ret',[
            Overload([imm],
                (lambda pop: b'\xC2' + int_to_16(pop)))
        ],[R])

        def shx_imm_reg(amount,x,shiftright) -> bytes:
            r = rex(None,x)

            if amount == 1:
                return r + bytes([
                    0b11010000 | x.w,
                    0b11100000 | (shiftright << 3) | x.reg])

            return r + bytes([
                0b11000000 | x.w,
                0b11100000 | (shiftright << 3) | x.reg]) + immediate_data(
                False,amount)

        def shx_reg_reg(amount,x,shiftright) -> bytes:
            assert amount == cl

            return rex(None,x) + bytes([
                0b11010010 | x.w,
                0b11100000 | (shiftright << 3) | x.reg])

        def shx_imm_addr(amount,x,shiftright) -> bytes:
            r = rex(None,x)
            rmsd = x.mod_rm_sib_disp(0b100 | shiftright)
            if amount == 1:
                return r + bytes([0b11010000 | (x.size > 1)]) + rmsd

            return r + bytes(
                [0b11000000 | (x.size > 1)]) + rmsd + immediate_data(False,
                amount)

        def shx_reg_addr(amount,x,shiftright) -> bytes:
            assert amount == cl

            return rex(None,x) + bytes(
                [0b11010010 | (x.size > 1)]) + x.mod_rm_sib_disp(
                0b100 | shiftright)

        cl_only = FixedRegister[[abi.reg_indices[Register(abi.ptr_size,cl.code)]]]

        self.shl = X86OpDescription('shl',[
            Overload([imm,reg],
                (lambda amount,x: shx_imm_reg(amount,x,False))),
            Overload([cl_only,reg],
                (lambda amount,x: shx_reg_reg(amount,x,False))),
            Overload([imm,m],
                (lambda amount,x: shx_imm_addr(amount,x,False))),
            Overload([cl_only,m],
                (lambda amount,x: shx_reg_addr(amount,x,False)))
        ],[R,RW])
        self.shl.modifies_test_flags = True

        self.shr = X86OpDescription('shr',[
            Overload([imm,reg],
                (lambda amount,x: shx_imm_reg(amount,x,True))),
            Overload([cl_only,reg],
                (lambda amount,x: shx_reg_reg(amount,x,True))),
            Overload([imm,m],
                (lambda amount,x: shx_imm_addr(amount,x,True))),
            Overload([cl_only,m],
                (lambda amount,x: shx_reg_addr(amount,x,True)))
        ],[R,RW])
        self.shr.modifies_test_flags = True

        self.sub = X86OpDescription('sub',[
            Overload([reg,reg],
                (lambda a,b: _op_reg_reg(0b00101000,a,b))),
            Overload([m,reg],
                (lambda a,b: _op_addr_reg(0b00101000,a,b,True))),
            Overload([reg,m],
                (lambda a,b: _op_addr_reg(0b00101000,b,a,False))),
            Overload([imm,reg],
                (lambda a,b: _op_imm_reg(0b10000000,0b101,0b00101100,a,b))),
            Overload([imm,m],
                (lambda a,b: _op_imm_addr(0b10000000,0b101,a,b)))
        ],[R,RW])
        self.sub.modifies_test_flags = True

        self.test = X86OpDescription('test',[
            Overload([reg,reg],
                (lambda a,b: _op_reg_reg(0b10000100,a,b))),
            Overload([m,reg],
                (lambda a,b: _op_addr_reg(0b10000100,a,b,True))),
            Overload([imm,reg],
                (lambda a,b: _op_imm_reg(0b11110110,0,0b10101000,a,b))),
            Overload([imm,m],
                (lambda a,b: rex(None,b) + bytes([0b11110110 | (b.size > 1)]) + b.mod_rm_sib_disp(0) + immediate_data(b.size > 1,a)))
        ],[R,R])
        self.test.modifies_test_flags = True

        def xor_reg_reg(a : Register,b : Register) -> bytes:
            return _op_reg_reg(0b00110000,a,b)

        self.xor = X86OpDescription('xor',[
            Overload([reg,reg],xor_reg_reg),
            Overload([reg,m],
                (lambda a,b: _op_addr_reg(0b00110000,b,a,True))),
            Overload([m,reg],
                (lambda a,b: _op_addr_reg(0b00110000,a,b,False))),
            Overload([imm,reg],
                (lambda a,b: _op_imm_reg(0b10000000,0b110,0b00110100,a,b))),
            Overload([imm,m],
                (lambda a,b: _op_imm_addr(0b10000000,0b110,a,b)))
        ],[R,RW])
        self.xor.modifies_test_flags = True

        # This pseudo instruction serves two purposes: to replace "mov $0, %x"
        # with "xor %x, %x" (it's shorter and at least as fast), and to allow
        # moving a 64-bit immediate value to a memory location (by splitting it
        # into two moves), which lets us treat moves to memory and registers
        # uniformly.

        def macro_mov_imm_reg(a : int,b : Register):
            if a == 0: return xor_reg_reg(b,b)
            return mov_imm_reg(a,b)

        macro_mov_imm_m = mov_imm_m
        if abi.ptr_size == 8:
            # Note: this function assumes little-endian format
            def macro_mov_imm_m(a : int,b : Address):
                if not fits_in_sdword(a):
                    # noinspection PyTypeChecker
                    return mov_imm_m(a & 0xffffffff,b) + mov_imm_m(a >> 32,b + 4)
                return mov_imm_m(a,b)

        self.macro_mov = X86OpDescription('macro_mov',
            self.mov.overloads[0:3] + [
                Overload(self.mov.overloads[3].params,macro_mov_imm_reg),
                Overload([mov_imm,m],macro_mov_imm_m)
        ],[R,W])

        # this "op" does way too much, but our intermediate compile process is
        # not sophisticated enough to handle a case where simultaneously the
        # register allocation can affect the instructions emitted and the
        # instructions emitted can affect the register allocation
        def macro_jump_table(jt : JumpTable,val : Register,tmp1 : Register,tmp2 : Register):
            table = b''
            diff1 = jt.targets[1].displacement - jt.targets[0].displacement

            for i in range(2,len(jt.targets)):
                if (jt.targets[i].displacement - jt.targets[i - 1].displacement) != diff1:
                    jmp_size = JMP_DISP_MIN_LEN
                    force_wide = False

                    # can we use short jumps?
                    dist_extra = 0
                    for t in reversed(jt.targets):
                        if jt.here.displacement - t.displacement + dist_extra > 127:
                            # no, we can not
                            jmp_size = JMP_DISP_MAX_LEN
                            force_wide = True
                            break
                        dist_extra += jmp_size

                    dist_extra = 0
                    for t in reversed(jt.targets):
                        table = jmp_d(
                            Displacement(
                                jt.here.displacement - t.displacement + dist_extra,
                                force_wide)) + table
                        dist_extra += jmp_size

                    spacing = jmp_size
                    break
            else:
                # We don't need the actual table. We can just do JMP reg * diff1
                spacing = -diff1

            r = b''

            if spacing in (1,2,4,8):
                scale = spacing
                index = val
            else:
                scale = 1
                index = tmp2
                r += imul_reg_imm_reg(val,spacing,index)

            rip = getattr(abi,'r_rip',None)
            if rip:
                # RIP addresses cannot have indexes so we have to load the start
                # address into a register first

                jmp = (
                    lea(abi.Address(0,tmp1,index,scale),tmp1)
                    + jmp_reg(tmp1))
                r += lea(abi.Address(len(jmp),rip),tmp1) + jmp
            else:
                # the offset of the JMP instruction needs to include its own size
                proto_jmp = jmp_m(abi.Address(127,tmp1,val,scale))

                pop = pop_reg(tmp1)
                jmp = jmp_m(
                    abi.Address(len(pop) + len(proto_jmp),tmp1,index,scale))

                assert len(jmp) == len(proto_jmp)

                # TODO: use RelocBuffer to allow using an absolute address offset
                # instead of this CALL + POP chicanery. It could be done by
                # changing pyinternals to always support relocation and tracking
                # the location of relocatable offsets.

                r += call_d(Displacement(0)) + pop + jmp

            return r + table

        self.jump_table = X86OpDescription('macro_jump_table',[
            Overload([JumpTable,reg,reg,reg],macro_jump_table)
        ],[R,R,W,W])

class JumpTable:
    def __init__(self,targets : Sequence[Target]) -> None:
        assert len(targets) > 2
        self.targets = targets
        self.here = Target()

def _orderless_cmp(a,b):
    return (a[0] == b[0] and a[1] == b[1]) or (a[0] == b[1] and a[1] == b[0])

class X86ExtraState(ExtraState):
    """x86-specific implementation of ExtraState.

    This removes redundant CMP and TEST instructions that can arise from an
    'elif' test with the same two values being compared as the 'if' test, or
    an 'if' test after an arithmetic instruction that sets test flags in an
    equivalent way.

    """
    def __init__(self,ops : BasicOps,last_cmp : Tuple[object,object]=None) -> None:
        self.ops = ops
        self.last_cmp = last_cmp

    def copy(self):
        return X86ExtraState(self.ops,self.last_cmp)

    def conform_to(self,other : 'X86ExtraState'):
        if self.last_cmp != other.last_cmp:
            self.last_cmp = None

    def _handle_cmp(self,ab,instr):
        if self.last_cmp is None:
            self.last_cmp = ab
        elif _orderless_cmp(self.last_cmp,ab):
            return None

        return instr

    def process_instr(self,i : int,instr : Instr):
        if isinstance(instr.op,X86OpDescription):
            if instr.op is self.ops.cmp:
                assert len(instr.args) == 2
                return self._handle_cmp(instr.args,instr)
            elif instr.op is self.ops.test:
                assert len(instr.args) == 2
                if instr.args[0] == instr.args[1]:
                    return self._handle_cmp((instr.args[0],0),instr)
            elif instr.op.modifies_test_flags:
                assert instr.op.param_dirs[-1].writes and not any(pd.writes for pd in instr.op.param_dirs[0:-1])
                self.last_cmp = instr.args[-1],0
            elif instr.op is self.ops.call:
                self.last_cmp = None
            elif self.last_cmp is not None:
                # check if either of the last compared values are written to
                for i,p in enumerate(instr.op.param_dirs):
                    if p.writes and instr.args[i] in self.last_cmp:
                        assert isinstance(instr.args[i],MutableValue)
                        self.last_cmp = None
                        break
        return instr

class X86OpGen(JumpCondOpGen):
    def __init__(self,abi : 'X86Abi') -> None:
        super().__init__(abi)
        self.ops = abi.ops

    def bin_op(self,a : Value,b : Value,dest : MutableValue,op_type : OpType) -> IRCode:
        ensure_same_size(self.abi.ptr_size,a,b)

        op = {
            OpType.add : self.ops.add,
            OpType.sub : self.ops.sub,
            OpType.mul : self.ops.imul,
            OpType.or_ : self.ops.or_,
            OpType.xor : self.ops.xor
        }[op_type]

        if op_type == OpType.mul and (isinstance(a,Immediate) or isinstance(b,Immediate)):
            if isinstance(a,Immediate):
                a,b = b,a
            return [Instr(self.ops.imul_imm,a,b,dest)]

        r = [] # type: IRCode

        if dest is not a:
            if op_type in commutative_ops and b is dest:
                a,b = b,a
            else:
                r.append(Instr(self.ops.mov,a,dest))

        if (isinstance(b,Immediate) and cast(Immediate,b).val == 1 and
                op_type in (OpType.add,OpType.sub) and
                not self.abi.tuning.prefer_addsub_over_incdec):
            r.append(Instr(
                self.ops.inc if op_type == OpType.add else self.ops.dec,
                dest))

        r.append(Instr(op,b,dest))
        return r

    def unary_op(self,a : Value,dest : MutableValue,op_type : UnaryOpType) -> IRCode:
        op = {
            UnaryOpType.neg : self.ops.neg
        }[op_type]

        r = [] # type: IRCode

        if dest is not a:
            r.append(Instr(self.ops.mov,a,dest))
        r.append(Instr(op,dest))
        return r

    def load_addr(self,addr : IndirectVar,dest : MutableValue) -> IRCode:
        return [Instr(self.ops.lea,addr,dest)]

    def move(self,src : Value,dest : MutableValue) -> IRCode:
        return [Instr(self.ops.macro_mov,src,dest)]

    def jump(self,dest : Target) -> IRCode:
        return [Instr(self.ops.jmp,dest),IRJump(dest,False,1)]

    def _basic_jump_if(self,dest,cond) -> IRCode:
        if cond.a == 0 or (isinstance(cond.b,(Immediate,int)) and cond.b != 0):
            if cond.a == 0:
                instr = Instr(self.ops.test,cond.b,cond.b)
            else:
                instr = Instr(self.ops.cmp,cond.b,cond.a)

            test_t = ([test_E,test_NE,test_G,test_GE,test_L,test_LE]
                    if cond.signed else
                [test_E,test_NE,test_A,test_NB,test_B,test_BE])[cond.op.value - 1]
        else:
            if cond.b == 0:
                instr = Instr(self.ops.test,cond.a,cond.a)
            else:
                instr = Instr(self.ops.cmp,cond.a,cond.b)

            test_t = ([test_E,test_NE,test_L,test_LE,test_G,test_GE]
                    if cond.signed else
                [test_E,test_NE,test_B,test_BE,test_A,test_NB])[cond.op.value - 1]

        return [instr,Instr(self.ops.jcc,test_t,dest),IRJump(dest,True,2)]

    def jump_table(self,val : Value,targets : Sequence[Target]) -> IRCode:
        if len(targets) < 2:
            if not targets: return []

            return self.jump_if(targets[0],BinCmp(val,Immediate(0),CmpType.eq))

        # the two Vars are just temporaries needed by the rather involved "op"
        return [Instr(self.ops.jump_table,JumpTable(targets),val,Var(),Var())]

    def _reg_to_ir(self,reg):
        return FixedRegister(self.abi.reg_indices[reg])

    def _func_arg(self,i,prev_frame=False):
        if i < len(self.abi.r_arg):
            return self._reg_to_ir(self.abi.r_arg[i])

        if not self.abi.shadow:
            i -= len(self.abi.r_arg)
        return ArgStackItem(i,prev_frame)

    def _call_impl(self,func : Value,args : Sequence[Value]=(),store_ret : Optional[Var]=None) -> IRCode:
        arg_regs = min(len(args),len(self.abi.r_arg))
        r = [LockRegs(range(arg_regs))] # type: IRCode
        for i,arg in enumerate(args):
            r += self.move(arg,self._func_arg(i))

        scratch = self.abi.r_scratch[:]
        scratch.append(self.abi.r_ret)
        scratch.extend(self.abi.r_arg[min(len(self.abi.r_arg),len(args)):])

        r.append(InvalidateRegs(
            [self.abi.reg_indices[r] for r in scratch],
            [self.abi.reg_indices[r] for r in self.abi.r_pres]))
        r.append(Instr(self.ops.call,func))

        r.append(UnlockRegs(range(arg_regs)))

        if store_ret is not None:
            r.append(CreateVar(store_ret,self._reg_to_ir(self.abi.r_ret)))

        return r

    def shift(self,src : Value,shift_dir : ShiftDir,amount : Value,dest : MutableValue) -> IRCode:
        r = [] # type: IRCode
        if dest is not src:
            r.append(Instr(self.ops.mov,src,dest))
        r.append(Instr(self.ops.shl if shift_dir == ShiftDir.left else self.ops.shr,amount,dest))
        return r

    def get_return_address(self,v : Value) -> IRCode:
        return [Instr(self.ops.pop,v)]

    def return_value(self,v : Value):
        return self.move(v,FixedRegister(self.abi.reg_indices[self.abi.r_ret]))

    def get_compiler(self,regs_used : int,stack_used : int,args_used : int):
        return X86Compiler(self.abi,self.ops,regs_used,stack_used,args_used)

    def get_cur_func_arg(self,i : int):
        return self._func_arg(i,True)

    def allocater_extra_state(self):
        return X86ExtraState(self.ops)

    def process_indirection(self,instr : Instr,ov : Overload,inds : Sequence[int]):
        assert len(inds) == 1
        i = inds[0]
        arg = instr.args[i]

        scale = arg.scale or self.abi.ptr_size

        if not fits_imm32(self.abi,arg.offset):
            assert arg.base is None and arg.index is None, (
                "if it ever comes up, this case should be implemented")
            return Instr(
                self.abi.IndirectionAdapter(instr.op,i,0,True,False,0),
                *splice(instr.args,i,1,(Immediate(arg.offset),)))

        insert = ()
        if arg.base is not None:
            insert += (arg.base,)
        if arg.index is not None:
            insert += (arg.index,)

        op = self.abi.IndirectionAdapter(
            instr.op,
            i,
            arg.offset,
            arg.base is not None,
            arg.index is not None,
            scale)
        return Instr(op,*splice(instr.args,i,1,insert)), op.wrap_overload(ov)

class X86Compiler(IRCompiler):
    def __init__(self,abi : 'X86Abi',ops : BasicOps,regs_used : int,stack_used : int,args_used : int) -> None:
        self.abi = abi
        self.ops = ops

        # we rely on the indices of abi.r_pres being 0 to len(abi.r_pres)-1
        # for this calculation
        assert max(abi.reg_indices[r] for r in abi.r_pres) < len(abi.r_pres)
        self.regs_saved = min(regs_used,len(abi.r_pres))

        self.stack_size = stack_used + self.unreserved_offset
        if abi.shadow:
            self.stack_size += args_used
        else:
            self.stack_size += max(args_used-len(abi.r_arg),0)

    @property
    def unreserved_offset(self):
        # plus 1 for the function return address
        return self.regs_saved + 1

    def _reg_to_ir(self,reg):
        return FixedRegister(self.abi.reg_indices[reg])

    def _sp_shift(self):
        return (self.stack_size-self.unreserved_offset)*self.abi.ptr_size

    def prolog(self):
        # The function return value is already on the stack. We push the
        # registers that we need to preserve, onto the stack and move the stack
        # pointer down.
        r = annotate(debug.RETURN_ADDRESS)
        for i in range(self.regs_saved):
            reg = self.abi.r_pres[i]
            r.append(self.ops.push(self._reg_to_ir(reg)))
            r.extend(annotate(debug.SaveReg(reg)))

        sp_shift = self._sp_shift()
        if sp_shift:
            r.append(self.ops.sub(Immediate(sp_shift),self._reg_to_ir(self.abi.r_sp)))
        r.extend(annotate(debug.PrologEnd(self.stack_size)))
        return r

    def epilog(self):
        # Move the stack pointer back up and restore the registers we needed to
        # preserve.
        r = []

        sp_shift = self._sp_shift()
        if sp_shift:
            r.append(self.ops.add(Immediate(sp_shift),self._reg_to_ir(self.abi.r_sp)))

        r.extend(annotate(debug.EPILOG_START))

        for i in range(self.regs_saved-1,-1,-1):
            reg = self.abi.r_pres[i]
            r.append(self.ops.pop(self._reg_to_ir(reg)))
            r.extend(annotate(debug.RestoreReg(reg)))

        r.append(self.ops.ret())

        return r

    def get_reg(self,index : int,size : int):
        return self.abi.Register(size or self.abi.ptr_size,cast(Register,self.abi.all_regs[index]).code)

    def get_stack_addr(self,index : int,size : int,sect : StackSection):
        if sect == StackSection.local:
            offset = (self.stack_size - (index + self.unreserved_offset) - 1) * self.abi.ptr_size
        elif sect == StackSection.args:
            offset = index * self.abi.ptr_size
        else:
            assert sect == StackSection.previous
            offset = (self.stack_size + index) * self.abi.ptr_size
        return self.abi.Address(offset,self.abi.r_sp,size=size or self.abi.ptr_size)

    def get_displacement(self,amount : int,force_wide : bool):
        return Displacement(amount,force_wide)

    def get_immediate(self,val : int,size : int):
        return val

    def nops(self,size : int):
        return [self.ops.nop()] * size


def fits_imm32(abi,x):
    """Return True if x fits in a 32-bit immediate value without sign-extend.

    32-bit immediate values are interpreted as signed. In 64-bit mode, these
    values get sign-extended to 64 bits and thus have their binary
    representation altered, which can make a difference when comparing
    addresses.

    """
    if abi.ptr_size == 8:
        return fits_in_sdword(x)

    return True


class X86Abi(abi.Abi):
    code_gen = X86OpGen

    Address = Address
    Register = Register
    IndirectionAdapter = IndirectionAdapter

    def __init__(self,*args,**kwds):
        super().__init__(*args,**kwds)
        self.ops = BasicOps(self)

class CdeclAbi(X86Abi):
    r_ret = eax
    r_scratch = [ecx,edx,ebp]
    r_pres = [ebx,esi,edi]
    r_sp = esp
    r_arg = [] # type: List[Register]
    ptr_size = 4
    char_size = 1
    short_size = 2
    int_size = 4
    long_size = 4

# noinspection PyPep8Naming
class X86_64Abi(X86Abi):
    has_cmovecc = True

    r_ret = rax
    r_sp = rsp
    r_rip = rip

    ptr_size = 8
    char_size = 1
    short_size = 2
    int_size = 4
    long_size = 8

    Address = Address64
    Register = Register64
    IndirectionAdapter = IndirectionAdapter64

class SystemVAbi(X86_64Abi):
    r_scratch = [r10,r11]
    r_pres = [rbx,rbp,r12,r13,r14,r15]

    r_arg = [rdi,rsi,rdx,rcx,r8,r9]


class MicrosoftX64Abi(X86_64Abi):
    shadow = True

    r_scratch = [r10,r11]
    r_pres = [rbx,rsi,rdi,rbp,r12,r13,r14,r15]
    r_arg = [rcx,rdx,r8,r9]
