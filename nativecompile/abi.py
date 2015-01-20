#  Copyright 2015 Rouslan Korneychuk
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


import itertools

from . import pyinternals
from . import x86_ops
from . import x86_64_ops


def fits_imm32(abi,x):
    """Return True if x fits in a 32-bit immediate value without sign-extend.

    32-bit immediate values are interpreted as signed. In 64-bit mode, these
    values get sign-extended to 64 bits and thus have their binary
    representation altered, which can make a difference when comparing
    addresses.

    """
    if abi.ptr_size == 8:
        return x86_ops.fits_in_sdword(x)

    return True

def string_addr(x):
    if isinstance(x,str): return pyinternals.raw_addresses[x]
    return x

class Listify:
    def __init__(self,op,name):
        self.op = op
        self.name = name
    
    def __call__(self,*args):
        return [self.op(*[string_addr(a) for a in args])]
    
    def __repr__(self):
        return '<{} op>'.format(self.name)
    
    def __hash__(self):
        return hash(self.op)
    
    def __eq__(self,b):
        if isinstance(b,Listify):
            return self.op is b.op
        
        return NotImplemented
    
    def __ne__(self,b):
        if isinstance(b,Listify):
            return self.op is not b.op
        
        return NotImplemented


def string_addr_method(m):
    return lambda self,*args: m(self,*[string_addr(a) for a in args])

class OpAbstractor:
    def __init__(self,abi,op):
        self.abi = abi
        self.op = op
    
    @string_addr_method
    def add(self,a,b):
        if a == 1 and not self.abi.tuning.prefer_addsub_over_incdec:
            if isinstance(b,self.abi.Address):
                return [self.op.incl(b)]
            return [self.op.inc(b)]

        if isinstance(a,int) and isinstance(b,self.abi.Address):
            return [(self.op.addq if self.abi.ptr_size == 8 else self.op.addl)(a,b)]
        return [self.op.add(a,b)]

    @string_addr_method
    def sub(self,a,b):
        if a == 1 and not self.abi.tuning.prefer_addsub_over_incdec:
            if isinstance(b,self.abi.Address):
                return [self.op.decl(b)]
            return [self.op.dec(b)]

        if isinstance(a,int) and isinstance(b,self.abi.Address):
            return [(self.op.subq if self.abi.ptr_size == 8 else self.op.subl)(a,b)]
        return [self.op.sub(a,b)]

    # Note: this function assumes little-endian format
    @string_addr_method
    def mov(self,a,b):
        if a != b:
            if a == 0 and isinstance(b,self.abi.Register):
                return [self.op.xor(b,b)]
            if isinstance(a,int) and isinstance(b,self.abi.Address):
                if self.abi.ptr_size == 8:
                    if x86_ops.fits_in_sdword(a):
                        return [self.op.movq(a,b)]
                    return [self.op.movl(a & 0xffffffff,b),self.op.movl(a >> 32,b+4)]
                return [self.op.movl(a,b)]
            return [self.op.mov(a,b)]

        return []

    @string_addr_method
    def cmp(self,a,b):
        assert (not isinstance(a,int)) or fits_imm32(self.abi,a)

        if isinstance(a,int) and isinstance(b,self.abi.Address):
            return [(self.op.cmpq if self.abi.ptr_size == 8 else self.op.cmpl)(a,b)]
        return [self.op.cmp(a,b)]
    
    def __getattr__(self,name):
        op = getattr(self.op,name)
        return Listify(op,name) if callable(op) else op


class Tuning:
    prefer_addsub_over_incdec = True
    build_seq_loop_threshhold = 5
    unpack_seq_loop_threshhold = 5
    build_set_loop_threshhold = 5
    mem_copy_loop_threshhold = 9


class Abi:
    has_cmovecc = False
    
    def __init__(self,*,assembly=False):
        self.tuning = Tuning()
        self.assembly = assembly
    
    def comment(self,x,*args):
        return [self._op.Assembly().comment(x.format(*args) if args else x)] if self.assembly else []
    
    @property
    def op(self):
        return OpAbstractor(self,self._op.Assembly() if self.assembly else self._op)

class CdeclAbi(Abi):
    _op = x86_ops
    
    shadow = 0
    r_ret = x86_ops.eax
    r_scratch = [x86_ops.ecx,x86_ops.edx,x86_ops.ebp]
    r_pres = [x86_ops.ebx,x86_ops.esi,x86_ops.edi]
    r_sp = x86_ops.esp
    r_arg = []
    ptr_size = 4
    char_size = 1
    short_size = 2
    int_size = 4
    long_size = 4
    
    Address = x86_ops.Address
    Register = x86_ops.Register
    Displacement = x86_ops.Displacement
    Test = x86_ops.Test


class X86_64Abi(Abi):
    has_cmovecc = True
    
    _op = x86_64_ops
    
    r_ret = x86_64_ops.rax
    r_sp = x86_64_ops.rsp
    r_rip = x86_64_ops.rip
    
    ptr_size = 8
    char_size = 1
    short_size = 2
    int_size = 4
    long_size = 8
    
    Address = x86_64_ops.Address
    Register = x86_64_ops.Register
    Displacement = x86_64_ops.Displacement
    Test = x86_64_ops.Test

class SystemVAbi(X86_64Abi):
    shadow = 0

    r_scratch = [x86_64_ops.r10,x86_64_ops.r11]
    r_pres = [x86_64_ops.rbx,x86_64_ops.rbp,x86_64_ops.r12,x86_64_ops.r13,x86_64_ops.r14,x86_64_ops.r15]
    
    r_arg = [x86_64_ops.rdi,x86_64_ops.rsi,x86_64_ops.rdx,x86_64_ops.rcx,x86_64_ops.r8,x86_64_ops.r9]


class MicrosoftX64Abi(X86_64Abi):
    shadow = 32

    r_scratch = [x86_64_ops.r10,x86_64_ops.r11]
    r_pres = [x86_64_ops.rbx,x86_64_ops.rsi,x86_64_ops.rdi,x86_64_ops.rbp,x86_64_ops.r12,x86_64_ops.r13,x86_64_ops.r14,x86_64_ops.r15]
    r_arg = [x86_64_ops.rcx,x86_64_ops.rdx,x86_64_ops.r8,x86_64_ops.r9]


for nm in itertools.chain.from_iterable(x86_ops.TEST_MNEMONICS):
    attr = 'test_'+nm.upper()
    t = getattr(x86_ops,attr)
    setattr(CdeclAbi,attr,t)
    setattr(X86_64Abi,attr,t)

del t
del attr
