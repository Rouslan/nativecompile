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


import struct

from . import elf
from .reloc_buffer import RelocAbsAddress


MODE_32 = 32
MODE_64 = 64

DWARF_VERSION = 4
ARANGES_VERSION = 2
CI_VERSION = 4

COMP_UNIT_SIZE_32_FMT = struct.Struct('<I')
COMP_UNIT_HEADER_32_FMT = struct.Struct('<HIB')
ARANGES_HEADER_32_FMT = struct.Struct('<HIBB')

COMP_UNIT_SIZE_64_FMT = struct.Struct('<4sQ')
COMP_UNIT_HEADER_64_FMT = struct.Struct('<HQB')
ARANGES_HEADER_64_FMT = struct.Struct('<HQBB')


class Enum:
    def __init__(self,name,val):
        assert isinstance(val,int)
        self.name = name
        self.val = val
        setattr(self.__class__,name,self)

    def __str__(self): return self.name
    def __int__(self): return self.val


def enc_uleb128(x):
    assert x >= 0
    r = bytearray()
    while True:
        b = x & 0x7f
        x >>= 7
        if x: b |= 0x80
        r.append(b)
        if not x: return r

def enc_sleb128(x):
    r = bytearray()
    more = True
    while more:
        b = x & 0x7f
        x >>= 7
        if (x == 0 and not b & 0x40) or (x == -1 and b & 0x40):
            more = False
        else:
            b |= 0x80
        r.append(b)
    return r


class TAG(Enum): pass

TAG('array_type',0x01)
TAG('class_type',0x02)
TAG('entry_point',0x03)
TAG('enumeration_type',0x04)
TAG('formal_parameter',0x05)
TAG('imported_declaration',0x08)
TAG('label',0x0a)
TAG('lexical_block',0x0b)
TAG('member',0x0d)
TAG('pointer_type',0x0f)
TAG('reference_type',0x10)
TAG('compile_unit',0x11)
TAG('string_type',0x12)
TAG('structure_type',0x13)
TAG('subroutine_type',0x15)
TAG('typedef',0x16)
TAG('union_type',0x17)
TAG('unspecified_parameters',0x18)
TAG('variant',0x19)
TAG('common_block',0x1a)
TAG('common_inclusion',0x1b)
TAG('inheritance',0x1c)
TAG('inlined_subroutine',0x1d)
TAG('module',0x1e)
TAG('ptr_to_member_type',0x1f)
TAG('set_type',0x20)
TAG('subrange_type',0x21)
TAG('with_stmt',0x22)
TAG('access_declaration',0x23)
TAG('base_type',0x24)
TAG('catch_block',0x25)
TAG('const_type',0x26)
TAG('constant',0x27)
TAG('enumerator',0x28)
TAG('file_type',0x29)
TAG('friend',0x2a)
TAG('namelist',0x2b)
TAG('namelist_item',0x2c)
TAG('packed_type',0x2d)
TAG('subprogram',0x2e)
TAG('template_type_parameter',0x2f)
TAG('template_value_parameter',0x30)
TAG('thrown_type',0x31)
TAG('try_block',0x32)
TAG('variant_part',0x33)
TAG('variable',0x34)
TAG('volatile_type',0x35)
TAG('dwarf_procedure',0x36)
TAG('restrict_type',0x37)
TAG('interface_type',0x38)
TAG('namespace',0x39)
TAG('imported_module',0x3a)
TAG('unspecified_type',0x3b)
TAG('partial_unit',0x3c)
TAG('imported_unit',0x3d)
TAG('condition',0x3f)
TAG('shared_type',0x40)
TAG('type_unit',0x41)
TAG('rvalue_reference_type',0x42)
TAG('template_alias',0x43)


CHILDREN_yes = b'\1'
CHILDREN_no = b'\0'


class FORM:
    def __init__(self,val):
        self.val = val

    def __str__(self):
        return '{} (DW_{})'.format(self.val,self.__class__.__name__)


class CLASS_address(FORM): pass
class CLASS_block(FORM): pass
class CLASS_constant(FORM): pass
class CLASS_exprloc(FORM): pass
class CLASS_flag(FORM): pass
class CLASS_lineptr(FORM): pass
class CLASS_loclistptr(FORM): pass
class CLASS_macptr(FORM): pass
class CLASS_rangelistptr(FORM): pass
class CLASS_reference(FORM): pass
class CLASS_string(FORM): pass


def form(name,base,index,encode):
    return type('FORM_'+name,(base,),{'index':index,'encode':encode})

def make_form(*args):
    f = form(*args)
    globals()[f.__name__] = f

def enc_block_n(size):
    def encode(self,d):
        assert isinstance(self.val,(bytes,bytearray)) and len(self.val) == size
        return self.val
    return encode

def enc_block(self,d):
    assert isinstance(self.val,(bytes,bytearray))
    return enc_uleb128(len(self.val)) + self.val

def enc_ref(self,d):
    return d.mode.enc_int(self.val,d.mode.ref_size)

def get_ref(self,d):
    r = d.refmap.get(id(self.val))
    assert r is not None, 'Only back-references are supported' # for now
    return r

make_form('addr',CLASS_address,0x01,(lambda self,d: RelocAbsAddress(self.val)))
make_form('block2',CLASS_block,0x03,enc_block_n(2))
make_form('block4',CLASS_block,0x04,enc_block_n(4))
make_form('data2',CLASS_constant,0x05,(lambda self,d: d.mode.enc_int(self.val,2)))
make_form('data4',CLASS_constant,0x06,(lambda self,d: d.mode.enc_int(self.val,4)))
make_form('data8',CLASS_constant,0x07,(lambda self,d: d.mode.enc_int(self.val,8)))
make_form('block',CLASS_block,0x09,enc_block)
make_form('block1',CLASS_block,0x0a,enc_block_n(1))
make_form('data1',CLASS_constant,0x0b,(lambda self,d: d.mode.enc_int(self.val,1)))
make_form('flag',CLASS_flag,0x0c,(lambda self,d: b'\1' if self.val else b'\0'))
make_form('strp',CLASS_string,0x0e,enc_ref)
make_form('sdata',CLASS_constant,0x0d,(lambda self,d: enc_sleb128(self.val)))
make_form('udata',CLASS_constant,0x0f,(lambda self,d: enc_uleb128(self.val)))
make_form('ref1',CLASS_reference,0x11,(lambda self,d: d.mode.enc_int(get_ref(self,d),1)))
make_form('ref2',CLASS_reference,0x12,(lambda self,d: d.mode.enc_int(get_ref(self,d),2)))
make_form('ref4',CLASS_reference,0x13,(lambda self,d: d.mode.enc_int(get_ref(self,d),4)))
make_form('ref8',CLASS_reference,0x14,(lambda self,d: d.mode.enc_int(get_ref(self,d),8)))
make_form('ref_udata',CLASS_reference,0x15,(lambda self,d: enc_uleb128(get_ref(self,d))))


SEC_OFFSET_CLASSES = CLASS_lineptr,CLASS_loclistptr,CLASS_macptr,CLASS_rangelistptr

class FORM_sec_offset():
    index = 0x17

    def __init__(self,val):
        self.val = val

    def encode(self,d):
        return d.mode.enc_int(d.sec_offset_maps[self.sec_offset_index][self],d.mode.ref_size)

class FORM_line_offset(FORM_sec_offset,CLASS_lineptr):
    def __init__(self,val):
        assert False,'Not implemented'

class FORM_loclist_offset(FORM_sec_offset,CLASS_loclistptr):
    def values(self,op):
        return self.val(op)

class FORM_mac_offset(FORM_sec_offset,CLASS_macptr):
    def __init__(self,val):
        assert False,'Not implemented'

class FORM_rangelist_offset(FORM_sec_offset,CLASS_rangelistptr):
    def values(self):
        return self.val()

SEC_OFFSET_FORMS = FORM_line_offset,FORM_loclist_offset,FORM_mac_offset,FORM_rangelist_offset
for i,sof in enumerate(SEC_OFFSET_FORMS):
    sof.sec_offset_index = i


make_form('exprloc',CLASS_exprloc,0x18,enc_block)

# special case: no value
class FORM_flag_present(CLASS_flag):
    index = 0x19
    def __init__(self):
        self.val = None

    def encode(self,e):
        return b''


def smallest_block_form(val):
    if len(val) == 1: return FORM_block1(val)
    if len(val) == 2: return FORM_block2(val)
    if len(val) == 4: return FORM_block4(val)
    if len(val) == 8: return FORM_block8(val)
    return FORM_block(val)


class ATE(Enum,FORM_data1):
    __init__ = Enum.__init__

ATE('address',0x01)
ATE('boolean',0x02)
ATE('complex_float',0x03)
ATE('float',0x04)
ATE('signed',0x05)
ATE('signed_char',0x06)
ATE('unsigned',0x07)
ATE('unsigned_char',0x08)
ATE('imaginary_float',0x09)
ATE('packed_decimal',0x0a)
ATE('numeric_string',0x0b)
ATE('edited',0x0c)
ATE('signed_fixed',0x0d)
ATE('unsigned_fixed',0x0e)
ATE('decimal_float',0x0f)
ATE('UTF',0x10)


class AT(Enum):
    def __init__(self,name,val,types):
        super().__init__(name,val)
        self.types = types

# hooray for sed
AT('sibling',0x01,CLASS_reference)
AT('location',0x02,(CLASS_exprloc,CLASS_loclistptr))
AT('name',0x03,CLASS_string)
AT('ordering',0x09,CLASS_constant)
AT('byte_size',0x0b,(CLASS_constant,CLASS_exprloc,CLASS_reference))
AT('bit_offset',0x0c,(CLASS_constant,CLASS_exprloc,CLASS_reference))
AT('bit_size',0x0d,(CLASS_constant,CLASS_exprloc,CLASS_reference))
AT('stmt_list',0x10,CLASS_lineptr)
AT('low_pc',0x11,CLASS_address)
AT('high_pc',0x12,(CLASS_address,CLASS_constant))
AT('language',0x13,CLASS_constant)
AT('discr',0x15,CLASS_reference)
AT('discr_value',0x16,CLASS_constant)
AT('visibility',0x17,CLASS_constant)
AT('import',0x18,CLASS_reference)
AT('string_length',0x19,(CLASS_exprloc,CLASS_loclistptr))
AT('common_reference',0x1a,CLASS_reference)
AT('comp_dir',0x1b,CLASS_string)
AT('const_value',0x1c,(CLASS_block,CLASS_constant,CLASS_string))
AT('containing_type',0x1d,CLASS_reference)
AT('default_value',0x1e,CLASS_reference)
AT('inline',0x20,CLASS_constant)
AT('is_optional',0x21,CLASS_flag)
AT('lower_bound',0x22,(CLASS_constant,CLASS_exprloc,CLASS_reference))
AT('producer',0x25,CLASS_string)
AT('prototyped',0x27,CLASS_flag)
AT('return_addr',0x2a,(CLASS_exprloc,CLASS_loclistptr))
AT('start_scope',0x2c,(CLASS_constant,CLASS_rangelistptr))
AT('bit_stride',0x2e,(CLASS_constant,CLASS_exprloc,CLASS_reference))
AT('upper_bound',0x2f,(CLASS_constant,CLASS_exprloc,CLASS_reference))
AT('abstract_origin',0x31,CLASS_reference)
AT('accessibility',0x32,CLASS_constant)
AT('address_class',0x33,CLASS_constant)
AT('artificial',0x34,CLASS_flag)
AT('base_types',0x35,CLASS_reference)
AT('calling_convention',0x36,CLASS_constant)
AT('count',0x37,(CLASS_constant,CLASS_exprloc,CLASS_reference))
AT('data_member_location',0x38,(CLASS_constant,CLASS_exprloc,CLASS_loclistptr))
AT('decl_column',0x39,CLASS_constant)
AT('decl_file',0x3a,CLASS_constant)
AT('decl_line',0x3b,CLASS_constant)
AT('declaration',0x3c,CLASS_flag)
AT('discr_list',0x3d,CLASS_block)
AT('encoding',0x3e,CLASS_constant)
AT('external',0x3f,CLASS_flag)
AT('frame_base',0x40,(CLASS_exprloc,CLASS_loclistptr))
AT('friend',0x41,CLASS_reference)
AT('identifier_case',0x42,CLASS_constant)
AT('macro_info',0x43,CLASS_macptr)
AT('namelist_item',0x44,CLASS_reference)
AT('priority',0x45,CLASS_reference)
AT('segment',0x46,(CLASS_exprloc,CLASS_loclistptr))
AT('specification',0x47,CLASS_reference)
AT('static_link',0x48,(CLASS_exprloc,CLASS_loclistptr))
AT('type',0x49,CLASS_reference)
AT('use_location',0x4a,(CLASS_exprloc,CLASS_loclistptr))
AT('variable_parameter',0x4b,CLASS_flag)
AT('virtuality',0x4c,CLASS_constant)
AT('vtable_elem_location',0x4d,(CLASS_exprloc,CLASS_loclistptr))
AT('allocated',0x4e,(CLASS_constant,CLASS_exprloc,CLASS_reference))
AT('associated',0x4f,(CLASS_constant,CLASS_exprloc,CLASS_reference))
AT('data_location',0x50,CLASS_exprloc)
AT('byte_stride',0x51,(CLASS_constant,CLASS_exprloc,CLASS_reference))
AT('entry_pc',0x52,CLASS_address)
AT('use_UTF8',0x53,CLASS_flag)
AT('extension',0x54,CLASS_reference)
AT('ranges',0x55,CLASS_rangelistptr)
AT('trampoline',0x56,(CLASS_address,CLASS_flag,CLASS_reference,CLASS_string))
AT('call_column',0x57,CLASS_constant)
AT('call_file',0x58,CLASS_constant)
AT('call_line',0x59,CLASS_constant)
AT('description',0x5a,CLASS_string)
AT('binary_scale',0x5b,CLASS_constant)
AT('decimal_scale',0x5c,CLASS_constant)
AT('small',0x5d,CLASS_reference)
AT('decimal_sign',0x5e,CLASS_constant)
AT('digit_count',0x5f,CLASS_constant)
AT('picture_string',0x60,CLASS_string)
AT('mutable',0x61,CLASS_flag)
AT('threads_scaled',0x62,CLASS_flag)
AT('explicit',0x63,CLASS_flag)
AT('object_pointer',0x64,CLASS_reference)
AT('endianity',0x65,CLASS_constant)
AT('elemental',0x66,CLASS_flag)
AT('pure',0x67,CLASS_flag)
AT('recursive',0x68,CLASS_flag)
AT('signature',0x69,CLASS_reference)
AT('main_subprogram',0x6a,CLASS_flag)
AT('data_bit_offset',0x6b,CLASS_constant)
AT('const_expr',0x6c,CLASS_flag)
AT('enum_class',0x6d,CLASS_flag)
AT('linkage_name',0x6e,CLASS_string)


def op_method(code,enc1=None,enc2=None):
    code = bytes((code,))
    if enc1 is None:
        r = lambda: code
    elif enc2 is None:
        r = lambda val: code + enc1(val)
    else:
        r = lambda val1,val2: code + enc1(val1) + enc2(val2)

    return staticmethod(r)

def enc_reg(reg):
    return enc_uleb128(int(reg) if isinstance(reg,Register) else reg)

def op_constant(code,size,signed):
    code = bytes((code,))
    return lambda self,val: code + self.mode.enc_int(val,size,signed)

def op_native_val(code):
    code = bytes((code,))
    return lambda self,val: code + self.mode.enc_int(val)

def op_indexed(code):
    def inner(index):
        assert 0 <= index <= 31
        return bytes((code + index,))

    return staticmethod(inner)

class OP:
    def __init__(self,mode):
        self.mode = mode

    addr = op_native_val(0x03)
    deref = op_method(0x06)
    const1u = op_constant(0x08,1,False)
    const1s = op_constant(0x09,1,True)
    const2u = op_constant(0x0a,2,False)
    const2s = op_constant(0x0b,2,True)
    const4u = op_constant(0x0c,4,False)
    const4s = op_constant(0x0d,4,True)
    const8u = op_constant(0x0e,8,False)
    const8s = op_constant(0x0f,8,True)
    constu = op_method(0x10,enc_uleb128)
    consts = op_method(0x11,enc_sleb128)
    dup = op_method(0x12)
    drop = op_method(0x13)
    over = op_method(0x14)
    pick = op_constant(0x15,1,False)
    swap = op_method(0x16)
    rot = op_method(0x17)
    xderef = op_method(0x18)
    abs = op_method(0x19)
    and_ = op_method(0x1a)
    div = op_method(0x1b)
    minus = op_method(0x1c)
    mod = op_method(0x1d)
    mul = op_method(0x1e)
    neg = op_method(0x1f)
    not_ = op_method(0x20)
    or_ = op_method(0x21)
    plus = op_method(0x22)
    plus_uconst = op_method(0x23,enc_uleb128)
    shl = op_method(0x24)
    shr = op_method(0x25)
    shra = op_method(0x26)
    xor = op_method(0x27)
    skip = op_constant(0x2f,2,True)
    bra = op_constant(0x28,2,True)
    eq = op_method(0x29)
    ge = op_method(0x2a)
    gt = op_method(0x2b)
    le = op_method(0x2c)
    lt = op_method(0x2d)
    ne = op_method(0x2e)

    @staticmethod
    def lit(index):
        assert 0 <= index <= 31
        return bytes((0x30 + index,))

    @staticmethod
    def reg(index):
        if isinstance(index,Register): index = int(index)
        assert 0 <= index <= 31
        return bytes((0x50 + index,))

    @staticmethod
    def breg(index,offset):
        if isinstance(index,Register): index = int(index)
        assert 0 <= index <= 31
        return bytes((0x70 + index,)) + enc_sleb128(offset)

    regx = op_method(0x90,enc_reg)
    fbreg = op_method(0x91,enc_sleb128)
    bregx = op_method(0x92,enc_reg,enc_sleb128)
    piece = op_method(0x93,enc_uleb128)
    deref_size = op_constant(0x94,1,False)
    xderef_size = op_constant(0x95,1,False)
    nop = op_method(0x96)
    push_object_address = op_method(0x97)
    call2 = op_constant(0x98,2,False)
    call4 = op_constant(0x99,4,False)
    call_ref = op_native_val(0x9a)
    form_tls_address = op_method(0x9b)
    call_frame_cfa = op_method(0x9c)
    bit_piece = op_method(0x9d,enc_uleb128,enc_uleb128)
    implicit_value = op_method(0x9e,enc_block)
    stack_value = op_method(0x9f)


    # helper methods:

    def const_int(self,val):
        """Return the most compact representation of the constant 'val'."""
        # actually, some values can probably be represented more compactly
        # using leb128 instead of the smallest fitting const- method, but that
        # would add encoding and decoding overhead anyway

        if 0 <= val <= 31:
            return self.lit(val)

        aval = abs(val)
        signed = val < 0
        if aval <= 0xff:
            base = 0
        elif aval <= 0xffff:
            base = 1
        elif aval <= 0xffffffff:
            base = 2
        elif aval <= 0xffffffffffffffff:
            base = 3
        else:
            return (self.consts if signed else self.constu)(val)

        return bytes((8 + 2 * base + signed,)) + self.mode.enc_int(val,1 << base,signed)



class CFA:
    def __init__(self,mode):
        self.mode = mode

    @staticmethod
    def advance_loc(val):
        assert 0 <= val < (1<<6)
        return bytes(((1<<6) | val,))

    @staticmethod
    def offset(reg,offset):
        reg = int(reg)
        assert 0 <= reg < (1<<6)
        return bytes(((2<<6) | reg,)) + enc_uleb128(offset)

    @staticmethod
    def restore(reg):
        reg = int(reg)
        assert 0 <= reg < (1<<6)
        return bytes(((3<<6) | reg,))

    nop = op_method(0x00)
    set_loc = op_native_val(0x01)
    advance_loc1 = op_constant(0x02,1,False)
    advance_loc2 = op_constant(0x03,2,False)
    advance_loc4 = op_constant(0x04,4,False)
    offset_extended = op_method(0x05,enc_reg,enc_uleb128)
    restore_extended = op_method(0x06,enc_reg)
    undefined = op_method(0x07,enc_reg)
    same_value = op_method(0x08,enc_reg)
    register = op_method(0x09,enc_reg,enc_reg)
    remember_state = op_method(0x0a)
    restore_state = op_method(0x0b)
    def_cfa = op_method(0x0c,enc_reg,enc_uleb128)
    def_cfa_register = op_method(0x0d,enc_reg)
    def_cfa_offset = op_method(0x0e,enc_uleb128)
    def_cfa_expression = op_method(0x0f,enc_block)
    expression = op_method(0x10,enc_reg,enc_block)
    offset_extended_sf = op_method(0x11,enc_reg,enc_sleb128)
    def_cfa_sf = op_method(0x12,enc_reg,enc_sleb128)
    def_cfa_offset_sf = op_method(0x13,enc_sleb128)
    val_offset = op_method(0x14,enc_uleb128,enc_uleb128)
    val_offset_sf = op_method(0x15,enc_uleb128,enc_sleb128)
    val_expression = op_method(0x16,enc_uleb128,enc_block)

    # helper methods:

    def advance_loc_smallest(self,amount):
        if amount < (1 << 6):
            return self.advance_loc(amount)
        if amount <= 0xff:
            return self.advance_loc1(amount)
        if amount <= 0xffff:
            return self.advance_loc2(amount)
        if amount <= 0xffffffff:
            return self.advance_loc4(amount)

        r = b''
        while amount > 0xffffffff:
            r += b'\x04\xff\xff\xff\xff'
            amount -= 0xffffffff
        if amount: r += self.advance_loc_smallest(amount)
        return r


class Register(Enum): pass

class Reg32(Register): pass

Reg32('eax',0)
Reg32('ecx',1)
Reg32('edx',2)
Reg32('ebx',3)
Reg32('esp',4)
Reg32._sp = Reg32.esp
Reg32('ebp',5)
Reg32._bp = Reg32.ebp
Reg32('esi',6)
Reg32('edi',7)
Reg32('RA',8)
Reg32('eFLAGS',9)
Reg32._FLAGS = Reg32.eFLAGS
for i in range(8):
    Reg32('st' + str(i),i + 16)
for i in range(8):
    Reg32('xmm' + str(i),i + 32)


class Reg64(Register): pass

Reg64('rax',0)
Reg64('rdx',1)
Reg64('rcx',2)
Reg64('rbx',3)
Reg64('rsi',4)
Reg64('rdi',5)
Reg64('rbp',6)
Reg64._bp = Reg64.rbp
Reg64('rsp',7)
Reg64._sp = Reg64.rsp
for i in range(8,16):
    Reg64('r' + str(i),i)
Reg64('RA',16)
for i in range(16):
    Reg64('xmm' + str(i),i + 17)
for i in range(8):
    Reg64('st' + str(i),i + 33)
Reg64('rFLAGS',49)
Reg64._FLAGS = Reg64.rFLAGS


def reg(mode):
    return Reg64 if mode.ptr_size == 8 else Reg32


class StringTable:
    def __init__(self):
        self.entries = {}
        self.ordered = []
        self.len = 0

    def __getitem__(self,k):
        if not isinstance(k,bytes):
            k = k.encode()

        r = self.entries.get(k)
        if r is None:
            r = FORM_strp(self.len)
            self.len += len(k) + 1
            self.entries[k] = r
            self.ordered.append(k)
        return r

    def __len__(self):
        return self.len

    def write_to_file(self,f):
        for s in self.ordered:
            f.write(s)
            f.write(b'\0')


class DIE:
    """A DWARF debugging information entry."""

    def __init__(self,tag,**attrvals):
        if isinstance(tag,str):
            try:
                tag = getattr(TAG,tag)
            except AttributeError:
                raise ValueError('"{}" is not a DWARF tag'.format(tag))

        self.__dict__['tag'] = tag
        self.__dict__['children'] = []
        self.__dict__['attr'] = {}

        for n,v in attrvals.items():
            self.__setattr__(n,v)

    def __getattr__(self,name):
        return self.attr[getattr(AT,name)]

    def __setattr__(self,name,value):
        at = getattr(AT,name)
        assert isinstance(value,at.types)
        self.attr[at] = value

    def __delattr__(self,name):
        del self.attr[getattr(AT,name)]

    def signature(self):
        return self.tag,tuple((n,v.index) for n,v in self.attr.items())

    def __str__(self,indent=0):
        space = ' '*indent
        r = '{}{}:\n'.format(space,self.tag)
        for n,v in self.attr.items():
            r += '{}    {}: {}\n'.format(space,n,v)
        for c in self.children:
            r += c.__str__(indent+4)

        return r


class Mode:
    def __init__(self,mode,ptr_size):
        assert mode in (MODE_32,MODE_64)
        self.mode = mode
        self.ptr_size = ptr_size
        self.seg_size = 0
        self.byteorder = 'little'

    @property
    def ref_size(self):
        return self.mode // 8

    def enc_int(self,x,size=None,signed=False):
        if size is None: size = self.ptr_size
        return x.to_bytes(size,byteorder=self.byteorder,signed=signed)


def sized_section(f):
    def inner(self,out,*args,**kwds):
        unit_size = COMP_UNIT_SIZE_64_FMT if self.mode.mode == MODE_64 else COMP_UNIT_SIZE_32_FMT
        start = out.tell()
        out.seek(start + unit_size.size)

        f(self,out,*args,**kwds)

        end = out.tell()
        size = end - start - unit_size.size
        out.seek(start)

        if self.mode.mode == MODE_64:
            out.write(COMP_UNIT_SIZE_64_FMT.pack(b'\xff\xff\xff\xff',size))
        else:
            # values 0xfffffff0 to 0xffffffff have special meaning
            if size >= 0xfffffff0:
                raise Exception('The debug information is too large to be encoded in one section')

            out.write(COMP_UNIT_SIZE_32_FMT.pack(size))

        out.seek(end)

    return inner

class TypeData:
    def __init__(self,sig,has_child,index):
        self.tag,self.attr = sig
        self.has_child = has_child
        self.index = index

class DebugInfo:
    def __init__(self,mode,data,call_frames=None,pres_regs=None):
        self.mode = mode
        self.data = data
        self.call_frames = call_frames
        self.pres_regs = pres_regs or []
        self.types = {}
        self.type_list = []
        self.refmap = {}
        self.sec_offset_maps = [{} for x in SEC_OFFSET_CLASSES]

        def get_types(die):
            s = die.signature()
            ti = self.types.get(s)
            if not ti:
                ti = TypeData(s,len(die.children) > 0,len(self.type_list)+1)
                self.type_list.append(ti)
                self.types[s] = ti
            elif die.children:
                ti.has_child = True

            for a in die.attr.values():
                if isinstance(a,SEC_OFFSET_FORMS):
                    # the offset is set to None for now and is given a value
                    # in one of the write_debug_- methods
                    self.sec_offset_maps[a.sec_offset_index][a] = None

            for c in die.children: get_types(c)

        get_types(data)

    loclist = property(lambda self: self.sec_offset_maps[FORM_loclist_offset.sec_offset_index])
    rangelist = property(lambda self: self.sec_offset_maps[FORM_rangelist_offset.sec_offset_index])

    @sized_section
    def _write_debug_info(self,out,start):
        unit_header = (
            COMP_UNIT_HEADER_64_FMT
                if self.mode.mode == MODE_64 else
            COMP_UNIT_HEADER_32_FMT)

        out.write(unit_header.pack(DWARF_VERSION,0,self.mode.ptr_size))

        def write_die(die):
            self.refmap[id(die)] = out.tell() - start

            ti = self.types.get(die.signature())
            assert ti is not None
            out.write(enc_uleb128(ti.index))
            for v in die.attr.values():
                out.write(v.encode(self))

            if ti.has_child:
                for c in die.children: write_die(c)
                out.write(b'\0')

        write_die(self.data)

    def write_debug_info(self,out):
        self._write_debug_info(out,out.tell())

    def write_debug_abbrev(self,out):
        for ti in self.type_list:
            out.write(enc_uleb128(ti.index))
            out.write(enc_uleb128(int(ti.tag)))
            out.write(CHILDREN_yes if ti.has_child else CHILDREN_no)

            for name,form in ti.attr:
                out.write(enc_uleb128(int(name)))
                out.write(enc_uleb128(form))
            out.write(b'\0\0')
        out.write(b'\0')

    #@sized_section
    #def write_debug_aranges(self,out):
    #    head = (
    #        ARANGES_HEADER_64_FMT
    #            if self.mode.mode == MODE_64 else
    #        ARANGES_HEADER_32_FMT)

    #    out.write(head.pack(ARANGES_VERSION,0,self.mode.ptr_size,0))
    #    elf.align_fo(out,self.mode.ptr_size * 2)

    def write_debug_ranges(self,out):
        baseloc = out.tell()
        mapping = self.rangelist

        for ar in mapping:
            mapping[ar] = out.tell() - baseloc

            for addr,size in ar.values():
                out.write(RelocAbsAddress(addr))
                out.write(RelocAbsAddress(size))
            out.write(b'\0' * (self.mode.ptr_size * 2))

    def write_debug_loclist(self,out):
        baseloc = out.tell()
        mapping = self.loclist
        op = OP(self.mode)

        for ll in mapping:
            mapping[ll] = out.tell() - baseloc

            for start,end,expr in ll.values(op):
                assert end >= start
                # these addresses are not supposed to be absolute addresses
                # even in an in-memory object file
                out.write(self.mode.enc_int(start))
                out.write(self.mode.enc_int(end))

                if len(expr) > 0xffff:
                    raise ValueError(
                        'The location description at {:x}-{:x} is too long (65535 bytes is the maximum).'
                            .format(start,end))
                out.write(self.mode.enc_int(len(expr),2))
                out.write(expr)
            out.write(b'\0' * (self.mode.ptr_size * 2))

    def _cf_pad(self,out,body_size):
        pad = (body_size + (12 if self.mode.mode == MODE_64 else 4)) % self.mode.ptr_size
        if pad != self.mode.ptr_size:
            out.write(b'\0' * pad)

    @sized_section
    def cie(self,out):
        cfa = CFA(self.mode)
        r = reg(self.mode)
        body = (b'\xff' * self.mode.ref_size
            + bytes((CI_VERSION,0,self.mode.ptr_size,self.mode.seg_size))
            + enc_uleb128(1)
            + enc_sleb128(-self.mode.ptr_size)
            + enc_uleb128(int(r.RA))
            + cfa.def_cfa(r._sp,self.mode.ptr_size)
            + cfa.offset(r.RA,1))
        for pr in self.pres_regs:
            body += cfa.same_value(pr)

        out.write(body)
        self._cf_pad(out,len(body))

    @sized_section
    def fde(self,out,addr,range,instr):
        start = out.tell()
        out.write(b'\0' * self.mode.ref_size)
        out.write(RelocAbsAddress(addr))
        out.write(self.mode.enc_int(range))
        out.write(instr)
        self._cf_pad(out,out.tell() - start)

    def write_debug_frame(self,out):
        self.cie(out)
        assert self.call_frames is not None
        for cf in self.call_frames(CFA(self.mode),reg(self.mode)):
            self.fde(out,*cf)


class DebugSection(elf.Section):
    type = elf.SHT_PROGBITS

    def __init__(self,suffix,callback,align=0):
        self.name = b'.debug_' + suffix
        self.callback = callback
        self.align = align

    def write(self,out):
        self.callback(out)


def elf_sections(mode,data,strings=None,callframes=None,pres_regs=None):
    info = DebugInfo(mode,data,callframes,pres_regs)
    r = [DebugSection(b'info',info.write_debug_info),
        DebugSection(b'abbrev',info.write_debug_abbrev)]

    if info.loclist:
        r.insert(0,DebugSection(b'loc',info.write_debug_loclist))
    if info.rangelist:
        r.insert(0,DebugSection(b'ranges',info.write_debug_ranges))
    if info.call_frames is not None:
        r.insert(0,DebugSection(b'frame',info.write_debug_frame,mode.ptr_size))

    if strings: r.append(DebugSection(b'str',strings.write_to_file))
    return r
