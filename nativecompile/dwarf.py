
import struct

from . import elf


MODE_32 = 32
MODE_64 = 64

DWARF_VERSION = 4

COMP_UNIT_SIZE_32_FMT = struct.Struct('<I')
COMP_UNIT_HEADER_32_FMT = struct.Struct('<HIB')

COMP_UNIT_SIZE_64_FMT = struct.Struct('<4sQ')
COMP_UNIT_HEADER_64_FMT = struct.Struct('<HQB')

class Enum:
    def __init__(self,name,val):
        assert isinstance(val,int)
        self.name = name
        self.val = val
        setattr(self.__class__,name,self)

    def __str__(self): return self.name
    def __int__(self): return self.val


class TAG(Enum): pass

TAG('compile_unit',0x11)
TAG('subprogram',0x2e)

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


def make_form(name,base,index,encode):
    name = 'FORM_'+name
    globals()[name] = type(name,(base,),{'index':index,'encode':encode})

make_form('addr',CLASS_address,0x1,(lambda self,mode: elf.RelocAddress(self.val)))
make_form('flag',CLASS_flag,0xc,(lambda self,mode: b'\1' if self.val else b'\0'))
make_form('strp',CLASS_string,0xe,(lambda self,mode: self.val.to_bytes(mode//8,byteorder='little',signed=False)))
make_form('sdata',CLASS_constant,0xd,(lambda self,mode: encode_s_leb128(self.val)))
make_form('udata',CLASS_constant,0xf,(lambda self,mode: encode_u_leb128(self.val)))

# special case: no value
class FORM_flag_present(CLASS_flag):
    index = 0x19
    def __init__(self):
        self.val = None

    def encode(self,mode):
        return b''


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


def encode_u_leb128(x):
    assert x >= 0
    r = bytearray()
    while True:
        b = x & 0x7f
        x >>= 7
        if x: b |= 0x80
        r.append(b)
        if not x: return r

def encode_s_leb128(x):
    r = bytearray()
    more = True
    while more:
        b = x & 0x7f
        x >>= 7
        if (x and not b & 0x40) or (x == -1 and b & 0x40):
            more = False
        else:
            b |= 0x80
        r.append(b)
    return r


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
    """A DWARF debugging information entry.

    "children" may be either a list of DIE objects or "True". Normally a DIE
    uses a different abbreviation declaration depending on whether it has
    children or not. Setting this to "True" will cause it to use the same
    declaration as an equivalent DIE with children, which may save a few bytes.

    """
    def __init__(self,tag):
        self.__dict__['tag'] = tag
        self.__dict__['children'] = []
        self.__dict__['attr'] = {}

    def __getattr__(self,name):
        return self.attr[getattr(AT,name)]

    def __setattr__(self,name,value):
        at = getattr(AT,name)
        assert isinstance(value,at.types)
        self.attr[at] = value

    def __delattr__(self,name):
        del self.attr[getattr(AT,name)]

    def signature(self):
        return (
            self.tag,
            bool(self.children),
            tuple((n,v.index) for n,v in self.attr.items()))

    def __str__(self,indent=0):
        space = ' '*indent
        r = '{}{}:\n'.format(space,self.tag)
        for n,v in self.attr.items():
            r += '{}    {}: {}\n'.format(space,n,v)
        if self.children is not True:
            for c in self.children:
                r += c.__str__(indent+4)

        return r



class DebugInfo:
    def __init__(self,mode,data):
        assert mode in (MODE_32,MODE_64)

        self.mode = mode
        self.data = data
        self.types = {}
        self.type_list = []

        def get_types(die):
            s = die.signature()
            i = self.types.get(s)
            if not i:
                self.type_list.append(s)
                i = len(self.type_list)
                self.types[s] = i
            if die.children is not True: # see the class DIE above
                for c in die.children: get_types(c)

        get_types(data)


    def write_debug_info(self,out):
        start = out.tell()

        if self.mode == MODE_64:
            out.seek(start + COMP_UNIT_SIZE_64_FMT.size)
            out.write(COMP_UNIT_HEADER_64_FMT.pack(DWARF_VERSION,0,8))
        else:
            out.seek(start + COMP_UNIT_SIZE_32_FMT.size)
            out.write(COMP_UNIT_HEADER_32_FMT.pack(DWARF_VERSION,0,4))

        def write_die(die):
            abbr_code = self.types.get(die.signature())
            assert abbr_code is not None
            out.write(encode_u_leb128(abbr_code))
            for v in die.attr.values():
                out.write(v.encode(self.mode))

            if die.children:
                if die.children is not True: # see the class DIE above
                    for c in die.children: write_die(c)
                out.write(b'\0')

        write_die(self.data)

        end = out.tell()
        size = end - start
        out.seek(start)
        if self.mode == MODE_64:
            size -= COMP_UNIT_SIZE_64_FMT.size
            out.write(COMP_UNIT_SIZE_64_FMT.pack(b'\xff\xff\xff\xff',size))
        else:
            size -= COMP_UNIT_SIZE_32_FMT.size

            # values 0xfffffff0 to 0xffffffff have special meaning
            if size >= 0xfffffff0:
                raise Exception('The debug information is too large to be encoded in one section')

            out.write(COMP_UNIT_SIZE_32_FMT.pack(size))

        out.seek(end)

    def write_debug_abbrev(self,out):
        for i,t in enumerate(self.type_list):
            tag,has_child,attr = t
            
            out.write(encode_u_leb128(i+1))
            out.write(encode_u_leb128(int(tag)))
            out.write(CHILDREN_yes if has_child else CHILDREN_no)
            
            for name,form in attr:
                out.write(encode_u_leb128(int(name)))
                out.write(encode_u_leb128(form))
            out.write(b'\0\0')


class DebugInfoSection(elf.Section):
    name = b'.debug_info'
    type = elf.SHT_PROGBITS

    def __init__(self,info):
        self.dinfo = info

    def write(self,out):
        self.dinfo.write_debug_info(out)

class DebugAbbrevSection(elf.Section):
    name = b'.debug_abbrev'
    type = elf.SHT_PROGBITS

    def __init__(self,info):
        self.dinfo = info

    def write(self,out):
        self.dinfo.write_debug_abbrev(out)

class DebugStrSection(elf.Section):
    name = b'.debug_str'
    type = elf.SHT_PROGBITS

    def __init__(self,strtab):
        self.s = strtab

    def write(self,out):
        self.s.write_to_file(out)


def elf_sections(mode,data,strings=None):
    info = DebugInfo(mode,data)
    r = [DebugInfoSection(info),DebugAbbrevSection(info)]
    if strings: r.append(DebugStrSection(strings))
    return r
