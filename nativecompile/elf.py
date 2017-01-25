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


import struct
import collections
import re

from .reloc_buffer import RelocAbsAddress,RelocBuffer,NonRelocWrapper


# these must be powers of 2
CODE_ALIGN = 16
HEADER_ALIGN = 4

MODE_32 = 32
MODE_64 = 64

# for the meanings of the different fields, see the ELF specification at
# www.sco.com/developers/devspecs/gabi41.pdf and ELF-64 specification at
# downloads.openwatcom.org/ftp/devel/docs/elf-64-gen.pdf
ELF_HEADER_FMT = struct.Struct('<4s3B9x2H5I6H')
ELF64_HEADER_FMT = struct.Struct('<4s5B7x2HI3QI6H')
SECTION_HEADER_FMT = struct.Struct('<10I')
ELF64_SECTION_HEADER_FMT = struct.Struct('<2I4Q2I2Q')
#PROGRAM_HEADER_FMT = struct.Struct('<8I')
#ELF64_PROGRAM_HEADER_FMT = struct.Struct('2I6Q')
SYMBOL_TABLE_FMT = struct.Struct('<3IBxH')
ELF64_SYMBOL_TABLE_FMT = struct.Struct('<IBxH2Q')
SYMTAB_PREADDR = struct.calcsize('<I')
ELF64_SYMTAB_PREADDR = struct.calcsize('<IBxH')

ET_REL = 1
ET_EXEC = 2
ET_DYN = 3

EM_386 = 3
EM_X86_64 = 62

EV_CURRENT = 1

ELFCLASS32 = 1
ELFCLASS64 = 2

ELFDATA2LSB = 1
ELFDATA2MSB = 2

ELFOSABI_SYSV = 0
ELFOSABI_HPUX = 1
ELFOSABI_STANDALONE = 255


PT_DYNAMIC = 2


SHN_UNDEF = 0

SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_HASH = 5
SHT_DYNAMIC = 6
SHT_NOTE = 7
SHT_NOBITS = 8
SHT_REL = 9
SHT_SHLIB = 10
SHT_DYNSYM = 11

SHF_ALLOC = 2
SHF_EXCINSTR = 4
SHF_X86_64_LARGE = 0x10000000

STB_LOCAL = 0
STB_GLOBAL = 1
STB_WEAK = 2

STT_NOTYPE = 0
STT_OBJECT = 1
STT_FUNC = 2
STT_SECTION = 3
STT_FILE = 4


def section_header(mode,name_index,type,flags,offset,size,addr=0,link=SHN_UNDEF,info=0,align=0,ent_size=0):
    return (ELF64_SECTION_HEADER_FMT if mode == MODE_64 else SECTION_HEADER_FMT).pack(
        name_index,type,flags,addr,offset,size,link,info,align,ent_size)


def symbol_table_entry(mode,name,value,size,bind,type,sh_index):
    assert type <= 0xf and bind <= 0xf
    info = (bind << 4) | type

    if mode == MODE_64:
        return ELF64_SYMBOL_TABLE_FMT.pack(name,info,sh_index,value,size)

    return SYMBOL_TABLE_FMT.pack(name,value,size,info,sh_index)

def write_symtab_entry_with_addr(mode,out,name,value,size,bind,type,sh_index):
    if isinstance(out,RelocBuffer):
        out.addrs.append((out.tell() + (
                ELF64_SYMTAB_PREADDR if mode == MODE_64 else SYMTAB_PREADDR
            ),RelocAbsAddress(value)))
    out.write(symbol_table_entry(mode,name,value,size,bind,type,sh_index))


class Section:
    align = 0
    flags = 0
    ent_size = 0
    link = None
    info = 0

class CodeSection(Section):
    name = b'.text'
    align = CODE_ALIGN
    type = SHT_PROGBITS
    flags = SHF_ALLOC | SHF_EXCINSTR

    def __init__(self,code):
        self.code = code

    def write(self,out):
        self.code.write(out)


class SimpleStrTable(Section):
    type = SHT_STRTAB

    def __init__(self,name):
        self.name = name
        self.data = b'\0'

    def write(self,out):
        out.write(self.data)

    def add(self,s):
        r = len(self.data)
        self.data += (s if isinstance(s,(bytes,bytearray)) else s.encode()) + b'\0'
        return r


SectionPos = collections.namedtuple('SectionPos','section offset size name_index')


def align_fo(f,alignment):
    if alignment > 1:
        pad = alignment - (f.tell() % alignment)
        if pad: f.write(b'\0' * pad)


def write_shared_object(mode,out,sections):
    assert mode in (MODE_32,MODE_64)

    s_pos = []

    if not isinstance(out,(RelocBuffer,NonRelocWrapper)):
        out = NonRelocWrapper(out,mode//8)

    # section indices 0 and 0xff00-0xffff have special meaning (and higher
    # indices can't be used because e_shnum of the ELF header is a 16-bit field)
    MAX_SECS = 0xfefe
    if len(sections) > MAX_SECS:
        raise Exception('There cannot be more than {} sections.'.format(MAX_SECS))

    snames = SimpleStrTable(b'.shstrtab')

    # the unit header will be written last
    out.seek((ELF64_HEADER_FMT if mode == MODE_64 else ELF_HEADER_FMT).size)

    sections = list(sections) + [snames]
    for s in sections:
        align_fo(out,s.align)
        offset = out.tell()
        name_index = snames.add(s.name)

        s.write(out)

        s_pos.append(SectionPos(s,offset,out.tell() - offset,name_index))

    align_fo(out,HEADER_ALIGN)
    sec_header_start = out.tell()

    # this is required by the ELF spec
    out.write(b'\0' * (ELF64_SECTION_HEADER_FMT
        if mode == MODE_64 else SECTION_HEADER_FMT).size)

    for s in s_pos:
        out.write(section_header(
            mode,
            s.name_index,
            s.section.type,
            s.section.flags,
            s.offset,
            s.size,
            0,
            SHN_UNDEF if s.section.link is None else (sections.index(s.section.link) + 1),
            s.section.info,
            s.section.align,
            s.section.ent_size))

    out.seek(0)
    if mode == MODE_64:
        out.write(ELF64_HEADER_FMT.pack(
            b'\x7fELF',
            ELFCLASS64,
            ELFDATA2LSB,
            EV_CURRENT,
            ELFOSABI_SYSV,
            0,
            ET_REL,
            EM_X86_64,
            EV_CURRENT,
            0,
            0,
            sec_header_start,
            0,
            ELF64_HEADER_FMT.size,
            0,
            0,
            ELF64_SECTION_HEADER_FMT.size,
            len(s_pos) + 1,
            len(s_pos)))
    else:
        out.write(ELF_HEADER_FMT.pack(
            b'\x7fELF',
            ELFCLASS32,
            ELFDATA2LSB,
            EV_CURRENT,
            ET_REL,
            EM_386,
            EV_CURRENT,
            0,
            0,
            sec_header_start,
            0,
            ELF_HEADER_FMT.size,
            0,
            0,
            SECTION_HEADER_FMT.size,
            len(s_pos) + 1,
            len(s_pos)))

    return s_pos


def symbolify(x):
    """Transform x into a name GDB will accept as a symbol.

    If x is already an acceptable name, it will be returned unchanged.

    """
    return symbolify.pattern.sub('$',x)

# This is the inverse of only what I know GDB will accept. GDB might be more
# permissive.
symbolify.pattern = re.compile(r'\A[0-9]|[^a-zA-Z0-9$_]')
