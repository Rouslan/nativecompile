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


import binascii
from itertools import islice

from .pyinternals import read_address,raw_addresses
from .dinterval import *


DEBUG_PRINT = False


# how many bytes to scan in memory before giving up
SEARCH_LIMIT = 0x10000

# maximum number of non-matching instructions allowed between a "pair" of
# instructions
INSTR_ADJACENCY_FUZZ = 8


MODE_16 = 0
MODE_32 = 1
MODE_64 = 2

class MachineCodeParseError(Exception):
    pass

class InvalidOpCodeError(MachineCodeParseError):
    pass

class SearchLimitReachedError(MachineCodeParseError):
    pass


MOD_ANY = True

# enforcing the memory-only and register-only variants of the ModRM byte are
# not necessary for parsing, but it was easy to implement and having a stricter
# parser makes it easier to verify correctness
MEM_ONLY = True + 1
REG_ONLY = True + 2

def mod_rm_size(mode,data,mod_type=MOD_ANY):
    mod = data >> 6
    rm = data & 0b111

    size = 1

    if mod_type == MEM_ONLY:
        if mod == 3: raise InvalidOpCodeError()
    elif mod_type == REG_ONLY:
        if mod != 3: raise InvalidOpCodeError()

    if mode != MODE_16:
        # address displacement
        if (mod == 0 and rm == 0b101) or mod == 2:
            size += 4
        elif mod == 1:
            size += 1

        # SIB byte
        if mod != 3 and rm == 0b100: size += 1
    else:
        # address displacement
        if (mod == 0 and rm == 0b110) or mod == 2:
            size += 2
        elif mod == 1:
            size += 1

    return size


class MemReader:
    def __init__(self,position=0):
        self.position = position
        self.__scanned = 0

    def advance(self,amount=1):
        self.position += amount
        self.__scanned += amount
        if self.__scanned > SEARCH_LIMIT:
            raise SearchLimitReachedError()

    def read(self,amount):
        r = read_address(self.position,amount)
        self.advance(amount)
        return r

    @property
    def current(self):
        return read_address(self.position,1)[0]


class Address:
    def __init__(self,base=None,index=None,scale=1,offset=0,rip_rel=False):
        self.base = base
        self.index = index
        self.scale = scale
        self.offset = offset
        self.rip_rel = rip_rel

    def normalize(self,position):
        if self.rip_rel:
            self.offset += position
            self.rip_rel = False

    def __repr__(self):
        return "Address({!r},{!r},{!r},{!r},{!r})".format(self.base,self.index,self.scale,self.offset,self.rip_rel)


def rex_bit(b):
    b = 1 << b
    return property(lambda self: bool(self.val & b))

class Rex:
    def __init__(self,val):
        self.val = val

    w = rex_bit(3)
    r = rex_bit(2)
    x = rex_bit(1)
    b = rex_bit(0)

def split_byte_332(x):
    return x & 0b111, (x >> 3) & 0b111, x >> 6

def to_int(x):
    return int.from_bytes(x,'little',signed=True)

def read_mod_rm(mode,data,rex):
    assert mode != MODE_16

    if rex is None: rex = Rex(0)

    rm,reg,mod = split_byte_332(data.current)
    data.advance()

    reg |= rex.r << 3

    if mod != 3:
        if mod == 0 and rm == 0b101:
            arg_b = Address(offset=to_int(data.read(4)),rip_rel=mode==MODE_64)
        else:
            if rm == 0b100:
                arg_b = Address(*split_byte_332(data.current))
                data.advance()

                if arg_b.base == 0b101 and mod == 0:
                    arg_b.base = None
                    arg_b.offset = to_int(data.read(4))
                else:
                    arg_b.base |= rex.b << 3

                arg_b.index |= rex.x << 3

                if arg_b.index == 0b100:
                    arg_b.index = None
            else:
                arg_b = Address(rm)

            if mod == 1:
                arg_b.offset = to_int(data.read(1))
            elif mod == 2:
                arg_b.offset = to_int(data.read(4))
    else:
        arg_b = rm | (rex.b << 3)

    return reg,arg_b



# immediate value sizes:
BYTE_OR_WORD = -1
WORD_OR_DWORD = -2
SEG_AND_NATIVE = -3
NATIVE_SIZE = -4

def immediate_size(type,op_size):
    if type == BYTE_OR_WORD: return 2 if op_size != MODE_16 else 1
    if type == WORD_OR_DWORD: return 4 if op_size != MODE_16 else 2
    if type == SEG_AND_NATIVE: return 2 + (2 << op_size)
    if type == NATIVE_SIZE: return 2 << op_size
    return type


class Format:
    def __init__(self,modrm,imm):
        self.modrm = modrm
        self.imm = imm


MX = Format(True,0)
X  = Format(False,0)
MB = Format(True,1)
B  = Format(False,1)
W  = Format(False,2)
E  = Format(False,3)
rX = Format(REG_ONLY,0)
rB = Format(REG_ONLY,1)
MV = Format(True,WORD_OR_DWORD)
V  = Format(False,WORD_OR_DWORD)
mX = Format(MEM_ONLY,0)
mV = Format(MEM_ONLY,WORD_OR_DWORD)
S  = Format(False,SEG_AND_NATIVE)
N  = Format(False,NATIVE_SIZE)
_  = None

one_byte_map = [
#    0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    MX, MX, MX, MX,  B,  V,  X,  X, MX, MX, MX, MX,  B,  V,  X,  _,   # 0
    MX, MX, MX, MX,  B,  V,  X,  X, MX, MX, MX, MX,  B,  V,  X,  X,   # 1
    MX, MX, MX, MX,  B,  V,  _,  X, MX, MX, MX, MX,  B,  V,  _,  X,   # 2
    MX, MX, MX, MX,  B,  V,  _,  X, MX, MX, MX, MX,  B,  V,  _,  X,   # 3
     X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,   # 4
     X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,   # 5
     X,  X, mV, MX,  _,  _,  _,  _,  V, MV,  B, MB,  X,  X,  X,  X,   # 6
     B,  B,  B,  B,  B,  B,  B,  B,  B,  B,  B,  B,  B,  B,  B,  B,   # 7
    MB, MV, MB, MB, MX, MX, MX, MX, MX, MX, MX, MX, MX, mX, MX, MX,   # 8
     X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  S,  X,  X,  X,  X,  X,   # 9
     B,  V,  B,  V,  X,  X,  X,  X,  B,  V,  X,  X,  X,  X,  X,  X,   # A
     B,  B,  B,  B,  B,  B,  B,  B,  N,  N,  N,  N,  N,  N,  N,  N,   # B
    MB, MB,  W,  X, mX, mX, MB, MV,  E,  X,  W,  X,  X,  B,  X,  X,   # C
    MX, MX, MX, MX,  B,  B,  _,  X, MX, MX, MX, MX, MX, MX, MX, MX,   # D
     B,  B,  B,  B,  B,  B,  B,  B,  V,  V,  S,  B,  X,  X,  X,  X,   # E
     _,  _,  _,  _,  X,  X, MX, MX,  X,  X,  X,  X,  X,  X, MX, MX    # F
]


# 0F is for AMD 3DNow! opcodes

# when used without a prefix, B8 represents the JMPE instruction, which is not
# supported here (because it is used to switch to IA-64 mode which is not
# useful for applications and is not supported by native x86 processors anyway)
two_byte_map = [
#    0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    MX, MX, MX, MX,  _,  _,  X,  _,  X,  X,  _,  _,  _, MX,  X, MB,   # 0
    MX, MX, MX, MX, MX, MX, MX, MX, mX, MX, MX, MX, MX, MX, MX, MX,   # 1
    rX, rX, rX, rX,  _,  _,  _,  _, MX, MX, MX, MX, MX, MX, MX, MX,   # 2
     X,  X,  X,  X,  X,  X,  _,  X,  _,  _,  _,  _,  _,  _,  _,  _,   # 3
    MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # 4
    MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # 5
    MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # 6
    MB, rB, rB, rB, MX, MX, MX,  X, MX, MX,  _,  _, MX, MX, MX, MX,   # 7
     V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,   # 8
    MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # 9
     X,  X,  X, MX, MB, MX,  _,  _,  X,  X,  X, MX, MB, MX, MX, MX,   # A
    MX, MX, mX, MX, mX, mX, MX, MX, MX,  _, MB, MX, MX, MX, MX, MX,   # B
    MX, MX, MB, mX, MB, rB, MB, mX,  X,  X,  X,  X,  X,  X,  X,  X,   # C
    MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # D
    MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # E
    mX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,  _    # F
]

three_byte_map_0x38 = [
#    0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # 0
    MX,  _,  _, MB, MX, MX,  _, MX, mX, mX, mX,  _, MX, MX, MX,  _,   # 1
    MX, MX, MX, MX, MX, MX,  _,  _, MX, MX, mX, MX, mX, mX, mX, mX,   # 2
    MX, MX, MX, MX, MX, MX,  _, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # 3
    MX, MX,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 4
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 5
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 6
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 7
    mX, mX,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 8
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 9
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # A
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # B
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # C
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _, MX, MX, MX, MX, MX,   # D
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # E
    MX, MX,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _    # F
]

three_byte_map_0x3A = [
#    0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
     _,  _,  _,  _, MB, MB, MB,  _, MB, MB, MB, MB, MB, MB, MB, MB,   # 0
     _,  _,  _,  _, MB, MB, MB, MB, MB, MB,  _,  _,  _, MB,  _,  _,   # 1
    MB, MB, MB,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 2
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 3
    MB, MB, MB,  _, MB,  _,  _,  _,  _,  _, MB, MB, MB,  _,  _,  _,   # 4
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 5
    MB, MB, MB, MB,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 6
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 7
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 8
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 9
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # A
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # B
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # C
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _, MB,   # D
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # E
     _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _    # F
]

one_byte_map_64 = one_byte_map[:]
one_byte_map_64[0x40:0x50] = [_] * 0x10 # REX prefixes
for i in (
        0x06,0x07,0x0E,0x16,0x17,0x1E,0x1F,0x27,0x2F,0x37,0x3F,0x60,0x61,0x62,
        0x82,0x9A,0xC4,0xC5,0xCE,0xD4,0xD5,0xD6,0xEA):
    one_byte_map_64[i] = _

two_byte_map_64 = two_byte_map[:]
two_byte_map_64[0x05] = X
two_byte_map_64[0x07] = X


def has_mandatory_prefix(op):
    """Given the second byte of a two-byte opcode, returns true if prefixes
    0x66, 0xF2 and 0xF3 change the instruction instead of having their usual
    meaning.

    This only applies to two-byte opcodes. For one-byte opcodes, this is never
    the case. For three-byte opcodes, this is always the case.

    """
    return (
        0x10 <= op < 0x18 or
        0x28 <= op < 0x30 or
        0x50 <= op < 0x80 or
        0xB8 <= op < 0xC0 or
        0xC2 <= op < 0xC7 or
        0xD0 <= op)


class FindOps:
    def __init__(self,mode,visited):
        """'visited' is used to keep track of addresses that have already been
        scanned. When an address that has already been visited is reached, this
        iterator will end (if 'set_position' is called with an address that has
        been visited, this iterator will produce zero items).

        Note: this class assumes that nothing else modifies 'visited' while a
        instance of this class is iterated over, otherwise any changes made
        will not be seen until set_position is called.

        """
        assert mode == MODE_32 or mode == MODE_64

        self.data = MemReader()
        self.mode = mode
        self.visited = visited
        self.start_pos = self.end_pos = self.data.position

    @property
    def position(self):
        return self.data.position

    # this is not made a property setter because of how much effect it has on
    # instances of this class
    def set_position(self,pos):
        self.visited |= Interval(self.start_pos,self.position)

        self.start_pos = self.data.position = pos
        self.end_pos = self.visited.closest_ge(pos)

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= self.end_pos:
            raise StopIteration()

        # This code does not take into account that certain instructions in
        # 64-bit mode actually have a default operand size of 64 bits. We can
        # get away with this because none of those instructions' lengths differ
        # between 32 and 64 bit operand sizes and neither does the output of
        # this function.
        op_size = MODE_32

        addr_override = False
        man_prefix = b""
        rex = None
        vex = False

        # instruction prefixes
        while True:
            if self.data.current in b"\x66\xF2\xF3":
                man_prefix = bytes([self.data.current])

                if self.data.current == 0x66:
                    op_size = MODE_16

            elif self.data.current == 0x67:
                addr_override = True

            # REX prefix
            elif self.mode == MODE_64 and (self.data.current >> 4) == 0x4:
                rex = Rex(self.data.current & 0xF)
                if rex.w:
                    op_size = MODE_64

            # prefixes we don't care about:
            elif self.data.current not in b"\xF0\x2E\x36\x3E\x26\x64\x65":
                break

            self.data.advance()
        
        op_s = self.data.position

        map = one_byte_map_64 if self.mode == MODE_64 else one_byte_map
        if self.data.current == 0x0F:
            self.data.advance()
            map = two_byte_map

            if self.data.current in b"\x38\x3A" or has_mandatory_prefix(self.data.current):
                op_override = False

                if self.data.current == 0x38:
                    self.data.advance()
                    map = three_byte_map_0x38
                elif self.data.current == 0x3A:
                    self.data.advance()
                    map = three_byte_map_0x3A
            else:
                man_prefix = b""

        elif self.mode == MODE_64 and self.data.current in b"\xC4\xC5":
            vex = True
            if man_prefix or rex is not None: raise InvalidOpCodeError()

            if self.data.current == 0xC4:
                # three-byte VEX prefix
                self.data.advance()
                m = self.data.current & 0b11111
                if m == 1: map = two_byte_map
                elif m == 2: map = three_byte_map_0x38
                elif m == 3: map = three_byte_map_0x3A
                else: raise InvalidOpCodeError()
            else:
                # two-byte VEX prefix
                map = two_byte_map

            self.data.advance(2)

        else:
            man_prefix = b""

        format = map[self.data.current]
        if format is None:
            raise InvalidOpCodeError()

        self.data.advance()

        op = read_address(op_s,self.data.position-op_s)
        if not (op_size == MODE_16 or addr_override or vex):
            ma = None
            mb = None

            if format.modrm:
                ma,mb = read_mod_rm(self.mode,self.data,rex)

            imm_s = self.data.position
            self.data.advance(immediate_size(format.imm,op_size))

            return (man_prefix+op),ma,mb,read_address(imm_s,self.data.position-imm_s)

        # we don't bother interpreting 16-bit operations or VEX-encoded instructions
        if format.modrm:
            self.data.advance(mod_rm_size(MODE_16 if addr_override else self.mode,self.data.current,format.modrm))

        self.data.advance(immediate_size(format.imm,op_size))

        return b"",None,None,None


MOVL = b"\xC7"
CALL = b"\xE8"
JMP_BYTE = b"\xEB"
JMP_FULL = b"\xE9"
RET = b"\xC3"
RET_N = b"\xC2"
GROUP_5 = b"\xFF"
JCC_BYTE_PREFIX = 7
JCC_FULL_PREFIX = 8
JMP_NF_EXT = 0b100,0b101


def find_move_1_addr_pair(position,mode,max_depth=0,visited=None):
    if isinstance(position,str): position = raw_addresses[position]
    if visited is None: visited = DInterval()
    addr_a = None
    after_a = 0
    branches = [position]

    if DEBUG_PRINT: print('scanning {:X}'.format(position))

    find_ops = FindOps(mode,visited)
    while branches:
        newpos = branches.pop()
        # if newpos is in visited, the following for loop will terminate
        # immediately
        find_ops.set_position(newpos)

        if DEBUG_PRINT: print('  branch {:X}:'.format(find_ops.position))

        for op,ma,mb,imm in find_ops:
            if DEBUG_PRINT: print('    op ' + binascii.hexlify(op).decode('ascii'))

            if op == MOVL:
                if DEBUG_PRINT: print('    MOV at {:X}'.format(find_ops.position))

                if (ma & 0b111) == 0 and isinstance(mb,Address) and imm[0] == 1 and not any(imm[1:]):
                    mb.normalize(find_ops.position)

                    # We do not handle the case where the address depends on
                    # any register (except rip). If such a case is encountered,
                    # the parser will need to be made more complex.
                    if not (mb.base is None and mb.index is None):
                        raise MachineCodeParseError()

                    mb = mb.offset

                    if addr_a is None:
                        addr_a = mb
                        after_a = 0
                    else:
                        yield addr_a,mb
                        addr_a = None
            else:
                if after_a == INSTR_ADJACENCY_FUZZ:
                    addr_a = None
                after_a += 1

                if op == CALL:
                    if DEBUG_PRINT: print('    CALL at {:X}'.format(find_ops.position))

                    if max_depth > 0:
                        for x in find_move_1_addr_pair(
                                find_ops.position + to_int(imm),
                                mode,
                                max_depth-1):
                            yield x

                elif op == RET or op == RET_N:
                    if DEBUG_PRINT: print('    RET at {:X}'.format(find_ops.position))

                    addr_a = None
                    break

                elif op == JMP_BYTE or op == JMP_FULL:
                    if DEBUG_PRINT: print('    JMP at {:X}'.format(find_ops.position))

                    newpos = find_ops.position + to_int(imm)
                    if DEBUG_PRINT: print('    jumping to {:X}'.format(find_ops.position))
                    find_ops.set_position(newpos)

                elif op == GROUP_5:
                    # JMPF shouldn't even occur and if JMPN is encountered, we
                    # are somewhere we don't need to be (most likely a linker
                    # stub)
                    if (ma & 0b111) in JMP_NF_EXT: return

                elif ((len(op) == 1 and (op[0] >> 4) == JCC_BYTE_PREFIX) or
                        (len(op) == 2 and op[0] == 0xf and (op[1] >> 4) == JCC_FULL_PREFIX)):
                    if DEBUG_PRINT: print('    Jcc at {:X}'.format(find_ops.position))

                    addr_a = None
                    branches.append(find_ops.position + to_int(imm))

    if DEBUG_PRINT: print('returning from {:X}'.format(position))


def take(itr,n):
    return list(islice(itr,0,n))

def swap(x):
    return x[1],x[0]

def find_async_flag_addresses(mode):
    """Get the addresses of "gil_drop_request", "pendingcalls_to_do" and
    "eval_breaker" """

    addrs = take(find_move_1_addr_pair('PyEval_AcquireThread',mode,1),3)

    if len(addrs) == 1:
        addrs_g = addrs[0]
    elif len(addrs) == 2:
        # PyEval_AcquireThread also calls _PyEval_SignalAsyncExc. If we
        # encountered two pairs of addresses, _PyEval_SignalAsyncExc must have
        # been inlined and we need to parse its code to determine which pair
        # came from where.

        addrs_s = take(find_move_1_addr_pair('_PyEval_SignalAsyncExc',mode),2)
        if len(addrs_s) != 1: raise MachineCodeParseError()

        addrs_s = frozenset(addrs_s[0])
        if addrs_s == frozenset(addrs[0]):
            addrs_g = addrs[1]
        elif addrs_s == frozenset(addrs[1]):
            addrs_g = addrs[0]
        else:
            raise MachineCodeParseError()
    else:
        raise MachineCodeParseError()

    if addrs_g[0] == addrs_g[1]: raise MachineCodeParseError()


    addrs = take(find_move_1_addr_pair('Py_AddPendingCall',mode),2)
    if len(addrs) != 1: raise MachineCodeParseError()
    addrs_p = addrs[0]

    if addrs_p[0] == addrs_p[1]: raise MachineCodeParseError()

    if addrs_g[1] != addrs_p[1]:
        if addrs_g[1] == addrs_p[0]:
            addrs_p = swap(addrs_p)
        elif addrs_g[0] == addrs_p[0]:
            addrs_g = swap(addrs_g)
            addrs_p = swap(addrs_p)
        elif addrs_g[0] == addrs_p[1]:
            addrs_g = swap(addrs_g)
        else:
            raise MachineCodeParseError()

    if addrs_g[0] == addrs_p[0]: raise MachineCodeParseError()

    return addrs_g[0],addrs_p[0],addrs_g[1]

