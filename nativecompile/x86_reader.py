
import binascii

from .pyinternals import read_address


# how many bytes to scan in memory before giving up
SEARCH_LIMIT = 0x10000

MODE_16 = 0
MODE_32 = 1

class InvalidOpCodeError(Exception):
    pass

class SearchLimitReachedError(Exception):
    pass


MOD_ANY = True
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

    if mode == MODE_32:
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
    def __init__(self,position):
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
    def __init__(self,base=None,index=None,scale=1,offset=0):
        self.base = base
        self.index = index
        self.scale = scale
        self.offset = offset

    def __repr__(self):
        return "Address({!r},{!r},{!r},{!r})".format(self.base,self.index,self.scale,self.offset)

class Register:
    def __init__(self,reg):
        self.reg = reg

def split_byte_332(x):
    return x & 0b111, (x >> 3) & 0b111, x >> 6

def to_int(x):
    return int.from_bytes(x,'little',signed=True)

def read_mod_rm(mode,data):
    assert mode == MODE_32 # we're not interested in 16-bit operations

    rm,reg,mod = split_byte_332(data.current)

    data.advance()

    if mod != 3:
        if mod == 0 and rm == 0b101:
            arg_b = Address(offset=to_int(data.read(4)))
        else:
            if rm == 0b100:
                arg_b = Address(*split_byte_332(data.current))
                data.advance()

                if arg_b.index == 0b100:
                    arg_b.index = None
                if arg_b.base == 0b101 and mod == 0:
                    arg_b.base = None
                    arg_b.offset = to_int(data.read(4))
            else:
                arg_b = Address(rm)

            if mod == 1:
                arg_b.offset = to_int(data.read(1))
            elif mod == 2:
                arg_b.offset = to_int(data.read(4))
    else:
        arg_b = Register(rm)

    return reg,arg_b



# immediate value sizes:
BYTE_OR_WORD = -1
WORD_OR_DWORD = -2
SEG_WORD_OR_DWORD = -3

def immediate_size(type,mode):
    if type == SEG_WORD_OR_DWORD:
        return 6 if mode else 4
    return -type << mode if type < 0 else type


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
S  = Format(False,SEG_WORD_OR_DWORD)
_  = None

one_byte_map = (
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
     B,  B,  B,  B,  B,  B,  B,  B,  V,  V,  V,  V,  V,  V,  V,  V,   # B
    MB, MB,  W,  X, mX, mX, MB, MV,  E,  X,  W,  X,  X,  B,  X,  X,   # C
    MX, MX, MX, MX,  B,  B,  _,  X, MX, MX, MX, MX, MX, MX, MX, MX,   # D
     B,  B,  B,  B,  B,  B,  B,  B,  V,  V,  S,  B,  X,  X,  X,  X,   # E
     _,  _,  _,  _,  X,  X, MX, MX,  X,  X,  X,  X,  X,  X, MX, MX    # F
)

# 0F is for AMD 3DNow! opcodes
two_byte_map = (
#    0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    MX, MX, MX, MX,  _,  _,  X,  _,  X,  X,  _,  _,  _,  _,  _, MB,   # 0
    MX, MX, MX, MX, MX, MX, MX, MX, mX,  _,  _,  _,  _,  _,  _,  _,   # 1
    rX, rX, rX, rX,  _,  _,  _,  _, MX, MX, MX, MX, MX, MX, MX, MX,   # 2
     X,  X,  X,  X,  X,  X,  _,  _,  _,  _,  _,  _,  _,  _,  _,  _,   # 3
    MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # 4
    MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,  _,  _, MX, MX, MX, MX,   # 5
    MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,  _,  _, MX, MX,   # 6
    MB, rB, rB, rB, MX, MX, MX,  X,  _,  _,  _,  _,  _,  _, MX, MX,   # 7
     V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,  V,   # 8
    MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # 9
     X,  X,  X, MX, MB, MX,  _,  _,  X,  X,  X, MX, MB, MX, MX, MX,   # A
    MX, MX, mX, MX, mX, mX, MX, MX,  _,  _, MB, MX, MX, MX, MX, MX,   # B
    MX, MX, MB,  _, MB, rB, MB, mX,  X,  X,  X,  X,  X,  X,  X,  X,   # C
     _, MX, MX, MX,  _, MX,  _, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # D
    MX, MX, MX, MX, MX, MX,  _, MX, MX, MX, MX, MX, MX, MX, MX, MX,   # E
     _, MX, MX, MX,  _, MX, MX, MX, MX, MX, MX,  _, MX, MX, MX,  _    # F
)

class FindOps:
    def __init__(self,position):
        self.data = MemReader(position)

    def set_position(self,val):
        self.data.position = val

    position = property((lambda self: self.data.position),set_position)

    def __iter__(self):
        return self

    def __next__(self):
        op_override = False
        addr_override = False

        # instruction prefixes
        while True:
            if self.data.current == 0x66: # operand-size override
                op_override = True
            elif self.data.current == 0x67: # address-size override
                addr_override = True
            # prefixes we don't care about:
            elif self.data.current not in b"\xF0\xF2\xF3\x2E\x36\x3E\x26\x64\x65":
                break

            self.data.advance()
        
        op_s = self.data.position

        map = one_byte_map
        if self.data.current == 0x0F:
            self.data.advance()
            map = two_byte_map

        format = map[self.data.current]
        if format is None:
            raise InvalidOpCodeError()

        self.data.advance()

        op = read_address(op_s,self.data.position-op_s)
        if not (op_override or addr_override):
            ma = None
            mb = None

            if format.modrm:
                ma,mb = read_mod_rm(MODE_32,self.data)

            imm_s = self.data.position
            self.data.advance(immediate_size(format.imm,MODE_32))

            return op,ma,mb,read_address(imm_s,self.data.position-imm_s)

        # we don't bother parsing 16-bit operations
        if format.modrm:
            self.data.advance(mod_rm_size(MODE_16 if addr_override else MODE_32,self.data.current,format.modrm))

        self.data.advance(immediate_size(format.imm,MODE_16 if op_override else MODE_32))

        return b"",None,None,None


MOVL = b"\xC7"
CALL = b"\xE8"
JMP_BYTE = b"\xEB"
JMP_FULL = b"\xE9"
RET = b"\xC3"
RET_N = b"\xC2"
JCC_BYTE_PREFIX = 7
JCC_FULL_PREFIX = 8

JMP_LIMIT = 0x1000

def find_move_1_addr_pair(position,max_depth=0):
    addr_a = None
    forward_pos = 0

    find_ops = FindOps(position)
    for op,ma,mb,imm in find_ops:
        if op == MOVL:
            if ma == 0 and isinstance(mb,Address) and imm == b"\x01\x00\x00\x00":
                if addr_a is None:
                    addr_a = mb
                else:
                    yield addr_a,mb
                    addr_a = None
        else:
            addr_a = None

            if op == CALL:
                if max_depth > 0:
                    for x in find_move_1_addr_pair(
                            find_ops.position + to_int(imm),
                            max_depth-1):
                        yield x

            elif op == RET or op == RET_N:
                if find_ops.position > forward_pos: break

            elif op == JMP_BYTE or op == JMP_FULL:
                jmp = to_int(imm)
                if JMP_LIMIT > jmp > 0:
                    find_ops.position += jmp

            elif ((len(op) == 1 and (op[0] >> 4) == JCC_BYTE_PREFIX) or
                    (len(op) == 2 and op[0] == 0xf and (op[1] >> 4) == JCC_FULL_PREFIX)):
                # keep track of the farthest position a conditional jump will
                # take us
                jmp = to_int(imm)
                if JMP_LIMIT > jmp > 0:
                    target = find_ops.position + jmp
                    if target > forward_pos: forward_pos = target

