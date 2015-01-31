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


import collections
from itertools import groupby

from . import pyinternals
from . import elf
from . import dwarf
from .reloc_buffer import RelocBuffer


GDB_JIT_SUPPORT = getattr(pyinternals,'GDB_JIT_SUPPORT',False)

InnerAnnotation = collections.namedtuple('InnerAnnotation','stack descr')

# we want a mutable "size"
class Annotation:
    __slots__ = 'stack','descr','size'

    def __init__(self,i_annot,size):
        self.stack,self.descr = i_annot
        self.size = size

    @staticmethod
    def mergable(a,b):
        return a.descr is None and b.descr is None and a.stack == b.stack


def annotate(stack,descr=None):
    return [InnerAnnotation(stack,descr)] if GDB_JIT_SUPPORT else []


def append_annot(seq,item):
    if item.size:
        if seq and Annotation.mergable(seq[-1],item):
            seq[-1].size += item.size
        else:
            seq.append(item)


RETURN_ADDRESS = 'RETURN_ADDRESS'
PROLOG_END = 'PROLOG_END'
EPILOG_START = 'EPILOG_START'
PYTHON_STACK_START = 'PYTHON_STACK_START'

class SaveReg:
    def __init__(self,reg):
        self.reg = reg

    def __repr__(self):
        return 'SaveReg({})'.format(self.reg)

class RestoreReg:
    def __init__(self,reg):
        self.reg = reg

    def __repr__(self):
        return 'RestoreReg({})'.format(self.reg)

class PushVariable:
    def __init__(self,name):
        self.name = name

    def __repr__(self):
        return 'PushVariable({})'.format(self.name)


class SymbolSection(elf.Section):
    name = b'.symtab'
    type = elf.SHT_SYMTAB
    link = 3 # index of the string table (TODO: don't hard-code)

    def __init__(self,mode,strtab,funcs):
        self.mode = mode
        self.strtab = strtab
        self.funcs = funcs
        self.ent_size = (elf.ELF64_SYMBOL_TABLE_FMT
                if mode == 64 else
            elf.SYMBOL_TABLE_FMT).size

        # one greater than the index of the last local symbol (STB_LOCAL)
        self.info = 2
        for f in self.funcs:
            if f.name: self.info += 1

    def write(self,out):
        out.write(b'\0' * self.ent_size)
        elf.write_symtab_entry_with_addr(self.mode,out,self.strtab.add(b'.text'),0,0,elf.STB_LOCAL,elf.STT_SECTION,1)

        for f in self.funcs:
            if f.name:
                elf.write_symtab_entry_with_addr(
                    self.mode,
                    out,
                    self.strtab.add(f.name),
                    f.offset,
                    len(f) - f.padding,
                    elf.STB_LOCAL,
                    elf.STT_FUNC,
                    1)


def dwarf_reg(mode,reg):
    """Convert a Register object to the corresponding dwarf value"""
    return getattr(dwarf.reg(mode),reg.name)

def dwarf_arg_loc(abi,mode,index):
    op = dwarf.OP(mode)
    if index >= len(abi.r_arg):
        r = op.fbreg(index * abi.ptr_size)
    else:
        r = op.reg(dwarf_reg(mode,abi.r_arg[index]))

    return dwarf.FORM_exprloc(r)


class StackBodyLocListEntry(dwarf.FORM_loclist_offset):
    def __init__(self,addr,annot):
        self.addr = addr
        self.annot = annot

    def values(self,op):
        assert self.annot

        bottom = self.annot[0].stack
        start = self.addr
        expr_end = op.stack_value()
        for stack,annots in groupby(self.annot,(lambda a: a.stack)):
            if stack < bottom: break
            end = start + sum(a.size for a in annots)
            assert end > start
            yield start,end,op.const_int(stack - bottom) + expr_end
            start = end

class PyFrameLocListEntry(dwarf.FORM_loclist_offset):
    def __init__(self,func,abi,mode):
        self.start = func.offset
        self.end = func.offset + len(func) - func.padding
        self.annot = func.annotation
        self.abi = abi
        self.mode = mode

    def values(self,op):
        regend = self.start
        for a in self.annot:
            if isinstance(a.descr,PushVariable) and a.descr.name == '__f':
                yield self.start,regend,op.reg(dwarf_reg(self.mode,self.abi.r_arg[0]))
                yield regend,self.end - regend + self.start,op.fbreg(a.stack * -self.abi.ptr_size)
                break
            regend += a.size
        else:
            assert False

class StackTopRangeEntry(dwarf.FORM_rangelist_offset):
    def __init__(self,addr,annot):
        self.addr = addr
        self.annot = annot

    def values(self):
        start = self.addr
        for used,annots in groupby(self.annot,(lambda a: a.r_ret_used)):
            end = start + sum(a.size for a in annots)
            assert end > start
            if used: yield start,end
            start = end

class CallFrameEntry:
    def __init__(self,addr,annot,ptr_size):
        self.addr = addr
        self.annot = annot
        self.ptr_size = ptr_size

    def __call__(self,op,reg):
        assert self.annot

        itr = iter(self.annot)
        a = next(itr)
        old_stack = a.stack
        in_body = False
        instr = b''
        span = a.size
        tot_size = 0

        def add_code(c):
            nonlocal instr,span,tot_size

            if span:
                instr += op.advance_loc_smallest(span)
                tot_size += span
                span = 0
            instr += c

        for a in itr:
            assert in_body or a.stack is not None

            if in_body:
                if a.descr is EPILOG_START:
                    in_body = False
                    add_code(op.def_cfa_offset(a.stack * self.ptr_size))
                    old_stack = a.stack
                    continue
            elif a.descr is PROLOG_END:
                in_body = True
                add_code(op.def_cfa_offset(a.stack * self.ptr_size))
                old_stack = a.stack
                continue
            elif a.stack != old_stack:
                add_code(op.def_cfa_offset(a.stack * self.ptr_size))
                old_stack = a.stack

            if isinstance(a.descr,SaveReg):
                add_code(op.offset(getattr(reg,a.descr.reg.name),a.stack))
            elif isinstance(a.descr,RestoreReg):
                add_code(op.restore(getattr(reg,a.descr.reg.name)))

            span += a.size

        tot_size += span

        return self.addr,tot_size,instr


def generate(abi,cu,entry_points):
    emode = abi.ptr_size * 8
    dmode = dwarf.Mode(dwarf.MODE_32,abi.ptr_size)

    st = dwarf.StringTable()
    ref = dwarf.FORM_ref_udata
    cfs = []

    dcu = dwarf.DIE('compile_unit',
        use_UTF8=dwarf.FORM_flag_present(),
        low_pc=dwarf.FORM_addr(0),
        high_pc=dwarf.FORM_udata(len(cu)))

    t_int = dwarf.DIE('base_type',
        name=st['int'],
        encoding=dwarf.ATE.signed,
        byte_size=dwarf.FORM_data1(abi.int_size))
    dcu.children.append(t_int)

    if abi.long_size > abi.int_size:
        t_ulong = dwarf.DIE('base_type',
            name=st['unsigned long'],
            encoding=dwarf.ATE.unsigned,
            byte_size=dwarf.FORM_data1(abi.long_size))
    else:
        assert abi.long_size == abi.int_size
        t_ulong = dwarf.DIE('base_type',
            name=st['unsigned int'],
            encoding=dwarf.ATE.unsigned,
            byte_size=dwarf.FORM_data1(abi.int_size))
    dcu.children.append(t_ulong)

    t_void = dwarf.DIE('unspecified_type',name=st['void'])
    dcu.children.append(t_void)

    t_vptr = dwarf.DIE('pointer_type',
        type=ref(t_void),
        byte_size=dwarf.FORM_data1(abi.ptr_size))
    dcu.children.append(t_vptr)


    for func in cu.functions:
        df = dwarf.DIE('subprogram')

        if func.pyfunc:
            dop = dwarf.OP(dmode)

            if func.name:
                df.name = st[elf.symbolify(func.name)]
            df.type = ref(t_vptr)
            df.frame_base = dwarf.FORM_exprloc(dop.call_frame_cfa())

            df.children.append(dwarf.DIE('formal_parameter',
                name=st['f'],
                type=ref(t_vptr),
                location=PyFrameLocListEntry(func,abi,dmode)
                    if abi.r_arg else dwarf_arg_loc(abi,dmode,0)))

            cfs.append(CallFrameEntry(func.offset,func.annotation,abi.ptr_size))

        df.low_pc = dwarf.FORM_addr(func.offset)
        df.high_pc = dwarf.FORM_udata(len(func) - func.padding)

        dcu.children.append(df)

    # if the string table's position in the list is changed, be sure to update
    # SymbolSection.link up above
    sym_strtab = elf.SimpleStrTable('.strtab')
    callframes = lambda op,r: (cf(op,r) for cf in cfs)
    reg = dwarf.reg(dmode)
    pres_regs = [getattr(reg,r.name) for r in abi.r_pres] + [reg._bp]
    sections = [
        elf.CodeSection(cu),
        SymbolSection(emode,sym_strtab,cu.functions),
        sym_strtab] + dwarf.elf_sections(dmode,dcu,st,callframes,pres_regs)

    out = RelocBuffer(abi.ptr_size)
    c_offset = elf.write_shared_object(emode,out,sections)[0].offset

    # adjust the entry points by the code segment's offset
    for e in entry_points:
        e.func.offset += c_offset

    # all addresses refer to the code segment
    for o,a in out.addrs:
        a.val += c_offset

    return out


