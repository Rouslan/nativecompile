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


from itertools import groupby

from . import pyinternals
from . import elf
from . import dwarf
from .compilation_unit import *
from .reloc_buffer import RelocBuffer


GDB_JIT_SUPPORT = getattr(pyinternals,'GDB_JIT_SUPPORT',False)


# we want a mutable "size"
class Annotation:
    __slots__ = 'descr','size'

    def __init__(self,descr,size):
        self.descr = descr
        self.size = size

    def __repr__(self):
        return 'Annotation({!r},{})'.format(self.descr,self.size)


RETURN_ADDRESS = 'RETURN_ADDRESS'

class PrologEnd:
    def __init__(self,stack):
        self.stack = stack

    def __repr__(self):
        return 'PrologEnd({})'.format(self.stack)

EPILOG_START = 'EPILOG_START'
PYTHON_STACK_START = 'PYTHON_STACK_START'

class SaveReg:
    def __init__(self,reg):
        self.reg = reg

    def __repr__(self):
        return 'SaveReg({!r})'.format(self.reg)

class RestoreReg:
    def __init__(self,reg):
        self.reg = reg

    def __repr__(self):
        return 'RestoreReg({!r})'.format(self.reg)

class VariableLoc:
    def __init__(self,name,loc):
        self.name = name
        self.loc = loc

    def __repr__(self):
        return 'VariableLoc({!r},{!r})'.format(self.name,self.loc)


class SymbolSection(elf.Section):
    name = b'.symtab'
    type = elf.SHT_SYMTAB

    def __init__(self,mode,strtab,funcs):
        self.mode = mode
        self.link = self.strtab = strtab
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


class DwarfLocation:
    def dwarf_loc_expr(self,op):
        raise NotImplementedError()


class VarLocListEntry(dwarf.FORM_loclist_offset):
    def __init__(self,var,func,start_index=0,start_offset=0):
        self.var = var
        self.func = func
        self.start_index = start_index
        self.start_offset = start_offset

    def values(self,op):
        end = self.start_offset + self.func.offset
        last_loc = None
        last_offset = 0

        for i in range(self.start_index,len(self.func.annotation)):
            annot = self.func.annotation[i]
            if isinstance(annot.descr,VariableLoc) and annot.descr.name == self.var:
                if annot.descr.loc != last_loc:
                    if last_loc is not None and end > last_offset:
                        yield last_offset,end,last_loc.dwarf_loc_expr(op)
                    last_loc = annot.descr.loc
                    last_offset = end

            end += annot.size

def find_vars(func):
    pos = func.offset
    found = set()
    for i,annot in enumerate(func.annotation):
        if isinstance(annot.descr,VariableLoc) and annot.descr.name not in found:
            found.add(annot.descr.name)
            yield annot.descr.name,VarLocListEntry(annot.descr.name,func,i,pos)
        pos += annot.size

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

        stack = 0
        prolog_stack = None
        instr = b''
        span = 0
        tot_size = 0

        def add_code(c):
            nonlocal instr,span

            if span:
                instr += op.advance_loc_smallest(span)
                span = 0
            instr += c

        def add_stack_code():
            add_code(op.def_cfa_offset(stack*self.ptr_size))

        for a in self.annot:
            if a.descr is RETURN_ADDRESS:
                assert prolog_stack is None
                stack += 1
                add_stack_code()
            elif a.descr is EPILOG_START:
                assert prolog_stack is not None
                if prolog_stack != stack:
                    stack = prolog_stack
                    add_stack_code()
            elif isinstance(a.descr,PrologEnd):
                assert prolog_stack is None
                prolog_stack = stack
                if stack != a.descr.stack:
                    stack = a.descr.stack
                    add_stack_code()
            elif isinstance(a.descr,SaveReg):
                assert prolog_stack is None
                stack += 1
                add_stack_code()
                add_code(op.offset(getattr(reg,a.descr.reg.name),stack))
            elif isinstance(a.descr,RestoreReg):
                assert prolog_stack is not None
                stack -= 1
                add_stack_code()
                add_code(op.restore(getattr(reg,a.descr.reg.name)))

            span += a.size
            tot_size += a.size

        assert stack == 1

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

    all_types = {}
    for func in cu.functions:
        if func.returns:
            func.returns.die(dcu,abi,st,all_types)
        for a in func.params:
            a.type.die(dcu,abi,st,all_types)

    for func in cu.functions:
        df = dwarf.DIE('subprogram')

        if func.name:
            df.name = st[elf.symbolify(func.name)]
        if func.returns:
            df.type = ref(all_types[func.returns])

        if func.annotation:
            df.frame_base = dwarf.FORM_exprloc(dwarf.OP(dmode).call_frame_cfa())
            cfs.append(CallFrameEntry(func.offset,func.annotation,abi.ptr_size))

        if func.callconv == CallConvType.utility:
            df.calling_convention = dwarf.CC.nocall

        params = [None] * len(func.params)
        #vars_ = []
        for var,form in find_vars(func):
            for i,a in enumerate(func.params):
                if a.name == var:
                    params[i] = dwarf.DIE('formal_parameter',
                        name=st[var],
                        type=ref(all_types[a.type]),
                        location=form)
                    break
            else:
                pass
                #vars_.append(dwarf.DIE('variable',
                #    name=st[var],
                #    type=ref(t_vptr),
                #    location=form))

        df.children.extend(p for p in params if p)
        #df.children.extend(vars_)

        df.low_pc = dwarf.FORM_addr(func.offset)
        df.high_pc = dwarf.FORM_udata(len(func) - func.padding)

        dcu.children.append(df)

    sym_strtab = elf.SimpleStrTable('.strtab')
    callframes = lambda op,r: (cf(op,r) for cf in cfs)

    sections = [
        elf.CodeSection(cu),
        SymbolSection(emode,sym_strtab,cu.functions),
        sym_strtab] + dwarf.elf_sections(dmode,dcu,st,callframes)

    out = RelocBuffer(abi.ptr_size)
    c_offset = elf.write_shared_object(emode,out,sections)[0].offset

    # adjust the entry points by the code segment's offset
    for e in entry_points:
        e.offset += c_offset

    # all addresses refer to the code segment
    for o,a in out.addrs:
        a.val += c_offset

    return out


