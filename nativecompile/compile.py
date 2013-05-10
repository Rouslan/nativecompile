
__all__ = ['compile','compile_asm']



import os
import io
import sys

from . import pyinternals
from .compile_raw import compile_raw


GDB_JIT_SUPPORT = getattr(pyinternals,'GDB_JIT_SUPPORT',False)
if GDB_JIT_SUPPORT:
    from . import elf
    from . import dwarf

    class SymbolSection(elf.Section):
        name = b'.symtab'
        type = elf.SHT_SYMTAB
        link = 3 # index of the string table

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
                        f.address,
                        len(f) - f.padding,
                        elf.STB_LOCAL,
                        elf.STT_FUNC,
                        1)


if pyinternals.ARCHITECTURE == "X86":
    from .x86_abi import CdeclAbi as Abi
elif pyinternals.ARCHITECTURE == "X86_64":
    from . import x86_64_abi
    if sys.platform in ('win32','cygwin'):
        Abi = x86_64_abi.MicrosoftX64Abi
    else:
        Abi = x86_64_abi.SystemVAbi
else:
    raise Exception("native compilation is not supported on this CPU")


def compile(code):
    out = io.BytesIO()
    
    cu,entry_points = compile_raw(code,Abi)


    if GDB_JIT_SUPPORT:
        mode = Abi.ptr_size * 8

        st = dwarf.StringTable()

        dcu = dwarf.DIE(dwarf.TAG.compile_unit)
        dcu.use_UTF8 = dwarf.FORM_flag_present()

        for func in cu.functions:
            func.name = None
            if func.entry_point:
                func.name = elf.symbolify(func.entry_point.co_name)

            df = dwarf.DIE(dwarf.TAG.subprogram)
            df.low_pc = dwarf.FORM_addr(func.address)
            df.high_pc = dwarf.FORM_udata(len(func) - func.padding)
            if func.name: df.name = st[func.name]

            dcu.children.append(df)

        sym_strtab = elf.SimpleStrTable('.strtab')
        sections = [
            elf.CodeSection(cu),
            SymbolSection(mode,sym_strtab,cu.functions),
            sym_strtab] + dwarf.elf_sections(mode,dcu,st)
        #sections = [elf.CodeSection(cu)] + dwarf.elf_sections(mode,dcu,st)

        out = elf.RelocBuffer(Abi.ptr_size)
        c_offset = elf.write_shared_object(mode,out,sections)[0].offset

        # adjust the entry points by the code segment's offset
        for e in entry_points:
            pyinternals.cep_set_offset(e,pyinternals.cep_get_offset(e) + c_offset)
    else:
        out = [f.code for f in cu.functions]

    
    return pyinternals.CompiledCode(out,entry_points)


def compile_asm(code):
    """Compile code and return the assembly representation"""
    return compile_raw(code,Abi,binary=False)[0].dump()

