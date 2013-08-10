
import collections
from itertools import groupby

from . import pyinternals
from . import elf
from . import dwarf


GDB_JIT_SUPPORT = getattr(pyinternals,'GDB_JIT_SUPPORT',False)

InnerAnnotation = collections.namedtuple('InnerAnnotation','stack r_ret_used')
DescribedAnnot = collections.namedtuple('DescribedAnnot','stack r_ret_used descr')

# we want a mutable "size"
class Annotation:
    __slots__ = 'stack','r_ret_used','descr','size'

    def __init__(self,i_annot,size):
        self.stack,self.r_ret_used = i_annot[0:2]
        self.descr = getattr(i_annot,'descr',None)
        self.size = size

    @staticmethod
    def mergable(a,b):
        return a.descr is None and b.descr is None and a.stack == b.stack and a.r_ret_used == b.r_ret_used


def annotate(stack,r_ret_used,descr=None):
    return [DescribedAnnot(stack,r_ret_used,descr)
            if descr is not None else
        InnerAnnotation(stack,r_ret_used)] if GDB_JIT_SUPPORT else []


def append_annot(seq,item):
    if item.size:
        if seq and Annotation.mergable(seq[-1],item):
            seq[-1].size += item.size
        else:
            seq.append(item)

def max_not_none(a,b):
    if a is None: return b
    if b is None: return a
    return max(a,b)

def annot_max(a,b):
    if a is None: return b
    if b is None: return a
    return InnerAnnotation._make(map(max_not_none,a[0:2],b[0:2]))


RETURN_ADDRESS = object()
EPILOG_START = object()
PYTHON_STACK_START = object()

class PrologEnd:
    def __init__(self,stack_reserved):
        self.reserved = stack_reserved

class SaveReg:
    def __init__(self,reg):
        self.reg = reg

class RestoreReg:
    def __init__(self,reg):
        self.reg = reg

class PushVariable:
    def __init__(self,name):
        self.name = name


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
                    f.address,
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
        self.start = func.address
        self.end = func.address + len(func) - func.padding
        self.annot = func.annotation
        self.abi = abi
        self.mode = mode

    def values(self,op):
        regend = self.start
        for a in self.annot:
            if isinstance(a.descr,PushVariable) and a.descr.name == 'f':
                yield self.start,regend,op.reg(dwarf_reg(self.mode,self.abi.r_arg[0]))
                yield regend,self.end - regend,op.fbreg(a.stack * -self.abi.ptr_size)
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

        # for the call frame, we need to record how much stack space is
        # allocated, not just how many items we have on it (which is what
        # a.stack specifies)
        for a in itr:
            assert a.stack is not None

            if in_body:
                if a.descr is EPILOG_START:
                    in_body = False
                    add_code(op.def_cfa_offset(a.stack * self.ptr_size))
                    old_stack = a.stack
                    continue
            elif isinstance(a.descr,PrologEnd):
                in_body = True
                add_code(op.def_cfa_offset(a.descr.reserved * self.ptr_size))
                old_stack = a.descr.reserved
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

        func.name = None
        if func.entry_point:
            dop = dwarf.OP(dmode)

            if func.entry_point.co_name:
                df.name = st[elf.symbolify(func.entry_point.co_name)]
            df.type = ref(t_vptr)
            df.frame_base = dwarf.FORM_exprloc(dop.call_frame_cfa())

            df.children.append(dwarf.DIE('formal_parameter',
                name=st['f'],
                type=ref(t_vptr),
                location=PyFrameLocListEntry(func,abi,dmode)
                    if abi.r_arg else dwarf_arg_loc(abi,dmode,0)))

            df.children.append(dwarf.DIE('formal_parameter',
                name=st['throwflag'],
                type=ref(t_int),
                location=dwarf_arg_loc(abi,dmode,1)))

            s_addr = func.address
            for i,a in enumerate(func.annotation):
                if a.descr is PYTHON_STACK_START:

                    v_size = dwarf.DIE('variable',
                        name=st['stack_len'],
                        type=ref(t_ulong),
                        location=StackBodyLocListEntry(s_addr,func.annotation[i:]))
                    df.children.append(v_size)

                    a_type = dwarf.DIE('array_type',
                        type=ref(t_vptr),
                        byte_stride=dwarf.FORM_sdata(-abi.ptr_size))
                    df.children.append(a_type)

                    a_type.children.append(dwarf.DIE('subrange_type',
                        type=ref(t_ulong),
                        lower_bound=dwarf.FORM_udata(0),
                        count=ref(v_size)))

                    df.children.append(dwarf.DIE('variable',
                        name=st['stack'],
                        type=ref(a_type),
                        start_scope=dwarf.FORM_udata(s_addr),
                        location=dwarf.FORM_exprloc(dop.fbreg(a.stack * -abi.ptr_size))))

                    break
                s_addr += a.size
            else:
                assert False, "Named function without PROLOG_END marked"

            cfs.append(CallFrameEntry(func.address,func.annotation,abi.ptr_size))

        df.low_pc = dwarf.FORM_addr(func.address)
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

    out = elf.RelocBuffer(abi.ptr_size)
    c_offset = elf.write_shared_object(emode,out,sections)[0].offset

    # adjust the entry points by the code segment's offset
    for e in entry_points:
        pyinternals.cep_set_offset(e,pyinternals.cep_get_offset(e) + c_offset)

    # all addresses refer to the code segment
    for o,a in out.addrs:
        a.val += c_offset

    return out


