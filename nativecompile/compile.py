
__all__ = ['compile','compile_asm']



import os
import io
import sys
from functools import partial

from . import pyinternals
from .compile_raw import compile_raw
from . import debug


DUMP_OBJ_FILE = False


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
    global DUMP_OBJ_FILE
    
    cu,entry_points = compile_raw(code,Abi)

    if debug.GDB_JIT_SUPPORT:
        out = debug.generate(Abi,cu,entry_points)
        if DUMP_OBJ_FILE:
            with open('OBJ_DUMP_{}_{}'.format(os.getpid(),int(DUMP_OBJ_FILE)),'wb') as f:
                f.write(out.buff.getbuffer())
            DUMP_OBJ_FILE += 1
    else:
        out = [f.code for f in cu.functions]
    
    return pyinternals.CompiledCode(out,entry_points)


def compile_asm(code):
    """Compile code and return the assembly representation"""
    return ''.join(f.dump() for f in compile_raw(code,Abi,op=Abi.ops.Assembly)[0].functions)

