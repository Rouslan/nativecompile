
import os
import tempfile
import atexit

from . import pyinternals

if pyinternals.ARCHITECTURE == "X86":
    from .x86_compile import compile_raw
elif pyinternals.ARCHITEXTURE == "X86_64":
    raise Exception("native compilation is not supported for the 64-bit version of Python")
else:
    raise Exception("native compilation is not supported on this CPU")




def compile(code):
    f = tempfile.NamedTemporaryFile(mode='wb',delete=False)
    
    def delete_f():
        f.close()
        os.remove(f.name)
    
    atexit.register(delete_f)
    
    parts,entry_points = compile_raw(code)
    for p in parts:
        f.write(p)

    f.close()
    
    return pyinternals.CompiledCode(f.name,code,entry_points)



