
from . import x86_ops

class CdeclAbi:
    shadow = 0
    ops = x86_ops
    r_ret = x86_ops.eax
    r_scratch = [x86_ops.ecx,x86_ops.edx]
    r_pres = [x86_ops.ebx,x86_ops.esi,x86_ops.edi]
    r_sp = x86_ops.esp
    r_bp = x86_ops.ebp
    r_arg = []
    ptr_size = 4
    char_size = 1
    short_size = 2
    int_size = 4
    long_size = 4
