
from . import x86_64_ops as _ops


class SystemVAbi:
    shadow = 0
    ops = _ops
    r_ret = _ops.rax
    r_scratch = [_ops.r10,_ops.r11]
    r_pres = [_ops.rbx,_ops.r12,_ops.r13,_ops.r14,_ops.r15]
    r_sp = _ops.rsp
    r_bp = _ops.rbp
    r_arg = [_ops.rdi,_ops.rsi,_ops.rdx,_ops.rcx,_ops.r8,_ops.r9]
    ptr_size = 8


class MicrosoftX64Abi:
    shadow = 32
    ops = _ops
    r_ret = _ops.rax
    r_scratch = [_ops.r10,_ops.r11]
    r_pres = [_ops.rbx,_ops.rsi,_ops.rdi,_ops.r12,_ops.r13,_ops.r14,_ops.r15]
    r_sp = _ops.rsp
    r_bp = _ops.rbp
    r_arg = [_ops.rcx,_ops.rdx,_ops.r8,_ops.r9]
    ptr_size = 8
