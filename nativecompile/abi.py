#  Copyright 2016 Rouslan Korneychuk
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

from typing import Dict,List,Optional

class Tuning:
    prefer_addsub_over_incdec = True
    build_seq_loop_threshhold = 5
    unpack_seq_loop_threshhold = 5
    build_set_loop_threshhold = 5
    mem_copy_loop_threshhold = 9


class Register:
    pass

class AbiMeta(type):
    def __new__(mcs,name,bases,namespace):
        r = type.__new__(mcs,name,bases,namespace)

        # general-purpose registers ordered by preferred usage
        r.all_regs = r.r_pres + r.r_scratch
        if r.r_ret is not None: r.all_regs.append(r.r_ret)
        r.all_regs += r.r_arg
        r.gen_regs = len(r.all_regs)
        if r.r_sp is not None: r.all_regs.append(r.r_sp)
        if r.r_rip is not None: r.all_regs.append(r.r_rip)

        r.reg_indices = {r:i for i,r in enumerate(r.all_regs)}

        return r

class Abi(metaclass=AbiMeta):
    code_gen = None
    has_cmovecc = False

    # these three are filled automatically by the metaclass
    all_regs = [] # type: List[Register]
    gen_regs = 0
    reg_indices = {} # type: Dict[Register,int]

    r_ret = None # type: Optional[Register]
    r_sp = None # type: Optional[Register]
    r_rip = None # type: Optional[Register]
    r_scratch = [] # type: List[Register]
    r_pres = [] # type: List[Register]
    r_arg = [] # type: List[Register]

    # If True, stack space is reserved for function calls for all arguments,
    # even those that are passed in registers, such that the called function
    # could move those arguments to where they would be if all arguments were
    # passed by stack in the first place.
    shadow = False

    ptr_size = 0
    char_size = 0
    short_size = 0
    int_size = 0
    long_size = 0

    def __init__(self,*,assembly=False):
        self.assembly = assembly
        self.tuning = Tuning()
