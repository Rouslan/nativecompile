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


import unittest

from .. import intermediate as ir
from .. import abi
from ..dinterval import *


class DRegister(abi.Register):
    def __init__(self,reg,size=3):
        self.reg = reg
        self.size = size

class DAddress:
    def __init__(self,offset=0,base=None,index=None,scale=1):
        self.offset = offset
        self.base = base
        self.index = index
        self.scale = scale

dummy_op = ir.OpDescription(
    'dummy_op',
    [ir.Overload([],(lambda: None))],
    [])

read_op = ir.OpDescription(
    'read_op',
    [ir.Overload([ir.FixedRegister],(lambda r: None)),
        ir.Overload([ir.AddressType],(lambda a: None))],
    [ir.ParamDir(True,False)])

write_op = ir.OpDescription(
    'write_op',
    [ir.Overload([ir.FixedRegister],(lambda r: None)),
        ir.Overload([ir.AddressType],(lambda a: None))],
    [ir.ParamDir(False,True)])

readwrite_op = ir.OpDescription(
    'readwrite_op',
    [ir.Overload([ir.FixedRegister],(lambda r: None))],
    [ir.ParamDir(True,True)])

move_op = ir.OpDescription(
    'move_op',
    [ir.Overload([ir.FixedRegister,ir.FixedRegister],(lambda a,b: None)),
        ir.Overload([ir.AddressType,ir.FixedRegister],(lambda a,b: None)),
        ir.Overload([ir.FixedRegister,ir.AddressType],(lambda a,b: None))],
    [ir.ParamDir(True,False),ir.ParamDir(False,True)])


class DIRCompiler(ir.IRCompiler):
    def __init__(self,abi):
        self.abi = abi

    def prolog(self):
        return []

    def epilog(self):
        return []

    def get_reg(self,index,size):
        return DRegister(self.abi.reg_indices[index],size)

    def get_stack_addr(self,index,size,sect):
        return DAddress(index*self.abi.ptr_size,self.abi.r_sp)

    def get_displacement(self,amount,force_wide):
        return amount

    def get_immediate(self,val,size):
        return val

    def nops(self,size):
        return []

def splice(seq,start,size,replacement):
    return tuple(seq[0:start]) + replacement + tuple(seq[start+size:])

class IndirectionAdapter(ir.RegAllocatorOverloads):
    def __init__(self,base_op,param,offset,base,index,scale) -> None:
        regs = base + index
        super().__init__(splice(base_op.param_dirs,param,1,(ir.ParamDir(True,False),) * regs))

        self.base_op = base_op
        self.param = param
        self.offset = offset
        self.base = base
        self.index = index
        self.scale = scale

    @property
    def name(self):
        return self.base_op.name

    def _to_base_args(self,args):
        return splice(args,self.param,self.base + self.index,(ir.AddressType(),))

    def wrap_overload(self,o):
        regs = self.base + self.index

        return ir.Overload(
            splice(o.params,self.param,1,(ir.FixedRegister,) * regs),
            (lambda *args: None))

    def best_match(self,args):
        return self.wrap_overload(self.base_op.best_match(self._to_base_args(args)))

    def to_ir2(self,args):
        o = self.base_op.exact_match(self._to_base_args(args))
        return ir.Instr2(self.base_op,self.wrap_overload(o),args)

    def assembly(self,args,addr,binary,annot=None):
        m = DAddress(
            self.offset,
            args[self.param] if self.base else None,
            args[self.param+self.base] if self.index else None,
            self.scale)

        return self.base_op.assembly(splice(args,self.param,self.base+self.index,(m,)),addr,binary,annot)

class DOpGen(ir.JumpCondOpGen):
    def __init__(self):
        super().__init__(DAbi())

    def _basic_jump_if(self,dest,cond):
        raise AssertionError()

    def _call_impl(self,func,args,store_ret):
        raise AssertionError()

    def load_addr(self,addr,dest):
        raise AssertionError()

    def shift(self,src,shift_dir,amount,dest):
        raise AssertionError()

    def get_cur_func_arg(self,i):
        raise AssertionError()

    def move(self,src,dest):
        return [ir.Instr(move_op,src,dest)]

    def jump_table(self,val,targets):
        raise AssertionError()

    def jump(self,dest):
        raise AssertionError()

    def process_indirection(self,instr,ov,inds):
        i = inds[0]
        arg = instr.args[i]

        scale = arg.scale or self.abi.ptr_size

        insert = ()
        if arg.base is not None:
            insert += (arg.base,)
        if arg.index is not None:
            insert += (arg.index,)

        op = IndirectionAdapter(
            instr.op,
            i,
            arg.offset,
            arg.base is not None,
            arg.index is not None,
            scale)
        return ir.Instr(op,*splice(instr.args,i,1,insert)),op.wrap_overload(ov)

    def unary_op(self,a,dest,op_type):
        raise AssertionError()

    def bin_op(self,a,b,dest,op_type):
        raise AssertionError()

    def get_return_address(self,v):
        raise AssertionError()

    def get_compiler(self,regs_used,stack_used,args_used):
        return DIRCompiler(self.abi)

    def return_value(self,v):
        raise AssertionError()

class DAbi(abi.Abi):
    code_gen = None

    r_ret = DRegister(0)
    r_sp = DRegister(1)
    r_scratch = [DRegister(i) for i in range(2,4)]
    r_pres = [DRegister(i) for i in range(4,10)]
    r_arg = [DRegister(i) for i in range(10,16)]

    ptr_size = 5
    char_size = 1
    short_size = 2
    int_size = 3
    long_size = 5

class MyTestCase(unittest.TestCase):
    def test_calc_var_intervals(self):
        a = ir.Var('a')
        b = ir.Var('b')
        c = ir.Var('c')
        else_ = ir.Target()
        endif = ir.Target()

        code = [
            ir.Instr(dummy_op),                   # 0
            ir.Instr(write_op,a),                 # 1
            ir.Instr(dummy_op),                   # 2
            ir.CreateVar(b,ir.FixedRegister(0)),  # 3
            ir.IRJump(else_,True,0),              # 4
            ir.Instr(dummy_op),                   # 5
            ir.Instr(write_op,b),                 # 6
            ir.Instr(write_op,c),                 # 7
            ir.Instr(read_op,a),                  # 8
            ir.Instr(dummy_op),                   # 9
            ir.IRJump(endif,False,0),             # 10
            else_,                                # 11
            ir.Instr(read_op,a),                  # 12
            ir.CreateVar(c,ir.FixedRegister(1)),  # 13
            endif,                                # 14
            ir.Instr(readwrite_op,a),             # 15
            ir.Instr(read_op,b),                  # 16
            ir.Instr(read_op,c)                   # 17
        ]
        ir.calc_var_intervals(code)

        self.assertEqual(a.lifetime.intervals,DInterval([(2,16)]))
        self.assertEqual(b.lifetime.intervals,DInterval([(4,5),(7,17)]))
        self.assertEqual(c.lifetime.intervals,DInterval([(8,11),(14,18)]))

        # if a variable is first written-to in a branch and there exists
        # another branch where it is not written-to, it cannot be read-from
        # after the branches converge
        a = ir.Var('a')
        code = [
            ir.Instr(dummy_op),
            ir.IRJump(else_,True,0),
            ir.Instr(dummy_op),
            ir.Instr(write_op,a),
            ir.Instr(dummy_op),
            ir.IRJump(endif,False,0),
            else_,
            ir.Instr(dummy_op),
            endif,
            ir.Instr(dummy_op),
            ir.Instr(read_op,a)
        ]
        self.assertRaises(ValueError,ir.calc_var_intervals,code)

        target1 = ir.Target()
        target2 = ir.Target()
        target3 = ir.Target()
        target4 = ir.Target()
        target5 = ir.Target()
        target6 = ir.Target()
        target7 = ir.Target()

        a = ir.Var('a')
        b = ir.Var('b')
        c = ir.Var('c')
        d = ir.Var('d')
        e = ir.Var('e')
                                                              #     a b c d e
        code = [ir.CreateVar(a,ir.FixedRegister(0)),          # 0   w
            ir.CreateVar(b,ir.FixedRegister(1)),              # 1   | w
            ir.Instr(read_op,ir.IndirectVar(64,b,None,0,0)),  # 2   | r
            ir.Instr(write_op,c),                             # 3   | | w
            ir.Instr(read_op,ir.IndirectVar(56,b,None,0,0)),  # 4   | r |
            ir.Instr(write_op,d),                             # 5   |   | w
            ir.Instr(read_op,c),                              # 6   |   r |
            ir.IRJump(target1,True,0),                        # 7   |   | |
            ir.Instr(read_op,d),                              # 8   |   | r
            ir.IRJump(target1,True,0),                        # 9   |   | |
            ir.Instr(read_op,c),                              # 10  |   r |
            ir.Instr(read_op,d),                              # 11  |   | r
            ir.Instr(read_op,a),                              # 12  r   | |
            ir.InvalidateRegs([6,7,8,12,13,14],[]),           # 13  |   | |
            ir.CreateVar(e,ir.FixedRegister(8)),              # 14  |   | | w
            ir.Instr(read_op,e),                              # 15  |   | | r
            ir.IRJump(target7,True,0),                        # 16  |   | | |
            ir.InvalidateRegs([6,7,8,9,10,11,12,13,14],[]),   # 17  |
            ir.CreateVar(e,ir.FixedRegister(8)),              # 18  |       w
            ir.Instr(read_op,e),                              # 19  |       r
            ir.IRJump(target2,True,0),                        # 20  |       |
            ir.Instr(dummy_op),                               # 21
            ir.IRJump(target6,False,0),                       # 22
            target7,                                          # 23  |   | | |
            ir.Instr(read_op,e),                              # 24  |   | | r
            ir.InvalidateRegs([6,7,8,10,11,12,13,14],[]),     # 25  |   | | |
            ir.Instr(read_op,e),                              # 26  |   | | r
            target1,                                          # 27  |   | |
            ir.Instr(read_op,c),                              # 28  |   r |
            ir.Instr(read_op,a),                              # 29  r     |
            ir.InvalidateRegs([6,7,8,11,12,13,14],[]),        # 30  |     |
            ir.CreateVar(e,ir.FixedRegister(8)),              # 31  |     | w
            ir.Instr(read_op,e),                              # 32  |     | r
            ir.IRJump(target5,True,0),                        # 33  |     | |
            ir.Instr(read_op,d),                              # 34  |     r
            ir.Instr(read_op,a),                              # 35  r
            ir.InvalidateRegs([6,7,8,11,12,13,14],[]),        # 36  |
            ir.CreateVar(e,ir.FixedRegister(8)),              # 37  |       w
            ir.Instr(read_op,e),                              # 38  |       r
            ir.IRJump(target4,True,0),                        # 39  |       |
            ir.Instr(dummy_op),                               # 40  |
            ir.InvalidateRegs([6,7,8,10,11,12,13,14],[]),     # 41  |
            ir.CreateVar(e,ir.FixedRegister(8)),              # 42  |       w
            ir.Instr(read_op,e),                              # 43  |       r
            ir.IRJump(target3,True,0),                        # 44  |       |
            target2,                                          # 45  |       |
            ir.Instr(dummy_op),                               # 46  |       |
            ir.Instr(dummy_op),                               # 47  |       |
            ir.Instr(read_op,a),                              # 48  r       |
            ir.InvalidateRegs([6,7,8,12,13,14],[]),           # 49          |
            target3,                                          # 50          |
            ir.Instr(dummy_op),                               # 51          |
            target4,                                          # 52          |
            ir.Instr(dummy_op),                               # 53          |
            target5,                                          # 54          |
            ir.Instr(read_op,e),                              # 55          r
            target6]                                          # 56
        ir.calc_var_intervals(code)

        self.assertEqual(a.lifetime.intervals,DInterval([(1,21),(23,49)]))
        self.assertEqual(b.lifetime.intervals,DInterval([(2,5)]))
        self.assertEqual(c.lifetime.intervals,DInterval([(4,17),(23,29)]))
        self.assertEqual(d.lifetime.intervals,DInterval([(6,17),(23,35)]))
        self.assertEqual(e.lifetime.intervals,DInterval([(15,17),(19,21),(23,27),(32,34),(38,40),(43,56)]))

    def test_var_block(self):
        block = ir.Block(3)
        a,b,c = block.parts
        else_ = ir.Target()
        endif = ir.Target()

        code = [
            ir.Instr(dummy_op),                   # 0
            ir.Instr(write_op,a),                 # 1
            ir.Instr(dummy_op),                   # 2
            ir.CreateVar(b,ir.FixedRegister(0)),  # 3
            ir.IRJump(else_,True,0),              # 4
            ir.Instr(dummy_op),                   # 5
            ir.Instr(write_op,b),                 # 6
            ir.Instr(write_op,c),                 # 7
            ir.Instr(read_op,a),                  # 8
            ir.Instr(dummy_op),                   # 9
            ir.IRJump(endif,False,0),             # 10
            else_,                                # 11
            ir.Instr(read_op,a),                  # 12
            ir.CreateVar(c,ir.FixedRegister(1)),  # 13
            endif,                                # 14
            ir.Instr(readwrite_op,a),             # 15
            ir.Instr(read_op,b),                  # 16
            ir.Instr(read_op,c)                   # 17
        ]
        ir.calc_var_intervals(code)

        self.assertEqual(a.lifetime.intervals,DInterval([(2,16)]))
        self.assertEqual(b.lifetime.intervals,DInterval([(4,5),(7,17)]))
        self.assertEqual(c.lifetime.intervals,DInterval([(8,11),(14,18)]))
        self.assertEqual(block.lifetime.intervals,DInterval([(2,18)]))


if __name__ == '__main__':
    unittest.main()
