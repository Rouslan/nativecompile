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

from .. import code_gen
from .. import x86_ops

signed = code_gen.signed
unsigned = code_gen.unsigned
not_ = code_gen.not_

def resolve_jumps_mini(cgen,irc,chunks):
    displacement = 0

    late_chunks = []

    for chunk in reversed(chunks):
        if isinstance(chunk,code_gen.Target):
            chunk.displacement = displacement
        elif not isinstance(chunk,code_gen.IRJump):
            if not isinstance(chunk,bytes):
                ir2 = chunk.op.to_ir2(chunk.args)
                for i,a in enumerate(ir2.args):
                    if isinstance(a,code_gen.AddressType):
                        ir,o = cgen.process_indirection(chunk,ir2.overload,(i,))
                        ir2 = code_gen.Instr2(ir.op,o,ir.args)
                        break
                chunk = irc.compile_early(ir2,displacement)

            late_chunks.append(chunk)
            displacement -= len(chunk)

    return b''.join(reversed(late_chunks))

class TestHighLevel(unittest.TestCase):
    def compare(self,high,*low):
        irc = high.cgen.get_compiler(0,0,0)

        high = resolve_jumps_mini(high.cgen,irc,high.code)
        low = [b''.join((chunk if isinstance(chunk,bytes) else irc.compile_early(chunk,0)) for chunk in low_item) for low_item in low]

        if len(low) == 1:
            self.assertEqual(high,low[0])
        else:
            self.assertIn(high,low)

    def test_if(self):
        abi_ = x86_ops.CdeclAbi()
        S = lambda: code_gen.Stitch(abi_)
        sr = lambda r: code_gen.FixedRegister(abi_.reg_indices[r])
        ops = x86_ops.BasicOps(abi_)

        body = b'ABC'
        d_body = x86_ops.Displacement(len(body))

        self.compare(
            S().if_(sr(abi_.r_ret))(body).endif(),
            [ops.test(abi_.r_ret,abi_.r_ret),ops.jcc(x86_ops.test_Z,d_body),body])

        self.compare(
            S().if_(not_(sr(abi_.r_scratch[0])))(body).endif(),
            [ops.test(abi_.r_scratch[0],abi_.r_scratch[0]),ops.jcc(x86_ops.test_NZ,d_body),body])

        self.compare(
            S().if_(signed(0) > sr(abi_.r_scratch[1]))(body).endif(),
            [ops.test(abi_.r_scratch[1],abi_.r_scratch[1]),ops.jcc(x86_ops.test_GE,d_body),body])

        self.compare(
            S().if_(signed(sr(abi_.r_pres[0])) >= 0)(body).endif(),
            [ops.test(abi_.r_pres[0],abi_.r_pres[0]),ops.jcc(x86_ops.test_L,d_body),body])

        self.compare(
            S().if_(unsigned(5) < sr(abi_.r_pres[1]))(body).endif(),
            [ops.cmp(code_gen.Immediate(5),abi_.r_pres[1]),ops.jcc(x86_ops.test_NB,d_body),body])

        self.compare(
            S().if_(unsigned(code_gen.IndirectVar(0,sr(abi_.r_sp))) > abi_.r_ret)(body).endif(),
            [ops.cmp(x86_ops.Address(0,abi_.r_sp),abi_.r_ret),ops.jcc(x86_ops.test_BE,d_body),body],
            [ops.cmp(abi_.r_ret,x86_ops.Address(0,abi_.r_sp)),ops.jcc(x86_ops.test_NB,d_body),body])

        self.compare(
            S().if_(unsigned(code_gen.IndirectVar(16,sr(abi_.r_sp))) != 12)(body).endif(),
            [ops.cmp(code_gen.Immediate(12),x86_ops.Address(16,abi_.r_sp)),ops.jcc(x86_ops.test_E,d_body),body])

    def test_if_else(self):
        abi_ = x86_ops.CdeclAbi()
        S = lambda: code_gen.Stitch(abi_)
        sr = lambda r: code_gen.FixedRegister(abi_.reg_indices[r])
        ops = x86_ops.BasicOps(abi_)

        body1 = b'ABC'

        body2 = b'DEF'
        d_body2 = x86_ops.Displacement(len(body2))

        body1_full = body1 + ops.jmp(d_body2).overload.func(d_body2)
        d_body1 = x86_ops.Displacement(len(body1_full))

        self.compare(
            S().if_(sr(abi_.r_ret))(body1).else_()(body2).endif(),
            [ops.test(abi_.r_ret,abi_.r_ret),ops.jcc(x86_ops.test_Z,d_body1),body1_full,body2])


if __name__ == '__main__':
    unittest.main()
