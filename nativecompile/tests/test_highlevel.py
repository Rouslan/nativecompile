
import unittest

from .. import abi
from .. import code_gen

signed = code_gen.signed
unsigned = code_gen.unsigned
not_ = code_gen.not_

def resolve_jumps_mini(chunks):
    displacement = 0

    late_chunks = []

    for chunk in reversed(chunks):
        if isinstance(chunk,code_gen.JumpTarget):
            chunk.displacement = displacement
        else:
            if isinstance(chunk,code_gen.DelayedCompileEarly):
                chunk = b''.join(chunk.compile(displacement))
            
            late_chunks.append(chunk)
            displacement += len(chunk)

    return b''.join(reversed(late_chunks))

class TestHighLevel(unittest.TestCase):
    def compare(self,high,low):
        self.assertEqual(resolve_jumps_mini(high.code),b''.join(low))

    def runTest(self):
        abi_ = abi.CdeclAbi()
        S = lambda: code_gen.Stitch(abi_)

        body = b'ABC'
        d_body = abi_.Displacement(len(body))

        self.compare(
            S().if_(code_gen.R_RET)(body).endif(),
            abi_.op.test(abi_.r_ret,abi_.r_ret) + abi_.op.jcc(abi_.test_Z,d_body) + [body])

        self.compare(
            S().if_(code_gen.R_RET)(body).endif(),
            abi_.op.test(abi_.r_ret,abi_.r_ret) + abi_.op.jcc(abi_.test_Z,d_body) + [body])

        self.compare(
            S().if_(not_(code_gen.R_SCRATCH1))(body).endif(),
            abi_.op.test(abi_.r_scratch[0],abi_.r_scratch[0]) + abi_.op.jcc(abi_.test_NZ,d_body) + [body])

        self.compare(
            S().if_(signed(0) > abi_.r_scratch[1])(body).endif(),
            abi_.op.test(abi_.r_scratch[1],abi_.r_scratch[1]) + abi_.op.jcc(abi_.test_GE,d_body) + [body])

        self.compare(
            S().if_(signed(abi_.r_pres[0]) >= 0)(body).endif(),
            abi_.op.test(abi_.r_pres[0],abi_.r_pres[0]) + abi_.op.jcc(abi_.test_L,d_body) + [body])

        self.compare(
            S().if_(unsigned(5) < abi_.r_pres[1])(body).endif(),
            abi_.op.cmp(5,abi_.r_pres[1]) + abi_.op.jcc(abi_.test_NB,d_body) + [body])

        addr = abi_.Address(base=abi_.r_sp)
        self.compare(
            S().if_(unsigned(addr) > abi_.r_ret)(body).endif(),
            abi_.op.cmp(abi_.r_ret,addr) + abi_.op.jcc(abi_.test_NB,d_body) + [body])

        addr = code_gen.addr(16,code_gen.R_SP)
        s = S()
        self.compare(
            s.if_(unsigned(addr) != 12)(body).endif(),
            abi_.op.cmpl(12,addr(s)) + abi_.op.jcc(abi_.test_E,d_body) + [body])


if __name__ == '__main__':
    unittest.main()
