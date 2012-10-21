
import unittest

from ..x86_abi import CdeclAbi
from .. import compile_raw as cr

signed = cr.signed
unsigned = cr.unsigned

class TestHighLevel(unittest.TestCase):
    def compare(self,high,low):
        self.assertEqual(cr.join(high.code),low)

    def runTest(self):
        f = cr.Frame(CdeclAbi.ops,CdeclAbi,cr.Tuning(),64)
        d_zero = f.Displacement(0)

        self.compare(
            f().if_[f.r_ret](b''),
            f.op.test(f.r_ret,f.r_ret) + f.op.jcc(f.test_Z,d_zero))

        body = b'ABC'
        self.compare(
            f().if_[f.r_ret](body),
            f.op.test(f.r_ret,f.r_ret) + f.op.jcc(f.test_Z,f.Displacement(len(body))) + body)

        self.compare(
            f().if_[signed(f.r_scratch[0]) == 0](b''),
            f.op.test(f.r_scratch[0],f.r_scratch[0]) + f.op.jcc(f.test_NZ,d_zero))

        self.compare(
            f().if_[signed(0) > f.r_scratch[1]](b''),
            f.op.test(f.r_scratch[1],f.r_scratch[1]) + f.op.jcc(f.test_GE,d_zero))

        self.compare(
            f().if_[signed(f.r_pres[0]) >= 0](b''),
            f.op.test(f.r_pres[0],f.r_pres[0]) + f.op.jcc(f.test_L,d_zero))

        self.compare(
            f().if_[unsigned(5) < f.r_pres[1]](b''),
            f.op.cmp(5,f.r_pres[1]) + f.op.jcc(f.test_NB,d_zero))

        self.compare(
            f().if_[unsigned(f.stack[0]) > f.r_ret](b''),
            f.op.cmp(f.r_ret,f.stack[0]) + f.op.jcc(f.test_NB,d_zero))

        self.compare(
            f().if_[unsigned(f.stack[1]) != 12](b''),
            f.op.cmpl(12,f.stack[1]) + f.op.jcc(f.test_E,d_zero))


if __name__ == '__main__':
    unittest.main()
