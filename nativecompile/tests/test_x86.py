
import unittest

from .. import x86_ops as ops



class TestAddress(unittest.TestCase):
    def runTest(self):
        self.assertEqual(
            ops.Address(base=ops.eax).mod_rm_sib_disp(),
            (0b00,0b000,b'')
        )
        
        self.assertEqual(
            ops.Address(base=ops.edi).mod_rm_sib_disp(),
            (0b00,0b111,b'')
        )
        
        self.assertEqual(
            ops.Address(127,ops.ecx).mod_rm_sib_disp(),
            (0b01,0b001,b'\x7f')
        )
        
        self.assertEqual(
            ops.Address(2147483647,ops.edx).mod_rm_sib_disp(),
            (0b10,0b010,b'\xff\xff\xff\x7f')
        )
        
        self.assertEqual(
            ops.Address(-1).mod_rm_sib_disp(),
            (0b00,0b101,b'\xff\xff\xff\xff')
        )
        
        self.assertEqual(
            ops.Address(base=ops.esi,index=ops.ebp).mod_rm_sib_disp(),
            (0b00,0b100,b'\x2e')
        )
        
        self.assertEqual(
            ops.Address(-128,index=ops.ebx,scale=8).mod_rm_sib_disp(),
            (0b01,0b100,b'\xdd\x80')
        )
        
        self.assertEqual(
            ops.Address(base=ops.esp).mod_rm_sib_disp(),
            (0b00,0b100,b'\x24')
        )


