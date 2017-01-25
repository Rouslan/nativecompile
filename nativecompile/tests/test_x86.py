#  Copyright 2015 Rouslan Korneychuk
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

from .. import x86_ops as ops


class TestAddress(unittest.TestCase):
    def runTest(self):
        self.assertEqual(
            ops.Address(base=ops.eax).mod_rm_sib_disp(0),
            b'\x00'
        )

        self.assertEqual(
            ops.Address(base=ops.edi).mod_rm_sib_disp(1),
            b'\x0f'
        )

        self.assertEqual(
            ops.Address(127,ops.ecx).mod_rm_sib_disp(2),
            b'\x51\x7f'
        )

        self.assertEqual(
            ops.Address(2147483647,ops.edx).mod_rm_sib_disp(3),
            b'\x9a\xff\xff\xff\x7f'
        )

        self.assertEqual(
            ops.Address(-1).mod_rm_sib_disp(4),
            b'\x25\xff\xff\xff\xff'
        )

        self.assertEqual(
            ops.Address(base=ops.esi,index=ops.ebp).mod_rm_sib_disp(5),
            b'\x2c\x2e'
        )

        self.assertEqual(
            ops.Address(-128,index=ops.ebx,scale=8).mod_rm_sib_disp(6),
            b'\x74\xdd\x80'
        )

        self.assertEqual(
            ops.Address(base=ops.esp).mod_rm_sib_disp(7),
            b'\x3c\x24'
        )


