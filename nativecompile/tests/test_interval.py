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


import unittest

from ..dinterval import *

class MyTestCase(unittest.TestCase):
    def test_or(self):
        a = DInterval([(0,3),(4,7),(11,23)])

        self.assertEqual(a | DInterval(Interval(5,6)),a)

        self.assertEqual(a | Interval(4,7),a)

        self.assertEqual(a | Interval(9,10),
            DInterval([(0,3),(4,7),(9,10),(11,23)]))

        self.assertEqual(a | Interval(9,11),
            DInterval([(0,3),(4,7),(9,23)]))

        self.assertEqual(a | Interval(7,10),
            DInterval([(0,3),(4,10),(11,23)]))

        self.assertEqual(a | Interval(7,11),
            DInterval([(0,3),(4,23)]))

        self.assertEqual(a | Interval(-1,100),
            DInterval([(-1,100)]))

        self.assertEqual(a | DInterval([(3,5),(9,12)]),
            DInterval([(0,7),(9,23)]))

    def test_and(self):
        a = DInterval([(0,3),(4,7),(11,23)])

        self.assertEqual(a & DInterval([(4,7)]),DInterval([(4,7)]))
        self.assertEqual(a & Interval(5,6),DInterval([(5,6)]))
        self.assertEqual(a & Interval(40,44),DInterval())
        self.assertEqual(a & Interval(-1,24),a)
        self.assertEqual(a & Interval(1,22),DInterval([(1,3),(4,7),(11,22)]))
        self.assertEqual(a & Interval(2,6),DInterval([(2,3),(4,6)]))
        self.assertEqual(a & DInterval([(-1,6),(9,10)]),DInterval([(0,3),(4,6)]))
        self.assertEqual(DInterval([(1,44)]) & DInterval([(0,20),(28,58)]),DInterval([(1,20),(28,44)]))


if __name__ == '__main__':
    unittest.main()
