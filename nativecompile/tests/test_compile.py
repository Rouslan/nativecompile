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


import sys
import subprocess
import io
import unittest


def exec_isolated(code):
    """Compile and run code in seperate process.

    A mistake in the compiler will cause a segfault most of the time. Running it
    in a seperate process will prevent it from terminating this test program.

    """
    s = subprocess
    with s.Popen(sys.executable,stdin=s.PIPE,stdout=s.PIPE,stderr=s.PIPE) as p:
        out,err = p.communicate(
            'import nativecompile;nativecompile.compile({})()'
            .format(repr(code)).encode('ascii'))
        rcode = p.returncode

    if rcode:
        raise Exception(
            'An error occured in the child process. Its output was:\n\n'+err.decode())

    return out.decode()


class TestCompile(unittest.TestCase):
    def compare_exec(self,code):
        """Run the code normally, then compile and run it and check to see that
        the results are the same."""

        res = io.StringIO()
        old = sys.stdout
        sys.stdout = res
        try:
            exec(code)
        finally:
            sys.stdout = old

        self.assertEqual(exec_isolated(code),res.getvalue())

    def test_compile(self):
        self.compare_exec('print("Hello World!")')

    def test_math(self):
        self.compare_exec('print((4 * 5 - 3) << 2)')

    def test_vars(self):
        self.compare_exec('''
x=23
y='goo'
print((x,y))
''')

    def test_for_loop(self):
        self.compare_exec('''
x = 0
for i in range(10):
    x += i
print(x)
''')

    def test_funcs(self):
        self.compare_exec('''
def a(x):
    return x * x

def b(x):
    return x + x

def c(x,y,z=None):
    return x + y + 9

print(a(b(9)))
print(c(1,2))
print(c(1,2,z=3))
''')

    def test_list_literal(self):
        self.compare_exec('print([3,2,1])')

    def test_if(self):
        self.compare_exec('''
x = 23
if x & 1:
    print('odd')
else:
    print('even')
''')

    def test_attr(self):
        self.compare_exec('''
x = " hello "
print(x.strip())
print(x.__class__.__name__)
''')

    def test_compare(self):
        self.compare_exec('''
a = 12
b = 14
c = [14,13,11]
d = a
print(a>b)
print(b>a)
print(a is b)
print(a is not b)
print(a is d)
print(a is not d)
print(b in c)
print(a in c)
print(b not in c)
print(a not in c)
''')

    def test_unpack(self):
        self.compare_exec('''
x = ['a','b','c']
a,b,c = x
print(a,b,c)
''')

    def test_class(self):
        self.compare_exec('''
class Thingy:
    a = 9
    def __init__(self,hi='go away'):
        self.hi = hi

    def __str__(self):
        return str(self.hi)

thing = Thingy()
print(thing)
''')

    def test_try_except_finally(self):
        self.compare_exec('''
a = 5
try:
    print(6/0)
except Exception:
    a = 1
print(a)

try:
    print('ok')
except:
    print('not ok')

try:
    a = 14
finally:
    a = 99
print(a)

try:
    a = "a"
    print(3/0)
except:
    a = "b"
finally:
    a += " c"
print(a)

try:
    try:
        1/0
    except:
        1/0
    a = 0
except:
    a = 2
print(a)

a=0
try:
    try:
        try:
            1/0
        finally:
            1/0
        a = -10
    finally:
        a = 2
except:
    a += 8
print(a)

a="car"
def func():
    global a
    try:
        try:
            try:
                1/0
            finally:
                return 777
            a = "horse"
        finally:
            a = "he"
    except:
        a += "llo"
print(func())
print(a)

a=0
try:
    for x in range(10):
        try:
            1/0
        finally:
            a += 2
except:
    a = a - 100
print(a)

a=0
for x in range(10):
    try:
        if x == 6: continue
        a += 1
    finally:
        a += 10
print(a)
''')

    def test_string_optimize(self):
        self.compare_exec('''
a = 'watermelon'
a += 'A' # so 'a' is no longer interned
oldid = id(a)
a += 'B'
print(oldid == id(a))
a = a + 'C'
print(oldid == id(a))
print(a)
''')

    def test_with(self):
        self.compare_exec('''
class Thing:
    def __enter__(self):
        print('start')
    def __exit__(self,exc_type,exc_value,traceback):
        print('finish')
        return exc_type == ZeroDivisionError

with Thing():
    print('stuff')

try:
    with Thing():
        1/0
except:
    print('apples!')

try:
    with Thing() as t:
        t.invalid_attribute
except:
    print('mushrooms!')
''')

    def test_var_kw_func(self):
        self.compare_exec('''
a = [1,2,3]
b = {'a':'A','b':'B','c':'C'}
def f(*args,**kwds):
  print(args)
  print(sorted(kwds.items(),key=lambda x: x[0]))

f(*a)
f(**b)
f(*a,**b)
f(0,*a,**b)
''')

    def test_yield(self):
        self.compare_exec('''
import sys

def gentest(x):
    try:
        yield 1
    except:
        print(sys.exc_info()[0])
        raise

    a = yield 2
    b = yield 2 * a
    for y in range(1):
        yield b
        yield x * y

    try:
        1/0
    except:
        print(sys.exc_info()[0])
        yield 9
        print(sys.exc_info()[0])

g = gentest(5)
print(next(g))
print(g.send(9))
print(g.send('yam'))
print(next(g))
print(next(g))

print(sys.exc_info()[0])
print(next(g))
print(sys.exc_info()[0])
try:
    next(g)
except StopIteration:
    print('almost done')

g = gentest(8)
print(next(g))
try:
    g.throw(ValueError,'hi')
except ValueError:
    print('done')
''')


if __name__ == '__main__':
    unittest.main()

