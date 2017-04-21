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


import os.path

from setuptools import setup,Extension
from setuptools.command.build_ext import build_ext
from distutils import log
from distutils.util import split_quoted
from distutils.dep_util import newer
from distutils.dir_util import mkpath

import make_ast_loader

DUMMY_PATH = 'DUMMY_PATH'

base_dir = os.path.dirname(os.path.realpath(__file__))


class CustomBuildExt(build_ext):
    user_options = build_ext.user_options + [
        ('cflags=',None,
            'extra command line arguments for the C compiler')]

    def initialize_options(self):
        build_ext.initialize_options(self)

        self.cflags = ''

    def finalize_options(self):
        build_ext.finalize_options(self)

        self.cflags = split_quoted(self.cflags)

        for e in self.extensions:
            e.extra_compile_args += self.cflags
            for i,s in enumerate(e.sources):
                if s is DUMMY_PATH:
                    e.sources[i] = os.path.join(self.build_temp,'astloader.c')

    def build_extensions(self):
        output = os.path.join(self.build_temp,'astloader.c')
        input = os.path.join(base_dir,'cpython','Python.asdl')
        if self.force or newer(input,output) or newer(make_ast_loader.__file__,output):
            log.info('creating astloader.c')
            if not self.dry_run:
                mkpath(self.build_temp)
                make_ast_loader.create(input,output)

        build_ext.build_extensions(self)


with open('README') as rfile:
    readme = rfile.read()

setup(
    name='NativeCompile',
    version='0.3.0',
    description='Compile Python code into native machine code',
    long_description=readme,
    author='Rouslan Korneychuk',
    author_email='rouslank@msn.com',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Software Development :: Compilers'],
    packages=['nativecompile','nativecompile.tests'],
    test_suite='nativecompile.tests',
    ext_modules=[
        Extension('nativecompile.pyinternals',['nativecompile/pyinternals.c']),
        Extension('nativecompile.astloader',[DUMMY_PATH])],
    headers=['nativecompile/pyinternals.h'],
    install_requires=['typing>=3.5.3'],
    zip_safe=True,
    cmdclass={'build_ext':CustomBuildExt})

