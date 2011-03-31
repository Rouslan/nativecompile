from distutils.core import setup,Extension

setup(
    name='nativecompile',
    version='0.1.0',
    description='Transform Python bytecode into runnable machine code',
    author='Rouslan Korneychuk',
    packages=['nativecompile','nativecompile.tests'],
    ext_modules=[Extension(
        'nativecompile.pyinternals',
        ['nativecompile/pyinternals.c'])])

