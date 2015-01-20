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


import imp
import importlib
import importlib._bootstrap
import os.path
import warnings
import types
import sys

from .compile import compile
from . import pyinternals


class Finder:
    def __init__(self,path):
        self.path = path

    def find_module(self,fullname):
        tail_module = fullname.rpartition('.')[2]
        base_path = os.path.join(self.path, tail_module)
        if os.path.isdir(base_path) and importlib._case_ok(self.path, tail_module):
            for suffix,loader in PYTHON_FILE_SUFFIXES:
                init_filename = '__init__' + suffix
                full_path = os.path.join(base_path, init_filename)
                if (os.path.isfile(full_path) and
                        importlib._case_ok(base_path, init_filename)):
                    return loader(fullname,full_path)
            else:
                msg = "Not importing directory {}: missing __init__"
                warnings.warn(msg.format(base_path), ImportWarning)
        for suffix,loader in FILE_SUFFIXES:
            mod_filename = tail_module + suffix
            full_path = os.path.join(self.path, mod_filename)
            if os.path.isfile(full_path) and importlib._case_ok(self.path, mod_filename):
                return loader(fullname,full_path)
        
        return None


def xloader(base):
    class inner(base):
        @importlib._bootstrap.module_for_loader
        def _load_module(self, module, *, sourceless=False):
            name = module.__name__
            code_object = self.get_code(name)
            module.__file__ = self.get_filename(name)
            if not sourceless:
                module.__cached__ = imp.cache_from_source(module.__file__)
            else:
                module.__cached__ = module.__file__
            module.__package__ = name
            if self.is_package(name):
                module.__path__ = [module.__file__.rsplit(path_sep, 1)[0]]
            else:
                module.__package__ = module.__package__.rpartition('.')[0]
            module.__loader__ = self

            ccode = compile(code_object)

            # stick the CompiledCode object here to keep it alive
            module.__nativecompile_compiled_code__ = ccode

            pyinternals.cep_exec(ccode.entry_points[0], module.__dict__)
            return module

    return inner


SourceLoader = xloader(importlib._bootstrap._SourceFileLoader)
SourcelessLoader = xloader(importlib._bootstrap._SourcelessFileLoader)
ExtensionLoader = importlib._bootstrap._ExtensionFileLoader

PYTHON_FILE_SUFFIXES = (
    [(suffix,SourceLoader) for suffix,mode,type in imp.get_suffixes()
     if type == imp.PY_SOURCE] +
    [(suffix,SourcelessLoader) for suffix,mode,type in imp.get_suffixes()
     if type == imp.PY_COMPILED])

FILE_SUFFIXES = (
    PYTHON_FILE_SUFFIXES +
    [(suffix,ExtensionLoader) for suffix,mode,type in imp.get_suffixes()
     if type == imp.C_EXTENSION])


def path_hook(path):
    if os.path.isdir(path):
        return Finder(path)
    else:
        raise ImportError("only directories are supported")

def install_importer():
    sys.path_hooks.append(path_hook)

def uninstall_importer():
    sys.path_hooks.remove(path_hook)

