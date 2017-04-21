__all__ = ['CallConvType','Param','Function','CompilationUnit']

import enum
from typing import NamedTuple

from . import c_types

class CallConvType(enum.Enum):
    default = 0
    utility = 1

Param = NamedTuple('Param',[('name',str),('type',c_types.CType)])

class Function:
    def __init__(self,code,padding=0,offset=0,name=None,annotation=None,returns=None,params=None,callconv=CallConvType.default):
        self.code = code

        # the size, in bytes, of the trailing padding (nop instructions) in
        # "code"
        self.padding = padding

        self.offset = offset
        self.name = name
        self.annotation = annotation or []
        self.returns = returns
        self.params = params or []
        self.callconv = callconv

    def __len__(self):
        return len(self.code)

class CompilationUnit:
    def __init__(self,functions):
        self.functions = functions

    def __len__(self):
        return sum(map(len,self.functions))

    def write(self,out):
        for f in self.functions:
            out.write(f.code)
