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


from typing import Any,Dict,Optional,Sequence,Tuple

ARCHITECTURE = ... # type: str
COUNT_ALLOCS = ... # type: bool
GDB_JIT_SUPPORT = ... # type: bool
REF_DEBUG = ... # type: bool
member_offsets = ... # type: Dict[str,Dict[str,int]]
raw_addresses = ... # type: Dict[str,int]

def create_cell(*args): ...
def read_address(address : int,length : int) -> bytes: ...
def set_utility_funcs(value : Dict): ...
def build_class(__func,__name,*bases,**keywords) -> type: ...

class CompiledCode:
    size = ... # type: int
    start_addr = ... # type: int
    def __init__(self, parts) -> None: ...

class Function:
    annotations = ... # type: Dict
    body = ... # type: FunctionBody
    defaults = ... # type: Tuple
    __name__ = ... # type: str
    def __init__(self,
        body : 'FunctionBody',
        name : str,
        globals : Dict,
        doc : Any=None,
        defaults : Optional[Tuple]=None,
        kwdefaults : Optional[Dict]=None,
        closure : Optional[Tuple]=None,
        annotations : Optional[Dict]=None) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __get__(self,instance,owner): ...

class FunctionBody:
    cells = ... # type: int
    code = ... # type: Optional[CompiledCode]
    consts = ... # type: Tuple[object]
    free_names = ... # type: Tuple[str]
    kwonly_params = ... # type: int
    name = ... # type: str
    names = ... # type: Tuple[str]
    pos_params = ... # type: int
    var_kw = ... # type: bool
    var_pos = ... # type: bool
    def __init__(self,
        code : Optional[CompiledCode],
        offset : int,
        name : str,
        names : Sequence[str],
        pos_params : int,
        var_pos : bool,
        kwonly_params : int,
        var_kw : bool,
        free_names : Sequence[str],
        cells : int,
        consts : Sequence[object]) -> None: ...

class Generator:
    def close(self, *args, **kwargs): ...
    def __del__(self, *args, **kwargs): ...
