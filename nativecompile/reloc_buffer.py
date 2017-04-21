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


import io


class RelocAddress:
    __slots__ = 'val'
    def __init__(self,val):
        self.val = val
    def get(self,addr_size,pos,base=0):
        raise NotImplementedError()

class RelocAbsAddress(RelocAddress):
    __slots__ = ()
    def get(self,addr_size,pos,base=0):
        return (self.val + base).to_bytes(addr_size,byteorder='little',signed=False)

class RelocRelAddress(RelocAddress):
    __slots__ = ()
    def get(self,addr_size,pos,base=0):
        return (self.val + pos + base).to_bytes(addr_size,byteorder='little',signed=False)

def _offset_addrs(addrs,amount):
    return ((p + amount,a) for p,a in addrs)

class RelocBytes:
    __slots__ = 'data','addrs'
    def __init__(self,data,addrs=()):
        self.data = data
        self.addrs = addrs

    def __add__(self,b):
        if isinstance(b,RelocBytes):
            return RelocBytes(self.data+b.data,self.addrs + tuple(_offset_addrs(b.addrs,len(self.data))))

        return RelocBytes(self.data+b,self.addrs)

    def __radd__(self,b):
        return RelocBytes(b+self.data,_offset_addrs(self.addrs,len(b)))

    def __len__(self):
        return len(self.data)


class RelocBufferLike:
    def __init__(self,dest):
        self.buff = dest

    def seek(self,*args):
        return self.buff.seek(*args)

    def tell(self):
        return self.buff.tell()

    def write(self,data):
        raise NotImplementedError()

    def _deferred_write(self,x):
        if x.data is not None:
            self.write(x.data)
        else:
            pos = self.buff.tell()
            x.targets.append((self,pos))
            self.buff.seek(pos + x.size)

class DeferredWrite:
    """A placeholder that allows supplying a value after a call to
    RelocBufferLike.write"""

    def __init__(self,size):
        self.size = size
        self.targets = []
        self.data = None

    def __call__(self,data,*args,**kwds):
        assert self.data is None, 'this has already been written-to'

        self.data = bytes(data)
        assert len(self.data) == self.size

        while self.targets:
            buff,pos = self.targets[-1]

            prev = buff.dest.tell()
            buff.dest.seek(pos)
            try:
                buff.write(data)
            finally:
                buff.dest.seek(prev)

            self.targets.pop()

class RelocBuffer(RelocBufferLike):
    """A file-like object containing addresses that can be rebased"""

    def __init__(self,addr_size):
        super().__init__(io.BytesIO())
        self.addr_size = addr_size
        self.addrs = []

    def write(self,data):
        if isinstance(data,RelocAddress):
            pos = self.buff.tell()
            self.addrs.append((pos,data))
            data = data.get(self.addr_size,pos)
        elif isinstance(data,RelocBytes):
            self.addrs.extend(_offset_addrs(data.addrs,self.tell()))
            data = data.data
        elif isinstance(data,DeferredWrite):
            self._deferred_write(data)
            return

        self.buff.write(data)

    def rebase(self,offset):
        with self.buff.getbuffer() as b:
            for pos,a in self.addrs:
                addr = a.get(self.addr_size,pos,offset)
                b[pos:pos+len(addr)] = addr

class NonRelocWrapper(RelocBufferLike):
    def __init__(self,dest,addr_size):
        super().__init__(dest)
        self.addr_size = addr_size

    def write(self,data):
        if isinstance(data,RelocAddress):
            data = data.get(self.addr_size,self.tell())
        elif isinstance(data,RelocBytes):
            data = data.data
        elif isinstance(data,DeferredWrite):
            self._deferred_write(data)
            return

        self.buff.write(data)
