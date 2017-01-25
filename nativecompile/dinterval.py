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


__all__ = 'Interval','DInterval'

from typing import Container,Generic,Iterable,List,Optional,Tuple,TYPE_CHECKING,TypeVar,Union

T_co = TypeVar('T_co',covariant=True)

# subclasses tuple for easy immutability
class Interval(tuple,Container[T_co],Generic[T_co]):
    """A left-closed right-open interval between two points.

    For any value p, p is in Lifetime(a,b) if a <= p < b.

    """
    __slots__ = ()

    # noinspection PyInitNewSignature
    def __new__(cls,start,end=None):
        if end is None:
            if isinstance(start,Interval):
                return start
            start,end = start
        if start > end: end = start
        return tuple.__new__(cls,(start,end))

    if TYPE_CHECKING:
        # noinspection PyUnusedLocal
        def __init__(self,start : Union[T_co,'Interval[T_co]',Tuple[T_co,T_co]],end : Optional[T_co]=None) -> None:
            pass

    @property
    def start(self) -> T_co: return self[0]

    @property
    def end(self) -> T_co: return self[1]

    def __bool__(self): return self.start != self.end

    def __len__(self):
        return self.end - self.start

    def __contains__(self,p):
        return self.start <= p < self.end

    def __repr__(self):
        return 'Interval({},{})'.format(*self)

    def __str__(self):
        return '\u2205' if not self else '[{},{})'.format(*self)

T = TypeVar('T')

class DInterval(Container[T],Iterable[Interval[T]],Generic[T]):
    """A discontinuous interval.

    This can be thought of as a union of left-closed right-open intervals. The
    interval doesn't actually have to be discontinuous since continuous
    intervals can be represented just as easily.

    """
    def __init__(self,x : Union[Interval[T],Iterable[Union[Interval[T],Tuple[T,T]]],None]=None) -> None:
        self._parts = [] # type: List[Interval[T]]
        if x is not None:
            if isinstance(x,DInterval):
                self._parts = x._parts[:]
            elif isinstance(x,Interval):
                self._parts.append(x)
            else:
                self._parts = [Interval(p) for p in x]

    @property
    def global_start(self) -> Optional[T]:
        if not self._parts: return None
        return self._parts[0].start

    @property
    def global_end(self) -> Optional[T]:
        if not self._parts: return None
        return self._parts[-1].end

    def __bool__(self):
        return len(self._parts) != 0

    def _ior_part(self,b,lo=0):
        if not b: return lo

        hi = len(self._parts)
        assert lo <= hi <= len(self._parts)

        while lo < hi:
            mid = (lo+hi)//2
            if self._parts[mid].end < b.start: lo = mid+1
            else: hi = mid

        assert hi == lo
        while hi < len(self._parts) and self._parts[hi].start <= b.end:
            hi += 1
        if hi > lo:
            b = Interval(min(b.start,self._parts[lo].start),max(b.end,self._parts[hi-1].end))

        self._parts[lo:hi] = [b]

        return lo

    def __ior__(self,b):
        if isinstance(b,DInterval):
            ins = 0
            for p in b._parts:
                ins = self._ior_part(p,ins)
            return self

        if isinstance(b,Interval):
            self._ior_part(b)
            return self

        return NotImplemented

    def __or__(self,b):
        return DInterval(self).__ior__(b)

    def _iand_part(self,b,new,lo=0):
        if not b: return lo

        hi = len(self._parts)
        assert lo <= hi <= len(self._parts)

        while lo < hi:
            mid = (lo+hi)//2
            if self._parts[mid].end < b.start: lo = mid+1
            else: hi = mid

        assert hi == lo

        while hi < len(self._parts) and self._parts[hi].start <= b.end:
            new_start = max(self._parts[hi].start,b.start)
            new_end = min(b.end,self._parts[hi].end)
            if new_start < new_end:
                new.append(Interval(new_start,new_end))
            hi += 1

        return hi - bool(new)

    def __iand__(self,b):
        if isinstance(b,DInterval):
            new = []
            ins = 0
            for p in b._parts:
                ins = self._iand_part(p,new,ins)
                if ins == len(self._parts): break
            self._parts = new
            return self

        if isinstance(b,Interval):
            new = []
            self._iand_part(b,new)
            self._parts = new
            return self

        return NotImplemented

    def __and__(self,b):
        return DInterval(self).__iand__(b)

    def __eq__(self,b):
        if isinstance(b,DInterval):
            return len(self._parts) == len(b._parts) and all(p1 == p2 for p1,p2 in zip(self._parts,b._parts))

        if isinstance(b,Interval):
            return len(self._parts) == 1 and self._parts[0] == b

        return NotImplemented

    def __ne__(self,b):
        r = self.__eq__(b)
        if r is NotImplemented: return r
        return not r

    def __contains__(self,b):
        lo = 0
        hi = len(self._parts)

        while lo < hi:
            mid = (lo+hi)//2
            mpart = self._parts[mid]
            if mpart.end < b: lo = mid+1
            elif mpart.start > b: hi = mid
            else: return b != mpart.end

        return False

    def __iter__(self):
        return iter(self._parts)

    def closest_ge(self,b : T) -> T:
        lo = 0
        hi = len(self._parts)

        while lo < hi:
            mid = (lo+hi)//2
            mpart = self._parts[mid]
            if mpart.end < b: lo = mid+1
            elif mpart.start > b: hi = mid
            else:
                if b != mpart.end: return b
                lo = mid+1
                break

        if lo >= len(self._parts): raise ValueError('no such value exists')
        return self._parts[lo].start

    def interval_at(self,b : T) -> Interval[T]:
        lo = 0
        hi = len(self._parts)

        while lo < hi:
            mid = (lo + hi) // 2
            mpart = self._parts[mid]
            if mpart.end < b:
                lo = mid + 1
            elif mpart.start > b:
                hi = mid
            else:
                if b != mpart.end: return mpart
                break

        raise ValueError('no interval covering %s' % b)

    def __str__(self):
        return '\u2205' if not self._parts else ' \u222a '.join(map(str,self._parts))

    def __repr__(self):
        return 'DInterval([{}])'.format(''.join('({!r},{!r})'.format(*p) for p in self._parts))
