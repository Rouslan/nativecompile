import bisect
from typing import Any,Callable,Generic,Iterable,Iterator,Optional,Sequence,TypeVar

T = TypeVar('T')

class SortedList(Sequence[T],Generic[T]):
    def __init__(self,items : Iterable[T]=(),key : Optional[Callable[[T],Any]]=None) -> None:
        self._key = key
        if key is not None:
            self._data = sorted(items,key=key)
            self._keys = sorted(map(key,self._data))
        else:
            self._data = sorted(items)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self,i):
        return self._data.__getitem__(i)

    def __delitem__(self,i):
        self._data.__delitem__(i)
        if self._key:
            self._keys.__delitem__(i)

    def __contains__(self,value : object) -> bool:
        i = self.insertion_index(value)
        return i < len(self._data) and self._data[i] == value

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def insertion_index(self,value):
        if self._key is not None:
            return bisect.bisect_left(self._keys,self._key(value))

        return bisect.bisect_left(self._data,value)

    def insertion_index_by_key(self,key) -> int:
        return bisect.bisect_left(self._data if self._key is None else self._keys,key)

    def index(self,value):
        i = self.insertion_index(value)
        if i < len(self._data) and self._data[i] == value:
            return i
        raise ValueError

    def index_by_key(self,key) -> int:
        i = self.insertion_index_by_key(key)
        if i < len(self._data) and (self._data if self._key is None else self._keys)[i] == key:
            return i
        raise ValueError

    def remove(self,value : T) -> None:
        del self[self.index(value)]

    def count(self,value : T) -> int:
        i = self.insertion_index(value)
        c = 0
        while i < len(self._data) and self._data[i] == value:
            i += 1
            c += 1
        return c

    def add_item(self,value : T):
        if self._key is not None:
            k = self._key(value)
            i = bisect.bisect_right(self._keys,k)
            self._keys.insert(i,k)
            self._data.insert(i,value)
        else:
            bisect.insort_right(self._data,value)

    def clear(self):
        del self._data
        if self._key:
            del self._keys

    def __eq__(self,b):
        return isinstance(b,SortedList) and self._data == b._data

    def __ne__(self,b):
        return not self.__eq__

    def __repr__(self):
        return 'SortedList({!r})'.format(self._data)

    def __getstate__(self):
        return self._key,self._data

    def __setstate__(self,state):
        self._key,self._data = state
        if self._key:
            self._keys = list(map(self._key,self._data))

    def copy(self) -> 'SortedList[T]':
        r = SortedList.__new__(SortedList)
        r._data = self._data[:]
        r._key = self._key
        if self._key is not None:
            r._keys = self._keys[:]

        return r
