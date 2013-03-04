

# subclasses tuple for easy immutability
class Interval(tuple):
    """A left-closed right-open interval between two points.

    For any value p, p is in Interval(a,b) if a <= p < b.

    """
    __slots__ = ()

    def __new__(cls,start,end):
        if start > end: end = start
        return tuple.__new__(cls,(start,end))

    @property
    def start(self): return self[0]

    @property
    def end(self): return self[1]

    def is_empty(self): return self.start == self.end

    def __len__(self):
        return self.end - self.start

    def __contains__(self,p):
        return self.start <= p < self.end

    def __repr__(self):
        return 'Interval({},{})'.format(*self)

    def __str__(self):
        return '∅' if self.is_empty() else '[{},{})'.format(*self)


class DInterval:
    """A discontinuous interval.

    This can be thought of as a union of left-closed right-open intervals. The
    interval doesn't actually have to be discontinuous since continuous
    intervals can be represented just as easily.

    """
    def __init__(self,x=None):
        self._parts = []
        if x is not None:
            if isinstance(x,DInterval):
                self._parts = DInterval._parts[:]
            elif isinstance(x,Interval):
                self._parts.append(x)
            else:
                raise TypeError(
                    'The parameter to DInterval should either be None or an ' +
                    'instance of DInterval or Interval')

    def _ior_part(self,b,lo=0,hi=None):
        if b.is_empty(): return -1

        if hi is None: hi = len(self._parts)
        assert lo <= hi <= len(self._parts)

        while lo < hi:
            mid = (lo+hi)//2
            if self._parts[mid].end < b.start: lo = mid+1
            else: hi = mid
            
        assert hi == lo
        while hi < len(self._parts) and self._parts[hi].start <= b.end:
            hi += 1
        if hi > lo:
            b = Interval(b.start,max(b.end,self._parts[hi-1].end))

        self._parts[lo:hi] = [b]

        return lo
                    

    def __ior__(self,b):
        if isinstance(b,DInterval):
            ins = 0
            for p in b._parts:
                ins = self._ior_part(p,ins) + 1
            return self

        if isinstance(b,Interval):
            self._ior_part(b)
            return self

        return NotImplemented

    def __or__(self,b):
        return DInterval(self).__ior__(b)

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

    def closest_ge(self,b):
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

        return self._parts[lo].start if lo < len(self._parts) else float('inf')

    def __str__(self):
        return '∅' if not self._parts else ' ∪ '.join(map(str,self._parts))
