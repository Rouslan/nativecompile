# Multimethod module based on an example by Guido van Rossum
# (http://www.artima.com/weblogs/viewpost.jsp?thread=101605)

import inspect


registry = {}


# Since this implementation doesn't match derived types to base types, a class
# can define __mmtype__ to specifiy what type it is a substitute for (it's not a
# complete solution but it's enough for this package).
def mmtype(x):
    return getattr(x,'__mmtype__',x)


class MultiMethod(object):
    def __init__(self, name):
        self.name = name
        self.typemap = {}
    def __call__(self, *args,**kwds):
        if kwds:
            raise TypeError("keyword arguments not supported by multimethods")
        types = tuple(mmtype(arg.__class__) for arg in args)
        function = self.typemap.get(types)
        if function is None:
            raise TypeError("no match for {} called with ({})".format(self.name,','.join(t.__name__ for t in types)))
        return function(*args)
    def register(self, types, function):
        if types in self.typemap:
            raise TypeError("duplicate registration")
        self.typemap[types] = function

    def inherit(self,b):
        """Add all overloads from b that this multimethod doesn't already
        define."""

        for k,v in b.typemap.items():
            self.typemap.setdefault(k,v)


def multimethod(function):
    i = (id(function.__globals__),function.__name__)
    mm = registry.get(i)
    if mm is None:
        mm = registry[i] = MultiMethod(function.__name__)
    types = tuple(mmtype(function.__annotations__[arg]) for arg in inspect.getfullargspec(function)[0])
    mm.register(types, function)
    return mm


