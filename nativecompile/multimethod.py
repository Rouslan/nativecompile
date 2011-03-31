# Multimethod module based on an example by Guido van Rossum
# (http://www.artima.com/weblogs/viewpost.jsp?thread=101605)

import inspect

registry = {}

class MultiMethod(object):
    def __init__(self, name):
        self.name = name
        self.typemap = {}
    def __call__(self, *args,**kwds):
        if kwds:
            raise TypeError("keyword arguments not supported by multimethods")
        types = tuple(arg.__class__ for arg in args)
        function = self.typemap.get(types)
        if function is None:
            raise TypeError("no match")
        return function(*args)
    def register(self, types, function):
        if types in self.typemap:
            raise TypeError("duplicate registration")
        self.typemap[types] = function


def multimethod(function):
    name = function.__name__
    mm = registry.get(name)
    if mm is None:
        mm = registry[name] = MultiMethod(name)
    types = tuple(function.__annotations__[arg] for arg in inspect.getfullargspec(function)[0])
    mm.register(types, function)
    return mm


