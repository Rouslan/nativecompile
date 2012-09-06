
#include <Python.h>
#include <structmember.h>
#include <frameobject.h>

#if defined(__linux__) || defined(__linux) || defined(linux)
    #include <sys/mman.h>
    #include <unistd.h>
    #include <fcntl.h>
    
    #define USE_MMAP 1
#endif


#if defined(__x86_64__) || defined(_M_X64)
    #define ARCHITECTURE "X86_64"
#elif defined(__i386__) || defined(__i386) || defined(_M_IX86) || defined(_X86_) || defined(__THW_INTEL__) || defined(__I86__) || defined(__INTEL__)
    #define ARCHITECTURE "X86"
#elif defined(__ppc__) || defined(_M_PPC) || defined(_ARCH_PPC)
    #define ARCHITECTURE "PowerPC"
#elif defined(__ia64__) || defined(__ia64) || defined(_M_IA64)
    #define ARCHITECTURE "IA64"
#else
    #define ARCHITECTURE "unknown"
#endif


#ifdef Py_REF_DEBUG
    #define REF_DEBUG_VAL 1
#else
    #define REF_DEBUG_VAL 0
#endif

#ifdef COUNT_ALLOCS
    #define COUNT_ALLOCS_VAL 1
#else
    #define COUNT_ALLOCS_VAL 0
#endif



/* copied from Python/ceval.c */
#define NAME_ERROR_MSG \
    "name '%.200s' is not defined"
#define GLOBAL_NAME_ERROR_MSG \
    "global name '%.200s' is not defined"
#define UNBOUNDLOCAL_ERROR_MSG \
    "local variable '%.200s' referenced before assignment"
#define UNBOUNDFREE_ERROR_MSG \
    "free variable '%.200s' referenced before assignment" \
    " in enclosing scope"
#define CANNOT_CATCH_MSG "catching classes that do not inherit from "\
                         "BaseException is not allowed"

#define NO_LOCALS_LOAD_MSG "no locals when loading %R"
#define NO_LOCALS_STORE_MSG "no locals found when storing %R"
#define NO_LOCALS_DELETE_MSG "no locals when deleting %R"
#define BUILD_CLASS_ERROR_MSG "__build_class__ not found"
#define CANNOT_IMPORT_MSG "cannot import name %S"
#define IMPORT_NOT_FOUND_MSG "__import__ not found"
#define BAD_EXCEPTION_MSG "'finally' pops bad exception"


static PyObject *load_args(PyObject ***pp_stack, int na);
static void err_args(PyObject *func, int flags, int nargs);
static PyObject *fast_function(PyObject *func, PyObject ***pp_stack, int n, int na, int nk);
static PyObject *do_call(PyObject *func, PyObject ***pp_stack, int na, int nk);
static PyObject *_cc_EvalCodeEx(PyObject *_co, PyObject *globals,
    PyObject *locals, PyObject **args, int argcount, PyObject **kws,
    int kwcount, PyObject **defs, int defcount, PyObject *kwdefs,
    PyObject *closure, int indexmult);
static PyObject *_cc_EvalCode(PyObject *_co, PyObject *globals, PyObject *locals);

#define EXT_POP(STACK_POINTER) (*(STACK_POINTER)++)

#define GETLOCAL(i)     (fastlocals[i])
#define SETLOCAL(i, value)      do { PyObject *tmp = GETLOCAL(i); \
                                     GETLOCAL(i) = value; \
                                     Py_XDECREF(tmp); } while (0)



static ternaryfunc old_func_call = NULL;


typedef PyObject *(*entry_type)(PyFrameObject *);

typedef struct {
    PyObject_HEAD

    entry_type entry;

    /* a tuple of CodeObjectWithCCode objects */
    PyObject *entry_points;

#ifdef USE_MMAP
    int fd;
    size_t len;
#endif
} CompiledCode;

static PyMemberDef CompiledCoded_members[] = {
    {"entry_points",T_OBJECT_EX,offsetof(CompiledCode,entry_points),READONLY,NULL},
    {NULL}
};


static void CompiledCode_dealloc(CompiledCode *self);
static PyObject *CompiledCode_new(PyTypeObject *type,PyObject *args,PyObject *kwds);
static PyObject *CompiledCode_call(CompiledCode *self,PyObject *args,PyObject *kw);

static PyTypeObject CompiledCodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "nativecompile.pyinternals.CompiledCode", /* tp_name */
    sizeof(CompiledCode),      /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)CompiledCode_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    (ternaryfunc)CompiledCode_call, /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "Executable machine code", /* tp_doc */
    0,	                       /* tp_traverse */
    0,	                       /* tp_clear */
    0,	                       /* tp_richcompare */
    0,	                       /* tp_weaklistoffset */
    0,	                       /* tp_iter */
    0,	                       /* tp_iternext */
    0,                         /* tp_methods */
    CompiledCoded_members,     /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    CompiledCode_new,          /* tp_new */
};




#define CO_COMPILED (1 << 31)
#define HAS_CCODE(obj) (((PyCodeObject*)obj)->co_flags & CO_COMPILED && \
                        ((CodeObjectWithCCode*)obj)->compiled_code)
#define GET_CCODE_FUNC(obj) ((entry_type)(\
    (char*)(((CodeObjectWithCCode*)obj)->compiled_code->entry) + \
    ((CodeObjectWithCCode*)obj)->offset))


/* Since PyCodeObject cannot be subclassed, to support extra fields, we copy a
 * PyCodeObject to this surrogate and identify it using a bit in co_flags not
 * used by CPython */
typedef struct {
    PyObject_HEAD
    int co_argcount;
    int co_kwonlyargcount;
    int co_nlocals;
    int co_stacksize;
    int co_flags;
    PyObject *co_code;
    PyObject *co_consts;
    PyObject *co_names;
    PyObject *co_varnames;
    PyObject *co_freevars;
    PyObject *co_cellvars;
    PyObject *co_filename;
    PyObject *co_name;
    int co_firstlineno;
    PyObject *co_lnotab;
    void *co_zombieframe;
    PyObject *co_weakreflist;

    /* compiled_code is not set until this object is added to a CompiledCode
       object. Once added, it cannot be removed, so its reference count doesn't
       need to be incremented or decremented here. */
    CompiledCode *compiled_code;

    unsigned int offset;
} CodeObjectWithCCode;


static PyObject *create_compiled_entry_point(PyObject *self,PyObject *_arg) {
    CodeObjectWithCCode *ep;
    PyCodeObject *arg = (PyCodeObject*)_arg;

    if(!PyCode_Check(_arg)) {
        PyErr_SetString(PyExc_TypeError,"argument must be a code object");
        return NULL;
    }

    if(arg->co_flags & CO_COMPILED) {
        PyErr_SetString(PyExc_TypeError,"argument is already a compiled entry point");
        return NULL;
    }

    ep = PyObject_MALLOC(sizeof(CodeObjectWithCCode));
    if(ep) {
        PyObject_Init((PyObject*)ep,&PyCode_Type);
        ep->co_argcount = arg->co_argcount;
        ep->co_kwonlyargcount = arg->co_kwonlyargcount;
        ep->co_nlocals = arg->co_nlocals;
        ep->co_stacksize = arg->co_stacksize;
        ep->co_flags = arg->co_flags | CO_COMPILED;
        ep->co_code = arg->co_code;
        Py_XINCREF(ep->co_code);
        ep->co_consts = arg->co_consts;
        Py_XINCREF(ep->co_consts);
        ep->co_names = arg->co_names;
        Py_XINCREF(ep->co_names);
        ep->co_varnames = arg->co_varnames;
        Py_XINCREF(ep->co_varnames);
        ep->co_freevars = arg->co_freevars;
        Py_XINCREF(ep->co_freevars);
        ep->co_cellvars = arg->co_cellvars;
        Py_XINCREF(ep->co_cellvars);
        ep->co_filename = arg->co_filename;
        Py_XINCREF(ep->co_filename);
        ep->co_name = arg->co_name;
        Py_XINCREF(ep->co_name);
        ep->co_firstlineno = arg->co_firstlineno;
        ep->co_lnotab = arg->co_lnotab;
        Py_XINCREF(ep->co_lnotab);
        ep->co_zombieframe = NULL;
        ep->co_weakreflist = NULL;

        ep->compiled_code = NULL;
        ep->offset = 0;
    }

    return (PyObject*)ep;
}


#define CEP_CHECK(X) \
    if(!(PyCode_Check((PyObject*)X) && ((CodeObjectWithCCode*)X)->co_flags & CO_COMPILED)) { \
        PyErr_SetString(PyExc_TypeError,"argument must be a compiled entry point"); \
        return NULL; \
    }

static PyObject *cep_get_compiled_code(PyObject *self,PyObject *_arg) {
    PyObject *ret;
    CodeObjectWithCCode *arg = (CodeObjectWithCCode*)_arg;

    CEP_CHECK(arg)

    ret = arg->compiled_code ? (PyObject*)arg->compiled_code : Py_None;
    Py_INCREF(ret);
    return ret;
}

static PyObject *cep_get_offset(PyObject *self,PyObject *_arg) {
    CodeObjectWithCCode *arg = (CodeObjectWithCCode*)_arg;

    CEP_CHECK(arg)

    return PyLong_FromUnsignedLong(arg->offset);
}

static PyObject *cep_set_offset(PyObject *self,PyObject *args) {
    PyObject *cep;
    unsigned int offset;

    if(!PyArg_ParseTuple(args,"OI",&cep,&offset)) return NULL;

    CEP_CHECK(cep)

    ((CodeObjectWithCCode*)cep)->offset = offset;

    Py_RETURN_NONE;
}

static PyObject *cep_exec(PyObject *self,PyObject *args) {
    PyObject *r;
    PyObject *cep;
    PyObject *globals = Py_None;
    PyObject *locals = Py_None;

    if(!PyArg_ParseTuple(args,"O|OO",&cep,&globals,&locals)) return NULL;

    CEP_CHECK(cep)

    if (globals == Py_None) {
        globals = PyEval_GetGlobals();
        if (locals == Py_None) {
            locals = PyEval_GetLocals();
        }
        if (!globals || !locals) {
            PyErr_SetString(PyExc_SystemError,
                            "globals and locals cannot be NULL");
            return NULL;
        }
    }
    else if (locals == Py_None)
        locals = globals;

    if (!PyDict_Check(globals)) {
        PyErr_Format(PyExc_TypeError, "globals must be a dict, not %.100s",
                     globals->ob_type->tp_name);
        return NULL;
    }
    if (!PyMapping_Check(locals)) {
        PyErr_Format(PyExc_TypeError,
            "locals must be a mapping or None, not %.100s",
            locals->ob_type->tp_name);
        return NULL;
    }
    if (PyDict_GetItemString(globals, "__builtins__") == NULL) {
        if (PyDict_SetItemString(globals, "__builtins__",
                                 PyEval_GetBuiltins()) != 0)
            return NULL;
    }

    if (PyCode_GetNumFree((PyCodeObject *)cep) > 0) {
        PyErr_SetString(PyExc_TypeError,
                        "code object may not contain free variables");
        return NULL;
    }

    r = _cc_EvalCode(cep,globals,locals);
    if(r) {
        Py_DECREF(r);
        Py_RETURN_NONE;
    }
    return NULL;
}



static void CompiledCode_dealloc(CompiledCode *self) {
    int i;
    PyObject *ep;

    if(self->entry_points) {
        for(i=0; i<PyTuple_GET_SIZE(self->entry_points); ++i) {
            ep = PyTuple_GET_ITEM(self->entry_points,i);
            ((CodeObjectWithCCode*)ep)->compiled_code = NULL;
        }
        Py_DECREF(self->entry_points);
    }
    if(self->entry) {
#ifdef USE_MMAP
        munmap(self->entry,self->len);
        close(self->fd);
#else
        PyMem_Free(self->entry);
#endif
    }
}

static PyObject *CompiledCode_new(PyTypeObject *type,PyObject *args,PyObject *kwds) {
    CompiledCode *self;
    PyObject *filename_o;
    const char *filename_s;
    PyObject *entry_points;
    int i;
    PyObject *item;

#ifdef USE_MMAP
    off_t slen;
    void *mem;
#else
    FILE *f;
    long len;
    size_t read;
#endif

    static char *kwlist[] = {"filename","entry_points",NULL};

    if(!PyArg_ParseTupleAndKeywords(args,kwds,"O&O",kwlist,
        PyUnicode_FSConverter,
        &filename_o,
        &entry_points)) return NULL;

    filename_s = PyBytes_AS_STRING(filename_o);
    
    self = (CompiledCode*)type->tp_alloc(type,0);
    if(self) {
        self->entry = NULL;

        self->entry_points = PyObject_CallFunctionObjArgs(
            (PyObject*)&PyTuple_Type,
            entry_points,NULL);
        if(!self->entry_points) goto error;

        i = PyTuple_Size(self->entry_points);
        if(PyErr_Occurred()) goto error;
        if(i < 1) {
            PyErr_SetString(PyExc_ValueError,"entry_points needs to have at least one entry");
            goto error;
        }

        while(--i >= 0) {
            item = PyTuple_GET_ITEM(self->entry_points,i);

            if(!(PyCode_Check(item) && ((PyCodeObject*)item)->co_flags & CO_COMPILED)) {
                PyErr_SetString(PyExc_TypeError,"an item in entry_points is not a compiled entry point");
                goto error;
            }
            if(((CodeObjectWithCCode*)item)->compiled_code) {
                PyErr_SetString(PyExc_TypeError,"an item in entry_points is already part of another CompiledCode object");
                goto error;
            }

            /* the reference count is deliberately not increased (see
               CodeObjectWithCCode) */
            ((CodeObjectWithCCode*)item)->compiled_code = self;
        }
        
#ifdef USE_MMAP
        /* use mmap to create an executable region of memory (merely using
           mprotect is not enough because some security-conscious systems don't
           allow marking allocated memory executable unless it was already
           executable) */

        if((self->fd = open(filename_s,O_RDONLY)) == -1) goto io_error;
        
        /* get the file length */
        if((slen = lseek(self->fd,0,SEEK_END)) == -1 || lseek(self->fd,0,SEEK_SET) == -1) {
            close(self->fd);
            goto io_error;
        }
        self->len = (size_t)slen;
        
        if((mem = mmap(0,self->len,PROT_READ|PROT_EXEC,MAP_PRIVATE,self->fd,0)) == MAP_FAILED) {
            close(self->fd);
            PyErr_SetFromErrno(PyExc_OSError);
            goto error;
        }
        self->entry = (entry_type)mem;

#else
        /* just load the file contents into memory */

        if((f = fopen(filename_s,"rb")) == NULL) goto io_error;
        
        /* get the file length */
        if(fseek(f,0,SEEK_END) || (len = ftell(f)) == -1 || fseek(f,0,SEEK_SET)) {
            fclose(f);
            goto io_error;
        }
        
        if((self->entry = (entry_type)PyMem_Malloc(len)) == NULL) {
            fclose(f);
            goto error;
        }
        
        read = fread(self->entry,1,len,f);
        fclose(f);
        if(read < (size_t)len) goto io_error;

#endif
        goto end; /* skip over the error handling code */
        
    io_error:
        PyErr_SetFromErrnoWithFilename(PyExc_IOError,filename_s);
    error:
        Py_DECREF(self);
        self = NULL;
    }
end:
    
    Py_DECREF(filename_o);
    
    return (PyObject*)self;
}

static PyObject *CompiledCode_call(CompiledCode *self,PyObject *args,PyObject *kw) {
    PyObject *r;

    assert(PyTuple_GET_SIZE(self->entry_points) > 0);

    r = _cc_EvalCode(
        PyTuple_GET_ITEM(self->entry_points,0),
        PyEval_GetGlobals(),
        PyEval_GetLocals());
    
    if(r) {
        Py_DECREF(r);
        Py_RETURN_NONE;
    }
    return NULL;
}




/* The following are modified versions of functions in Python/ceval.c */

static PyObject *
call_function(PyObject **pp_stack, int oparg)
{
    int na = oparg & 0xff;
    int nk = (oparg>>8) & 0xff;
    int n = na + 2 * nk;
    PyObject **pfunc = pp_stack + n;
    PyObject *func = *pfunc;
    PyObject *x, *w;

    if (PyCFunction_Check(func) && nk == 0) {
        int flags = PyCFunction_GET_FLAGS(func);

        if (flags & (METH_NOARGS | METH_O)) {
            PyCFunction meth = PyCFunction_GET_FUNCTION(func);
            PyObject *self = PyCFunction_GET_SELF(func);
            if (flags & METH_NOARGS && na == 0) {
                x = (*meth)(self,NULL);
            }
            else if (flags & METH_O && na == 1) {
                PyObject *arg = EXT_POP(pp_stack);
                x = (*meth)(self,arg);
                Py_DECREF(arg);
            }
            else {
                err_args(func, flags, na);
                x = NULL;
            }
        }
        else {
            PyObject *callargs;
            callargs = load_args(&pp_stack, na);
            x = PyCFunction_Call(func,callargs,NULL);
            Py_XDECREF(callargs);
        }
    } else {
        if (PyMethod_Check(func) && PyMethod_GET_SELF(func) != NULL) {
            PyObject *self = PyMethod_GET_SELF(func);
            Py_INCREF(self);
            func = PyMethod_GET_FUNCTION(func);
            Py_INCREF(func);
            Py_DECREF(*pfunc);
            *pfunc = self;
            na++;
            n++;
        } else
            Py_INCREF(func);
        if (PyFunction_Check(func))
            x = fast_function(func, &pp_stack, n, na, nk);
        else
            x = do_call(func, &pp_stack, na, nk);
        Py_DECREF(func);
    }

    while (pp_stack <= pfunc) {
        w = EXT_POP(pp_stack);
        Py_DECREF(w);
    }
    return x;
}

static PyObject *
fast_function(PyObject *func, PyObject ***pp_stack, int n, int na, int nk)
{
    PyCodeObject *co = (PyCodeObject *)PyFunction_GET_CODE(func);
    PyObject *globals = PyFunction_GET_GLOBALS(func);
    PyObject *argdefs = PyFunction_GET_DEFAULTS(func);
    PyObject *kwdefs = PyFunction_GET_KW_DEFAULTS(func);
    PyObject **d = NULL;
    int nd = 0;

    if (argdefs == NULL && co->co_argcount == n &&
        co->co_kwonlyargcount == 0 && nk==0 &&
        co->co_flags == (CO_OPTIMIZED | CO_NEWLOCALS | CO_NOFREE)) {
        PyFrameObject *f;
        PyObject *retval = NULL;
        PyThreadState *tstate = PyThreadState_GET();
        PyObject **fastlocals, **stack;
        int i;

        assert(globals != NULL);
        assert(tstate != NULL);
        f = PyFrame_New(tstate, co, globals, NULL);
        if (f == NULL)
            return NULL;

        fastlocals = f->f_localsplus;
        stack = (*pp_stack) + n - 1;

        for (i = 0; i < n; i++) {
            Py_INCREF(*stack);
            fastlocals[i] = *stack--;
        }

        retval = HAS_CCODE(co) ? GET_CCODE_FUNC(co)(f) : PyEval_EvalFrameEx(f,0);

        ++tstate->recursion_depth;
        Py_DECREF(f);
        --tstate->recursion_depth;
        return retval;
    }
    if (argdefs != NULL) {
        d = &PyTuple_GET_ITEM(argdefs, 0);
        nd = Py_SIZE(argdefs);
    }


    return _cc_EvalCodeEx((PyObject*)co, globals,
                          (PyObject *)NULL, (*pp_stack)+n-1, na,
                          (*pp_stack)+2*nk-1, nk, d, nd, kwdefs,
                          PyFunction_GET_CLOSURE(func),-1);
}

static PyObject *
update_keyword_args(PyObject *orig_kwdict, int nk, PyObject ***pp_stack,
                    PyObject *func)
{
    PyObject *kwdict = NULL;
    if (orig_kwdict == NULL)
        kwdict = PyDict_New();
    else {
        kwdict = PyDict_Copy(orig_kwdict);
        Py_DECREF(orig_kwdict);
    }
    if (kwdict == NULL)
        return NULL;
    while (--nk >= 0) {
        int err;
        PyObject *value = EXT_POP(*pp_stack);
        PyObject *key = EXT_POP(*pp_stack);
        if (PyDict_GetItem(kwdict, key) != NULL) {
            PyErr_Format(PyExc_TypeError,
                         "%.200s%s got multiple values "
                         "for keyword argument '%U'",
                         PyEval_GetFuncName(func),
                         PyEval_GetFuncDesc(func),
                         key);
            Py_DECREF(key);
            Py_DECREF(value);
            Py_DECREF(kwdict);
            return NULL;
        }
        err = PyDict_SetItem(kwdict, key, value);
        Py_DECREF(key);
        Py_DECREF(value);
        if (err) {
            Py_DECREF(kwdict);
            return NULL;
        }
    }
    return kwdict;
}

static PyObject *
load_args(PyObject ***pp_stack, int na)
{
    PyObject *args = PyTuple_New(na);
    PyObject *w;

    if (args == NULL)
        return NULL;
    while (--na >= 0) {
        w = EXT_POP(*pp_stack);
        PyTuple_SET_ITEM(args, na, w);
    }
    return args;
}

static PyObject *
do_call(PyObject *func, PyObject ***pp_stack, int na, int nk)
{
    PyObject *callargs = NULL;
    PyObject *kwdict = NULL;
    PyObject *result = NULL;

    if (nk > 0) {
        kwdict = update_keyword_args(NULL, nk, pp_stack, func);
        if (kwdict == NULL)
            goto call_fail;
    }
    callargs = load_args(pp_stack, na);
    if (callargs == NULL)
        goto call_fail;

    if (PyCFunction_Check(func))
        result = PyCFunction_Call(func, callargs, kwdict);
    else
        result = PyObject_Call(func, callargs, kwdict);
call_fail:
    Py_XDECREF(callargs);
    Py_XDECREF(kwdict);
    return result;
}

static void
err_args(PyObject *func, int flags, int nargs)
{
    if (flags & METH_NOARGS)
        PyErr_Format(PyExc_TypeError,
                     "%.200s() takes no arguments (%d given)",
                     ((PyCFunctionObject *)func)->m_ml->ml_name,
                     nargs);
    else
        PyErr_Format(PyExc_TypeError,
                     "%.200s() takes exactly one argument (%d given)",
                     ((PyCFunctionObject *)func)->m_ml->ml_name,
                     nargs);
}

static void
format_exc_check_arg(PyObject *exc, const char *format_str, PyObject *obj)
{
    const char *obj_str;

    if (!obj)
        return;

    obj_str = _PyUnicode_AsString(obj);
    if (!obj_str)
        return;

    PyErr_Format(exc, format_str, obj_str);
}

/* we store stack values in reverse order compared to how CPython stores them so
   this function is given an extra parameter to multiply the index by, which can
   be 1 or -1 */
static PyObject *
_cc_EvalCodeEx(PyObject *_co, PyObject *globals, PyObject *locals,
           PyObject **args, int argcount, PyObject **kws, int kwcount,
           PyObject **defs, int defcount, PyObject *kwdefs, PyObject *closure,
           int indexmult)
{
    PyCodeObject* co = (PyCodeObject*)_co;
    register PyFrameObject *f;
    register PyObject *retval = NULL;
    register PyObject **fastlocals, **freevars;
    PyThreadState *tstate = PyThreadState_GET();
    PyObject *x, *u;
    int total_args = co->co_argcount + co->co_kwonlyargcount;

    if (globals == NULL) {
        PyErr_SetString(PyExc_SystemError,
                        "PyEval_EvalCodeEx: NULL globals");
        return NULL;
    }

    assert(tstate != NULL);
    assert(globals != NULL);
    f = PyFrame_New(tstate, co, globals, locals);
    if (f == NULL)
        return NULL;

    fastlocals = f->f_localsplus;
    freevars = f->f_localsplus + co->co_nlocals;

    if (total_args || co->co_flags & (CO_VARARGS | CO_VARKEYWORDS)) {
        int i;
        int n = argcount;
        PyObject *kwdict = NULL;
        if (co->co_flags & CO_VARKEYWORDS) {
            kwdict = PyDict_New();
            if (kwdict == NULL)
                goto fail;
            i = total_args;
            if (co->co_flags & CO_VARARGS)
                i++;
            SETLOCAL(i, kwdict);
        }
        if (argcount > co->co_argcount) {
            if (!(co->co_flags & CO_VARARGS)) {
                PyErr_Format(PyExc_TypeError,
                    "%U() takes %s %d "
                    "positional argument%s (%d given)",
                    co->co_name,
                    defcount ? "at most" : "exactly",
                    co->co_argcount,
                    co->co_argcount == 1 ? "" : "s",
                    argcount + kwcount);
                goto fail;
            }
            n = co->co_argcount;
        }
        for (i = 0; i < n; i++) {
            x = args[i * indexmult];
            Py_INCREF(x);
            SETLOCAL(i, x);
        }
        if (co->co_flags & CO_VARARGS) {
            u = PyTuple_New(argcount - n);
            if (u == NULL)
                goto fail;
            SETLOCAL(total_args, u);
            for (i = n; i < argcount; i++) {
                x = args[i * indexmult];
                Py_INCREF(x);
                PyTuple_SET_ITEM(u, i-n, x);
            }
        }
        for (i = 0; i < kwcount; i++) {
            PyObject **co_varnames;
            PyObject *keyword = kws[2*i*indexmult];
            PyObject *value = kws[(2*i + 1) * indexmult];
            int j;
            if (keyword == NULL || !PyUnicode_Check(keyword)) {
                PyErr_Format(PyExc_TypeError,
                    "%U() keywords must be strings",
                    co->co_name);
                goto fail;
            }
            co_varnames = ((PyTupleObject *)(co->co_varnames))->ob_item;
            for (j = 0; j < total_args; j++) {
                PyObject *nm = co_varnames[j];
                if (nm == keyword)
                    goto kw_found;
            }
            for (j = 0; j < total_args; j++) {
                PyObject *nm = co_varnames[j];
                int cmp = PyObject_RichCompareBool(
                    keyword, nm, Py_EQ);
                if (cmp > 0)
                    goto kw_found;
                else if (cmp < 0)
                    goto fail;
            }
            if (j >= total_args && kwdict == NULL) {
                PyErr_Format(PyExc_TypeError,
                             "%U() got an unexpected "
                             "keyword argument '%S'",
                             co->co_name,
                             keyword);
                goto fail;
            }
            PyDict_SetItem(kwdict, keyword, value);
            continue;
          kw_found:
            if (GETLOCAL(j) != NULL) {
                PyErr_Format(PyExc_TypeError,
                         "%U() got multiple "
                         "values for keyword "
                         "argument '%S'",
                         co->co_name,
                         keyword);
                goto fail;
            }
            Py_INCREF(value);
            SETLOCAL(j, value);
        }
        if (co->co_kwonlyargcount > 0) {
            for (i = co->co_argcount; i < total_args; i++) {
                PyObject *name;
                if (GETLOCAL(i) != NULL)
                    continue;
                name = PyTuple_GET_ITEM(co->co_varnames, i);
                if (kwdefs != NULL) {
                    PyObject *def = PyDict_GetItem(kwdefs, name);
                    if (def) {
                        Py_INCREF(def);
                        SETLOCAL(i, def);
                        continue;
                    }
                }
                PyErr_Format(PyExc_TypeError,
                    "%U() needs keyword-only argument %S",
                    co->co_name, name);
                goto fail;
            }
        }
        if (argcount < co->co_argcount) {
            int m = co->co_argcount - defcount;
            for (i = argcount; i < m; i++) {
                if (GETLOCAL(i) == NULL) {
                    int j, given = 0;
                    for (j = 0; j < co->co_argcount; j++)
                        if (GETLOCAL(j))
                            given++;
                    PyErr_Format(PyExc_TypeError,
                        "%U() takes %s %d "
                        "argument%s "
                        "(%d given)",
                        co->co_name,
                        ((co->co_flags & CO_VARARGS) ||
                         defcount) ? "at least"
                                   : "exactly",
                             m, m == 1 ? "" : "s", given);
                    goto fail;
                }
            }
            if (n > m)
                i = n - m;
            else
                i = 0;
            for (; i < defcount; i++) {
                if (GETLOCAL(m+i) == NULL) {
                    PyObject *def = defs[i];
                    Py_INCREF(def);
                    SETLOCAL(m+i, def);
                }
            }
        }
    }
    else if (argcount > 0 || kwcount > 0) {
        PyErr_Format(PyExc_TypeError,
                     "%U() takes no arguments (%d given)",
                     co->co_name,
                     argcount + kwcount);
        goto fail;
    }
    if (PyTuple_GET_SIZE(co->co_cellvars)) {
        int i, j, nargs, found;
        Py_UNICODE *cellname, *argname;
        PyObject *c;

        nargs = total_args;
        if (co->co_flags & CO_VARARGS)
            nargs++;
        if (co->co_flags & CO_VARKEYWORDS)
            nargs++;

        for (i = 0; i < PyTuple_GET_SIZE(co->co_cellvars); ++i) {
            cellname = PyUnicode_AS_UNICODE(
                PyTuple_GET_ITEM(co->co_cellvars, i));
            found = 0;
            for (j = 0; j < nargs; j++) {
                argname = PyUnicode_AS_UNICODE(
                    PyTuple_GET_ITEM(co->co_varnames, j));
                if (Py_UNICODE_strcmp(cellname, argname) == 0) {
                    c = PyCell_New(GETLOCAL(j));
                    if (c == NULL)
                        goto fail;
                    GETLOCAL(co->co_nlocals + i) = c;
                    found = 1;
                    break;
                }
            }
            if (found == 0) {
                c = PyCell_New(NULL);
                if (c == NULL)
                    goto fail;
                SETLOCAL(co->co_nlocals + i, c);
            }
        }
    }
    if (PyTuple_GET_SIZE(co->co_freevars)) {
        int i;
        for (i = 0; i < PyTuple_GET_SIZE(co->co_freevars); ++i) {
            PyObject *o = PyTuple_GET_ITEM(closure, i);
            Py_INCREF(o);
            freevars[PyTuple_GET_SIZE(co->co_cellvars) + i] = o;
        }
    }

    if (co->co_flags & CO_GENERATOR) {
        Py_XDECREF(f->f_back);
        f->f_back = NULL;

        return PyGen_New(f);
    }

    retval = HAS_CCODE(co) ? GET_CCODE_FUNC(co)(f) : PyEval_EvalFrameEx(f,0);

fail:

    assert(tstate != NULL);
    ++tstate->recursion_depth;
    Py_DECREF(f);
    --tstate->recursion_depth;
    return retval;
}

static PyObject *
_cc_EvalCode(PyObject *_co, PyObject *globals, PyObject *locals) {
    return _cc_EvalCodeEx(_co,globals,locals,NULL,0,NULL,0,NULL,0,NULL,NULL,1);
}



#define POP() (*items++)

static PyObject *_make_function(int makeclosure,unsigned int arg,PyObject **items) {
    PyObject *val;
    PyObject *key;
    PyObject *func;
    int r;
    int posdefaults = arg & 0xff;
    int kwdefaults = (arg>>8) & 0xff;
    int num_annotations = (arg >> 16) & 0x7fff;


    val = POP();
    func = PyFunction_New(val, PyEval_GetGlobals());
    Py_DECREF(val);

    if(!func) goto fail_closure;


    if(makeclosure) {
        val = POP();
        r = PyFunction_SetClosure(func, val);
        Py_DECREF(val);
        if(r) goto fail_before_annot;
    }

    if(num_annotations > 0) {
        PyObject *names = POP();
        PyObject *annot = PyDict_New();
        if(!annot) {
            Py_DECREF(names);
            goto fail_in_annot;
        }

        while(--num_annotations >= 0) {
            val = POP();
            r = PyDict_SetItem(annot, PyTuple_GET_ITEM(names, num_annotations), val);
            Py_DECREF(val);

            if(r) {
                Py_DECREF(annot);
                Py_DECREF(names);
                goto fail_in_annot;
            }
        }

        r = PyFunction_SetAnnotations(func, annot);
        Py_DECREF(annot);
        Py_DECREF(names);
        if(r) goto fail_posdef;
    }

    if(posdefaults > 0) {
        PyObject *defs = PyTuple_New(posdefaults);
        if(!defs) goto fail_posdef;
        while(--posdefaults >= 0) {
            val = POP();
            PyTuple_SET_ITEM(defs, posdefaults, val);
        }
        r = PyFunction_SetDefaults(func, defs);
        Py_DECREF(defs);
        if(r) goto fail_kwdef;
    }
    if(kwdefaults > 0) {
        PyObject *defs = PyDict_New();
        if(!defs) goto fail_kwdef;
        while(--kwdefaults >= 0) {
            val = POP();
            key = POP();
            r = PyDict_SetItem(defs, key, val);
            Py_DECREF(val);
            Py_DECREF(key);

            if(r) {
                Py_DECREF(defs);
                goto fail_kwdef;
            }
        }
        r = PyFunction_SetKwDefaults(func, defs);
        Py_DECREF(defs);
        if(r) goto fail_end;
    }

    return func;

fail_closure:
    if(makeclosure) {
        val = POP();
        Py_DECREF(val);
    }
fail_before_annot:
    if(num_annotations) {
        val = POP();
        Py_DECREF(val);
    fail_in_annot:
        while(--num_annotations >= 0) {
            val = POP();
            Py_DECREF(val);
        }
    }
fail_posdef:
    while(--posdefaults >= 0) {
        val = POP();
        Py_DECREF(val);
    }
fail_kwdef:
    while(--kwdefaults >= 0) {
        val = POP();
        Py_DECREF(val);
        val = POP();
        Py_DECREF(val);
    }
fail_end:
    Py_XDECREF(func);
    return NULL;
}


static int
_unpack_iterable(PyObject *v, int argcnt, int argcntafter, PyObject **sp)
{
    int i = 0, j = 0;
    Py_ssize_t ll = 0;
    PyObject *it;
    PyObject *w;
    PyObject *l = NULL;

    assert(v != NULL);

    it = PyObject_GetIter(v);
    if (it == NULL)
        goto Error;

    for (; i < argcnt; i++) {
        w = PyIter_Next(it);
        if (w == NULL) {
            if (!PyErr_Occurred()) {
                PyErr_Format(PyExc_ValueError,
                    "need more than %d value%s to unpack",
                    i, i == 1 ? "" : "s");
            }
            goto Error;
        }
        *sp++ = w;
    }

    if (argcntafter == -1) {
        w = PyIter_Next(it);
        if (w == NULL) {
            if (PyErr_Occurred())
                goto Error;
            Py_DECREF(it);
            return 1;
        }
        Py_DECREF(w);
        PyErr_Format(PyExc_ValueError, "too many values to unpack "
                     "(expected %d)", argcnt);
        goto Error;
    }

    l = PySequence_List(it);
    if (l == NULL)
        goto Error;
    *sp++ = l;
    i++;

    ll = PyList_GET_SIZE(l);
    if (ll < argcntafter) {
        PyErr_Format(PyExc_ValueError, "need more than %zd values to unpack",
                     argcnt + ll);
        goto Error;
    }

    for (j = argcntafter; j > 0; j--, i++) {
        *sp++ = PyList_GET_ITEM(l, ll - j);
    }

    Py_SIZE(l) = ll - argcntafter;
    Py_DECREF(it);
    return 1;

Error:
    for (sp--; i > 0; i--, sp--)
        Py_DECREF(*sp);
    Py_XDECREF(it);
    return 0;
}



static PyObject *_exception_cmp(PyObject *exc,PyObject *type) {
    PyObject *ret;

    if (PyTuple_Check(type)) {
        Py_ssize_t i, length;
        length = PyTuple_Size(type);
        for (i = 0; i < length; i += 1) {
            PyObject *exc = PyTuple_GET_ITEM(type, i);
            if (!PyExceptionClass_Check(exc)) {
                PyErr_SetString(PyExc_TypeError,
                                CANNOT_CATCH_MSG);
                return NULL;
            }
        }
    }
    else {
        if (!PyExceptionClass_Check(type)) {
            PyErr_SetString(PyExc_TypeError,
                            CANNOT_CATCH_MSG);
            return NULL;
        }
    }
    ret = PyErr_GivenExceptionMatches(exc, type) ? Py_True : Py_False;
    Py_INCREF(ret);
    return ret;
}


static PyObject*
_do_raise(PyObject *exc, PyObject *cause)
{
    PyObject *type = NULL, *value = NULL;

    if (exc == NULL) {
        PyThreadState *tstate = PyThreadState_GET();
        PyObject *tb;
        type = tstate->exc_type;
        value = tstate->exc_value;
        tb = tstate->exc_traceback;
        if (type == Py_None) {
            PyErr_SetString(PyExc_RuntimeError,
                            "No active exception to reraise");
            return NULL;
            }
        Py_XINCREF(type);
        Py_XINCREF(value);
        Py_XINCREF(tb);
        PyErr_Restore(type, value, tb);
        return NULL;
    }

    if (PyExceptionClass_Check(exc)) {
        type = exc;
        value = PyObject_CallObject(exc, NULL);
        if (value == NULL)
            goto raise_error;
    }
    else if (PyExceptionInstance_Check(exc)) {
        value = exc;
        type = PyExceptionInstance_Class(exc);
        Py_INCREF(type);
    }
    else {
        Py_DECREF(exc);
        PyErr_SetString(PyExc_TypeError,
                        "exceptions must derive from BaseException");
        goto raise_error;
    }

    if (cause) {
        PyObject *fixed_cause;
        if (PyExceptionClass_Check(cause)) {
            fixed_cause = PyObject_CallObject(cause, NULL);
            if (fixed_cause == NULL)
                goto raise_error;
            Py_DECREF(cause);
        }
        else if (PyExceptionInstance_Check(cause)) {
            fixed_cause = cause;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "exception causes must derive from "
                            "BaseException");
            goto raise_error;
        }
        PyException_SetCause(value, fixed_cause);
    }

    PyErr_SetObject(type, value);
    Py_XDECREF(value);
    Py_XDECREF(type);
    return NULL;

raise_error:
    Py_XDECREF(value);
    Py_XDECREF(type);
    Py_XDECREF(cause);
    return NULL;
}

static int import_all_from(PyFrameObject *f, PyObject *v)
{
    PyObject *locals;
    PyObject *all;
    PyObject *dict, *name, *value;
    int skip_leading_underscores = 0;
    int pos;
    int err = -1;

    PyFrame_FastToLocals(f);
    if ((locals = f->f_locals) == NULL) {
        PyErr_SetString(PyExc_SystemError,
                        "no locals found during 'import *'");
        goto all_err;
    }

    all = PyObject_GetAttrString(v, "__all__");

    if (all == NULL) {
        if (!PyErr_ExceptionMatches(PyExc_AttributeError))
            goto all_err;
        PyErr_Clear();
        dict = PyObject_GetAttrString(v, "__dict__");
        if (dict == NULL) {
            if (!PyErr_ExceptionMatches(PyExc_AttributeError))
                goto all_err;
            PyErr_SetString(PyExc_ImportError,
            "from-import-* object has no __dict__ and no __all__");
            goto all_err;
        }
        all = PyMapping_Keys(dict);
        Py_DECREF(dict);
        if (all == NULL)
            goto all_err;
        skip_leading_underscores = 1;
    }

    for (pos = 0, err = 0; ; pos++) {
        name = PySequence_GetItem(all, pos);
        if (name == NULL) {
            if (!PyErr_ExceptionMatches(PyExc_IndexError))
                err = -1;
            else
                PyErr_Clear();
            break;
        }
        if (skip_leading_underscores &&
            PyUnicode_Check(name) &&
            PyUnicode_AS_UNICODE(name)[0] == '_')
        {
            Py_DECREF(name);
            continue;
        }
        value = PyObject_GetAttr(v, name);
        if (value == NULL)
            err = -1;
        else if (PyDict_CheckExact(locals))
            err = PyDict_SetItem(locals, name, value);
        else
            err = PyObject_SetItem(locals, name, value);
        Py_DECREF(name);
        Py_XDECREF(value);
        if (err != 0)
            break;
    }
    Py_DECREF(all);
all_err:
    PyFrame_LocalsToFast(f, 0);
    Py_DECREF(v);

    return err;
}

static PyObject *
_function_call(PyObject *func, PyObject *arg, PyObject *kw)
{
    PyObject *result;
    PyObject *argdefs;
    PyObject *kwtuple = NULL;
    PyObject **d, **k;
    Py_ssize_t nk, nd;

    argdefs = PyFunction_GET_DEFAULTS(func);
    if (argdefs != NULL && PyTuple_Check(argdefs)) {
        d = &PyTuple_GET_ITEM((PyTupleObject *)argdefs, 0);
        nd = PyTuple_GET_SIZE(argdefs);
    }
    else {
        d = NULL;
        nd = 0;
    }

    if (kw != NULL && PyDict_Check(kw)) {
        Py_ssize_t pos, i;
        nk = PyDict_Size(kw);
        kwtuple = PyTuple_New(2*nk);
        if (kwtuple == NULL)
            return NULL;
        k = &PyTuple_GET_ITEM(kwtuple, 0);
        pos = i = 0;
        while (PyDict_Next(kw, &pos, &k[i], &k[i+1])) {
            Py_INCREF(k[i]);
            Py_INCREF(k[i+1]);
            i += 2;
        }
        nk = i/2;
    }
    else {
        k = NULL;
        nk = 0;
    }

    result = _cc_EvalCodeEx(
        PyFunction_GET_CODE(func),
        PyFunction_GET_GLOBALS(func), (PyObject *)NULL,
        &PyTuple_GET_ITEM(arg, 0), PyTuple_GET_SIZE(arg),
        k, nk, d, nd,
        PyFunction_GET_KW_DEFAULTS(func),
        PyFunction_GET_CLOSURE(func),
        1);

    Py_XDECREF(kwtuple);

    return result;
}

static PyObject *
special_lookup(PyObject *o, char *meth, PyObject **cache)
{
    PyObject *res;
    res = _PyObject_LookupSpecial(o, meth, cache);
    if (res == NULL && !PyErr_Occurred()) {
        PyErr_SetObject(PyExc_AttributeError, *cache);
        return NULL;
    }
    return res;
}

static PyObject *exit_cache = NULL;
static PyObject *enter_cache = NULL;



/* Py_EnterRecursiveCall and Py_LeaveRecursiveCall are somewhat complicated
 * macros so they are wrapped in the following two functions */

static int _EnterRecursiveCall(char *where) {
    return Py_EnterRecursiveCall(where);
}

static void _LeaveRecursiveCall(void) {
    Py_LeaveRecursiveCall();
}


static PyMethodDef functions[] = {
    {"create_compiled_entry_point",create_compiled_entry_point,METH_O,NULL},
    {"cep_get_compiled_code",cep_get_compiled_code,METH_O,NULL},
    {"cep_get_offset",cep_get_offset,METH_O,NULL},
    {"cep_set_offset",cep_set_offset,METH_VARARGS,NULL},
    {"cep_exec",cep_exec,METH_VARARGS,NULL},
    {NULL}
};

static void free_module(void *_) {
    assert(old_func_call);
    PyFunction_Type.tp_call = old_func_call;
}

static struct PyModuleDef this_module = {
    PyModuleDef_HEAD_INIT,
    "pyinternals",
    NULL,
    -1,
    functions,
    NULL,
    NULL,
    NULL,
    free_module
};


#define ADD_ADDR(item) {#item,(unsigned long)item}
#define ADD_ADDR_OF(item) {#item,(unsigned long)(&item)}
#define ADD_ADDR_STR(item) {item,(unsigned long)item}

#define ADD_INT_OFFSET(name,type,member) \
    if(PyModule_AddIntConstant(m,name,offsetof(type,member)) == -1) return NULL

typedef struct {
    const char *name;
    unsigned long addr;
} AddrRec;

PyMODINIT_FUNC
PyInit_pyinternals(void) {
    PyObject *m;
    PyObject *addrs;
    PyObject *tmp;
    int ret, i;
    
    if(PyType_Ready(&CompiledCodeType) < 0) return NULL;

    m = PyModule_Create(&this_module);
    if(!m) return NULL;

    ADD_INT_OFFSET("REFCNT_OFFSET",PyObject,ob_refcnt);
    ADD_INT_OFFSET("TYPE_OFFSET",PyObject,ob_type);
    ADD_INT_OFFSET("VAR_SIZE_OFFSET",PyVarObject,ob_size);
    ADD_INT_OFFSET("TYPE_DEALLOC_OFFSET",PyTypeObject,tp_dealloc);
    ADD_INT_OFFSET("TYPE_ITERNEXT_OFFSET",PyTypeObject,tp_iternext);
    ADD_INT_OFFSET("TYPE_FLAGS_OFFSET",PyTypeObject,tp_flags);
    ADD_INT_OFFSET("LIST_ITEM_OFFSET",PyListObject,ob_item);
    ADD_INT_OFFSET("TUPLE_ITEM_OFFSET",PyTupleObject,ob_item);
    ADD_INT_OFFSET("FRAME_BACK_OFFSET",PyFrameObject,f_back);
    ADD_INT_OFFSET("FRAME_BUILTINS_OFFSET",PyFrameObject,f_builtins);
    ADD_INT_OFFSET("FRAME_GLOBALS_OFFSET",PyFrameObject,f_globals);
    ADD_INT_OFFSET("FRAME_LOCALS_OFFSET",PyFrameObject,f_locals);
    ADD_INT_OFFSET("FRAME_LOCALSPLUS_OFFSET",PyFrameObject,f_localsplus);
    ADD_INT_OFFSET("THREADSTATE_FRAME_OFFSET",PyThreadState,frame);
    ADD_INT_OFFSET("THREADSTATE_TRACEBACK_OFFSET",PyThreadState,exc_traceback);
    ADD_INT_OFFSET("THREADSTATE_VALUE_OFFSET",PyThreadState,exc_value);
    ADD_INT_OFFSET("THREADSTATE_TYPE_OFFSET",PyThreadState,exc_type);
    if(PyModule_AddStringConstant(m,"ARCHITECTURE",ARCHITECTURE) == -1) return NULL;
    if(PyModule_AddObject(m,"REF_DEBUG",PyBool_FromLong(REF_DEBUG_VAL)) == -1) return NULL;
    if(PyModule_AddObject(m,"COUNT_ALLOCS",PyBool_FromLong(COUNT_ALLOCS_VAL)) == -1) return NULL;

    
    addrs = PyDict_New();
    if(!addrs) return NULL;
    
    if(PyModule_AddObject(m,"raw_addresses",addrs) == -1) return NULL;

    AddrRec addr_records[] = {
        ADD_ADDR(Py_IncRef),
        ADD_ADDR(Py_DecRef),
        ADD_ADDR(PyDict_GetItem),
        ADD_ADDR(PyDict_SetItem),
        ADD_ADDR(PyDict_GetItemString),
        ADD_ADDR(_PyDict_NewPresized),
        ADD_ADDR(PyObject_GetItem),
        ADD_ADDR(PyObject_SetItem),
        ADD_ADDR(PyObject_DelItem),
        ADD_ADDR(PyObject_GetIter),
        ADD_ADDR(PyObject_GetAttr),
        ADD_ADDR(PyObject_SetAttr),
        ADD_ADDR(PyObject_IsTrue),
        ADD_ADDR(PyObject_RichCompare),
        ADD_ADDR(PyObject_Call),
        ADD_ADDR(PyObject_CallFunctionObjArgs),
        ADD_ADDR(PyEval_GetGlobals),
        ADD_ADDR(PyEval_GetBuiltins),
        ADD_ADDR(PyEval_GetLocals),
        ADD_ADDR(PyErr_Occurred),
        ADD_ADDR(PyErr_ExceptionMatches),
        ADD_ADDR(PyErr_Clear),
        ADD_ADDR(PyErr_Format),
        ADD_ADDR(PyErr_SetString),
        ADD_ADDR(PyNumber_Multiply),
        ADD_ADDR(PyNumber_TrueDivide),
        ADD_ADDR(PyNumber_FloorDivide),
        ADD_ADDR(PyNumber_Add),
        ADD_ADDR(PyNumber_Subtract),
        ADD_ADDR(PyNumber_Lshift),
        ADD_ADDR(PyNumber_Rshift),
        ADD_ADDR(PyNumber_And),
        ADD_ADDR(PyNumber_Xor),
        ADD_ADDR(PyNumber_Or),
        ADD_ADDR(PyNumber_InPlaceMultiply),
        ADD_ADDR(PyNumber_InPlaceTrueDivide),
        ADD_ADDR(PyNumber_InPlaceFloorDivide),
        ADD_ADDR(PyNumber_InPlaceRemainder),
        ADD_ADDR(PyNumber_InPlaceAdd),
        ADD_ADDR(PyNumber_InPlaceSubtract),
        ADD_ADDR(PyNumber_InPlaceLshift),
        ADD_ADDR(PyNumber_InPlaceRshift),
        ADD_ADDR(PyNumber_InPlaceAnd),
        ADD_ADDR(PyNumber_InPlaceXor),
        ADD_ADDR(PyNumber_InPlaceOr),
        ADD_ADDR(PyLong_AsLong),
        ADD_ADDR(PyList_New),
        ADD_ADDR(PyTuple_New),
        ADD_ADDR(PyTuple_Pack),
        ADD_ADDR(PySequence_Contains),
        ADD_ADDR(_EnterRecursiveCall),
        ADD_ADDR(_LeaveRecursiveCall),
        ADD_ADDR(call_function),
        ADD_ADDR(format_exc_check_arg),
        ADD_ADDR(_make_function),
        ADD_ADDR(_unpack_iterable),
        ADD_ADDR(_exception_cmp),
        ADD_ADDR(_do_raise),
        ADD_ADDR(import_all_from),
        ADD_ADDR(special_lookup),
    
        ADD_ADDR(Py_True),
        ADD_ADDR(Py_False),
        ADD_ADDR(Py_None),
        ADD_ADDR_OF(PyDict_Type),
        ADD_ADDR_OF(PyList_Type),
        ADD_ADDR_OF(PyTuple_Type),
        ADD_ADDR(PyExc_KeyError),
        ADD_ADDR(PyExc_NameError),
        ADD_ADDR(PyExc_StopIteration),
        ADD_ADDR(PyExc_UnboundLocalError),
        ADD_ADDR(PyExc_SystemError),
        ADD_ADDR(PyExc_ImportError),
        ADD_ADDR(PyExc_AttributeError),
        ADD_ADDR_OF(_PyThreadState_Current),
        ADD_ADDR(NAME_ERROR_MSG),
        ADD_ADDR(GLOBAL_NAME_ERROR_MSG),
        ADD_ADDR(UNBOUNDLOCAL_ERROR_MSG),
        ADD_ADDR(UNBOUNDFREE_ERROR_MSG),
        ADD_ADDR(NO_LOCALS_LOAD_MSG),
        ADD_ADDR(NO_LOCALS_STORE_MSG),
        ADD_ADDR(NO_LOCALS_DELETE_MSG),
        ADD_ADDR(BUILD_CLASS_ERROR_MSG),
        ADD_ADDR(CANNOT_IMPORT_MSG),
        ADD_ADDR(IMPORT_NOT_FOUND_MSG),
        ADD_ADDR(BAD_EXCEPTION_MSG),
        ADD_ADDR_STR("__build_class__"),
        ADD_ADDR_STR("__import__"),
        ADD_ADDR_STR("__exit__"),
        ADD_ADDR_STR("__enter__"),
        ADD_ADDR_OF(exit_cache),
        ADD_ADDR_OF(enter_cache)
    };

    for(i=0; i<sizeof(addr_records)/sizeof(AddrRec); ++i) {
        tmp = PyLong_FromUnsignedLong(addr_records[i].addr);
        ret = PyDict_SetItemString(addrs,addr_records[i].name,tmp);
        Py_DECREF(tmp);
        if(ret == -1) return NULL;
    }
    
    Py_INCREF(&CompiledCodeType);
    if(PyModule_AddObject(m,"CompiledCode",(PyObject*)&CompiledCodeType) == -1) return NULL;

    assert(!old_func_call);
    old_func_call = PyFunction_Type.tp_call;
    PyFunction_Type.tp_call = _function_call;
    
    return m;
}

