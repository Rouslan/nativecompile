
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


static PyObject *load_args(PyObject ***pp_stack, int na);
static void err_args(PyObject *func, int flags, int nargs);
static PyObject *fast_function(PyObject *func, PyObject ***pp_stack, int n, int na, int nk);
static PyObject *do_call(PyObject *func, PyObject ***pp_stack, int na, int nk);
static PyObject *_cc_EvalCodeEx(PyObject *_entrypoint, PyObject *globals, PyObject *locals,
    PyObject **args, int argcount, PyObject **kws, int kwcount,
    PyObject **defs, int defcount, PyObject *kwdefs, PyObject *closure);

#define EXT_POP(STACK_POINTER) (*(STACK_POINTER)++)

#define GETLOCAL(i)     (fastlocals[i])
#define SETLOCAL(i, value)      do { PyObject *tmp = GETLOCAL(i); \
                                     GETLOCAL(i) = value; \
                                     Py_XDECREF(tmp); } while (0)



PyObject *mcode_index;



typedef struct {
    PyObject_HEAD

    PyObject *code;
    
    PyObject *(*entry)(PyFrameObject *);
#ifdef USE_MMAP
    int fd;
    size_t len;
#endif
} CompiledCode;


static void CompiledCode_dealloc(CompiledCode *self) {
    if(self->entry) {
#ifdef USE_MMAP
        munmap(self->entry,self->len);
        close(self->fd);
#else
        PyMem_Free(self->entry);
#endif
    }
    Py_DECREF(self->code);
}


static PyObject *CompiledCode_new(PyTypeObject *type,PyObject *args,PyObject *kwds) {
    CompiledCode *self;
    PyObject *filename_o;
    const char *filename_s;
    PyObject *code;
#ifdef USE_MMAP
    off_t slen;
    void *mem;
#else
    FILE *f;
    long len;
    size_t read;
#endif

    static char *kwlist[] = {"filename","code",NULL};

    if(!PyArg_ParseTupleAndKeywords(args,kwds,"O&O!",kwlist,
        PyUnicode_FSConverter,
        &filename_o,
        &PyCode_Type,
        &code)) return NULL;

    filename_s = PyBytes_AS_STRING(filename_o);
    
    self = (CompiledCode*)type->tp_alloc(type,0);
    if(self) {
        self->entry = NULL;
        self->code = code;
        Py_INCREF(code);
        
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
        self->entry = (PyObject *(*)(PyFrameObject *))mem;

#else
        /* just load the file contents into memory */

        if((f = fopen(filename_s,"rb")) == NULL) goto io_error;
        
        /* get the file length */
        if(fseek(f,0,SEEK_END) || (len = ftell(f)) == -1 || fseek(f,0,SEEK_SET)) {
            fclose(f);
            goto io_error;
        }
        
        if((self->entry = (PyObject *(*)(void))PyMem_Malloc(len)) == NULL) {
            fclose(f);
            goto error;
        }
        
        read = fread(self->entry,1,len,f);
        fclose(f);
        if(read < len) goto io_error;

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
    PyThreadState *tstate = PyThreadState_GET();
    PyFrameObject *frame = PyFrame_New(
        tstate,
        (PyCodeObject*)self->code,
        PyEval_GetGlobals(),
        PyEval_GetLocals());

    r = self->entry(frame);

    ++tstate->recursion_depth;
    Py_DECREF(frame);
    --tstate->recursion_depth;

    return r;
}

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
    0,                         /* tp_members */
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



typedef struct {
    PyObject_HEAD

    PyObject *compiled_code;
    PyObject *code_object;
    unsigned int entry_offset;
} CompiledEntryPoint;


static void CompiledEntryPoint_dealloc(CompiledEntryPoint *self) {
    Py_DECREF(self->compiled_code);
    Py_DECREF(self->code_object);
}

static PyObject *CompiledEntryPoint_new(PyTypeObject *type,PyObject *args,PyObject *kwds) {
    PyObject *compiled_code;
    PyObject *code_object;
    unsigned int entry_offset;
    static char *kwlist[] = {"compiled_code","entry_offset","code",0};

    if(!PyArg_ParseTupleAndKeywords(args,kwds,"O!O!k",kwlist,
        &CompiledCodeType,&compiled_code,
        &PyCode_Type,&code_object,
        &entry_offset)) return NULL;
	
    CompiledEntryPoint *self = (CompiledEntryPoint*)type->tp_alloc(type,0);

    if(self) {
	self->compiled_code = compiled_code;
	Py_INCREF(compiled_code);
        self->code_object = code_object;
        Py_INCREF(code_object);
	self->entry_offset = entry_offset;
    }

    return (PyObject*)self;
}

static PyTypeObject CompiledEntryPointType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "nativecompile.pyinternals.CompiledEntryPoint", /* tp_name */
    sizeof(CompiledEntryPoint),      /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)CompiledEntryPoint_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
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
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    CompiledEntryPoint_new,    /* tp_new */
};





/* The following 8 are modified versions of functions in Python/ceval.c.
 * Profiling and statistics code has been removed because it uses global
 * variables not accessable to this module. */

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

    while (pp_stack > pfunc) {
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
    PyObject *centry;
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
        retval = PyEval_EvalFrameEx(f,0);
        ++tstate->recursion_depth;
        Py_DECREF(f);
        --tstate->recursion_depth;
        return retval;
    }
    if (argdefs != NULL) {
        d = &PyTuple_GET_ITEM(argdefs, 0);
        nd = Py_SIZE(argdefs);
    }

    centry = PyObject_GetAttr(func,mcode_index);
    if(centry) {
        int isin = PyObject_IsInstance(centry,(PyObject*)&CompiledEntryPointType);
        if(isin<0) return NULL;
        if(isin) {
            PyObject *r = _cc_EvalCodeEx(centry, globals,
                             (PyObject *)NULL, (*pp_stack)+n-1, na,
                             (*pp_stack)+2*nk-1, nk, d, nd, kwdefs,
                             PyFunction_GET_CLOSURE(func));
            Py_DECREF(centry);
            return r;
        }
        Py_DECREF(centry);
    } else {
        if(!PyErr_ExceptionMatches(PyExc_AttributeError)) return NULL;
        PyErr_Clear();
    }

    return PyEval_EvalCodeEx((PyObject*)co, globals,
                             (PyObject *)NULL, (*pp_stack)+n-1, na,
                             (*pp_stack)+2*nk-1, nk, d, nd, kwdefs,
                             PyFunction_GET_CLOSURE(func));
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

static PyObject *
_cc_EvalCodeEx(PyObject *_entrypoint, PyObject *globals, PyObject *locals,
           PyObject **args, int argcount, PyObject **kws, int kwcount,
           PyObject **defs, int defcount, PyObject *kwdefs, PyObject *closure)
{
    CompiledEntryPoint *entrypoint = (CompiledEntryPoint*)_entrypoint;
    PyCodeObject* co = (PyCodeObject*)(entrypoint->code_object);
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
            x = args[i];
            Py_INCREF(x);
            SETLOCAL(i, x);
        }
        if (co->co_flags & CO_VARARGS) {
            u = PyTuple_New(argcount - n);
            if (u == NULL)
                goto fail;
            SETLOCAL(total_args, u);
            for (i = n; i < argcount; i++) {
                x = args[i];
                Py_INCREF(x);
                PyTuple_SET_ITEM(u, i-n, x);
            }
        }
        for (i = 0; i < kwcount; i++) {
            PyObject **co_varnames;
            PyObject *keyword = kws[2*i];
            PyObject *value = kws[2*i + 1];
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

    retval = (((CompiledCode*)(entrypoint->compiled_code))->entry + entrypoint->entry_offset)(f);

fail:

    assert(tstate != NULL);
    ++tstate->recursion_depth;
    Py_DECREF(f);
    --tstate->recursion_depth;
    return retval;
}



/* Py_EnterRecursiveCall and Py_LeaveRecursiveCall are somewhat complicated
 * macros so they are wrapped in the following two functions */

static int _EnterRecursiveCall(char *where) {
    return Py_EnterRecursiveCall(where);
}

static void _LeaveRecursiveCall(void) {
    Py_LeaveRecursiveCall();
}


static struct PyModuleDef this_module = {
    PyModuleDef_HEAD_INIT,
    "pyinternals",
    NULL,
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

#define ADD_ADDR_NAME(item,name) \
    tmp = PyLong_FromUnsignedLong((unsigned long)(item)); \
    ret = PyDict_SetItemString(addrs,name,tmp); \
    Py_DECREF(tmp); \
    if(ret == -1) return NULL;

#define ADD_ADDR(item) ADD_ADDR_NAME(item,#item)


PyMODINIT_FUNC
PyInit_pyinternals(void) {
    PyObject *m;
    PyObject *addrs;
    PyObject *tmp;
    int ret;
    /*PyObject *(*fptr)(void);*/
    
    if(PyType_Ready(&CompiledCodeType) < 0) return NULL;
    if(PyType_Ready(&CompiledEntryPointType) < 0) return NULL;

    m = PyModule_Create(&this_module);
    if(PyModule_AddIntConstant(m,"refcnt_offset",offsetof(PyObject,ob_refcnt)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"type_offset",offsetof(PyObject,ob_type)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"type_dealloc_offset",offsetof(PyTypeObject,tp_dealloc)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"type_iternext_offset",offsetof(PyTypeObject,tp_iternext)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"list_item_offset",offsetof(PyListObject,ob_item)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"tuple_item_offset",offsetof(PyTupleObject,ob_item)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"frame_builtins_offset",offsetof(PyFrameObject,f_builtins)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"frame_globals_offset",offsetof(PyFrameObject,f_globals)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"frame_locals_offset",offsetof(PyFrameObject,f_locals)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"threadstate_frame_offset",offsetof(PyThreadState,frame)) == -1) return NULL;
    if(PyModule_AddStringConstant(m,"architecture",ARCHITECTURE) == -1) return NULL;
    if(PyModule_AddObject(m,"ref_debug",PyBool_FromLong(REF_DEBUG_VAL)) == -1) return NULL;
    if(PyModule_AddObject(m,"count_allocs",PyBool_FromLong(COUNT_ALLOCS_VAL)) == -1) return NULL;

    mcode_index = PyUnicode_InternFromString("__compiled_entry_point__");
    if(!mcode_index) return NULL;
    if(PyModule_AddObject(m,"mcode_index",mcode_index) == -1) return NULL;
    
    addrs = PyDict_New();
    if(!addrs) return NULL;
    
    if(PyModule_AddObject(m,"raw_addresses",addrs) == -1) return NULL;
    
    ADD_ADDR(Py_IncRef)
    ADD_ADDR(Py_DecRef)
    ADD_ADDR(PyDict_GetItem)
    ADD_ADDR(PyDict_SetItem)
    ADD_ADDR(PyObject_GetItem)
    ADD_ADDR(PyObject_SetItem)
    ADD_ADDR(PyObject_GetIter)
    ADD_ADDR(PyObject_GetAttr)
    ADD_ADDR(PyObject_IsTrue)
    ADD_ADDR(PyEval_GetGlobals)
    ADD_ADDR(PyEval_GetBuiltins)
    ADD_ADDR(PyEval_GetLocals)
    ADD_ADDR(PyErr_Occurred)
    ADD_ADDR(PyErr_ExceptionMatches)
    ADD_ADDR(PyErr_Clear)
    ADD_ADDR(PyErr_Format)
    ADD_ADDR(PyNumber_Multiply)
    ADD_ADDR(PyNumber_TrueDivide)
    ADD_ADDR(PyNumber_FloorDivide)
    ADD_ADDR(PyNumber_Add)
    ADD_ADDR(PyNumber_Subtract)
    ADD_ADDR(PyNumber_Lshift)
    ADD_ADDR(PyNumber_Rshift)
    ADD_ADDR(PyNumber_And)
    ADD_ADDR(PyNumber_Xor)
    ADD_ADDR(PyNumber_Or)
    ADD_ADDR(PyNumber_InPlaceMultiply)
    ADD_ADDR(PyNumber_InPlaceTrueDivide)
    ADD_ADDR(PyNumber_InPlaceFloorDivide)
    ADD_ADDR(PyNumber_InPlaceRemainder)
    ADD_ADDR(PyNumber_InPlaceAdd)
    ADD_ADDR(PyNumber_InPlaceSubtract)
    ADD_ADDR(PyNumber_InPlaceLshift)
    ADD_ADDR(PyNumber_InPlaceRshift)
    ADD_ADDR(PyNumber_InPlaceAnd)
    ADD_ADDR(PyNumber_InPlaceXor)
    ADD_ADDR(PyNumber_InPlaceOr)
    ADD_ADDR(PyList_New)
    ADD_ADDR(PyTuple_New)
    ADD_ADDR(_EnterRecursiveCall)
    ADD_ADDR(_LeaveRecursiveCall)
    ADD_ADDR(call_function)
    ADD_ADDR(format_exc_check_arg)
    ADD_ADDR(_cc_EvalCodeEx)
    
    ADD_ADDR_NAME(&PyDict_Type,"PyDict_Type")
    ADD_ADDR(PyExc_KeyError)
    ADD_ADDR(PyExc_NameError)
    ADD_ADDR(PyExc_StopIteration)
    ADD_ADDR_NAME(&_PyThreadState_Current,"_PyThreadState_Current")
    ADD_ADDR(NAME_ERROR_MSG)
    ADD_ADDR(GLOBAL_NAME_ERROR_MSG)
    ADD_ADDR(UNBOUNDLOCAL_ERROR_MSG)
    ADD_ADDR(UNBOUNDFREE_ERROR_MSG)
    
    Py_INCREF(&CompiledCodeType);
    if(PyModule_AddObject(m,"CompiledCode",(PyObject*)&CompiledCodeType) == -1) return NULL;

    Py_INCREF(&CompiledEntryPointType);
    if(PyModule_AddObject(m,"CompiledEntryPoint",(PyObject*)&CompiledEntryPointType) == -1) return NULL;
    
    return m;
}

