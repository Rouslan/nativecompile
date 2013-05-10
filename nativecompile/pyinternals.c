
#include <Python.h>
#include <structmember.h>
#include <frameobject.h>
#include <opcode.h>

#if defined(__linux__) || defined(__linux) || defined(linux)
    #include <sys/mman.h>
    #include <unistd.h>
    #include <stdlib.h>
    
    #define USE_POSIX 1
#elif defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__) || defined(__WINDOWS__)
    #include <Windows.h>

    #define USE_WIN 1
#endif

#if defined(__GNUC__) && !defined(NDEBUG)
#define GDB_JIT_SUPPORT 1
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
static int call_trace(Py_tracefunc func, PyObject *obj, PyFrameObject *frame,
                      int what, PyObject *arg);
static int call_trace_protected(Py_tracefunc func, PyObject *obj, PyFrameObject *frame,
                                int what, PyObject *arg);

#define EXT_POP(STACK_POINTER) (*(STACK_POINTER)++)

#define GETLOCAL(i)     (fastlocals[i])
#define SETLOCAL(i, value)      do { PyObject *tmp = GETLOCAL(i); \
                                     GETLOCAL(i) = value; \
                                     Py_XDECREF(tmp); } while (0)

#define CALL_FLAG_VAR 1
#define CALL_FLAG_KW 2


static ternaryfunc old_func_call = NULL;
static PyCFunction old_gen_send = NULL;
static PyCFunction old_gen_throw = NULL;
static PyCFunction old_gen_close = NULL;
static iternextfunc old_gen_iternext = NULL;


#ifdef GDB_JIT_SUPPORT

#include <stdint.h>

typedef enum {
    JIT_NOACTION = 0,
    JIT_REGISTER_FN,
    JIT_UNREGISTER_FN
} jit_actions_t;

struct jit_code_entry {
    struct jit_code_entry *next_entry;
    struct jit_code_entry *prev_entry;
    const void *symfile_addr;
    uint64_t symfile_size;
};

typedef struct {
    uint32_t version;
    uint32_t action_flag;
    struct jit_code_entry *relevant_entry;
    struct jit_code_entry *first_entry;
} jit_descriptor;

void __attribute__((noinline)) __jit_debug_register_code(void) {};

jit_descriptor __jit_debug_descriptor = {1,0,0,0};

int register_gdb(const void *addr,size_t size) {
    struct jit_code_entry *entry = PyMem_Malloc(sizeof(struct jit_code_entry));
    if(!entry) {
        PyErr_NoMemory();
        return -1;
    }

    entry->next_entry = __jit_debug_descriptor.first_entry;
    entry->prev_entry = NULL;
    entry->symfile_addr = addr;
    entry->symfile_size = size;

    if(__jit_debug_descriptor.first_entry)
        __jit_debug_descriptor.first_entry->prev_entry = entry;

    __jit_debug_descriptor.first_entry = entry;
    __jit_debug_descriptor.relevant_entry = entry;
    __jit_debug_descriptor.action_flag = JIT_REGISTER_FN;

    __jit_debug_register_code();

    return 0;
}

void unregister_gdb(const void *addr) {
    struct jit_code_entry *entry;
    for(entry = __jit_debug_descriptor.first_entry; entry; entry = entry->next_entry) {
        if(entry->symfile_addr == addr) {
            if(entry->next_entry) entry->next_entry->prev_entry = entry->prev_entry;

            if(entry->prev_entry) entry->prev_entry->next_entry = entry->next_entry;
            else __jit_debug_descriptor.first_entry = entry->next_entry;

            entry->next_entry = NULL;
            entry->prev_entry = NULL;

            __jit_debug_descriptor.relevant_entry = entry;
            __jit_debug_descriptor.action_flag = JIT_UNREGISTER_FN;
            __jit_debug_register_code();

            PyMem_Free(entry);
            return;
        }
    }
    assert(0);
}

#endif


typedef PyObject *(*entry_type)(PyFrameObject *,int);

typedef struct {
    PyObject_HEAD

    entry_type entry;
#ifdef USE_POSIX
    size_t size; /* needed by munmap */
#endif

    /* a tuple of CodeObjectWithCCode objects */
    PyObject *entry_points;

#ifdef GDB_JIT_SUPPORT
    unsigned char gdb_reg;
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
   PyCodeObject to this surrogate and identify it using a bit in co_flags not
   used by CPython */
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

static inline PyObject *call_ccode_or_evalframe(PyObject *obj,PyFrameObject *frame,int exc) {
    if(HAS_CCODE(obj)) {
        /* ternary operator not used so a breakpoint can be placed here: */
        return GET_CCODE_FUNC(obj)(frame,exc);
    } else return PyEval_EvalFrameEx(frame,exc);

    /* return HAS_CCODE(obj) ? GET_CCODE_FUNC(obj)(frame,exc) : PyEval_EvalFrameEx(frame,exc); */
}

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
#ifdef GDB_JIT_SUPPORT
        if(self->gdb_reg) unregister_gdb(self->entry);
#endif

#ifdef USE_POSIX
        munmap(self->entry,self->size);
#elif USE_WIN
        VirtualFree(self->entry,0,MEM_RELEASE);
#else
        PyMem_Free(self->entry);
#endif
    }
}

static int alloc_for_func(CompiledCode *self,long size) {
    assert(size >= 0);

    if(!size) {
        PyErr_SetString(PyExc_ValueError,"parts cannot be empty");
        return -1;
    }
        
#ifdef USE_POSIX
    if((self->entry = mmap(0,size,PROT_READ|PROT_WRITE|PROT_EXEC,MAP_PRIVATE|MAP_ANON,-1,0)) == MAP_FAILED) {
    /*if(posix_memalign((void**)&self->entry,sysconf(_SC_PAGESIZE),size)) {*/
        self->entry = NULL;
        PyErr_NoMemory();
        return -1;
    }
    self->size = size;
#elif defined(USE_WIN)
    if(!(self->entry = VirtualAlloc(0,size,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE))) {
        PyErr_NoMemory();
        return -1;
    }
#else
    if(!(self->entry = PyMem_Malloc(size))) {
        PyErr_NoMemory();
        return -1;
    }
#endif
    return 0;
}

static PyObject *CompiledCode_new(PyTypeObject *type,PyObject *args,PyObject *kwds) {
    CompiledCode *self;
    long size;
    PyObject *parts;
    PyObject *entry_points;
    Py_ssize_t i;
    PyObject *lparts = NULL;
    PyObject *item;

#ifdef GDB_JIT_SUPPORT
    Py_buffer buff = {0};
#endif

    static char *kwlist[] = {"parts","entry_points",NULL};

    if(!PyArg_ParseTupleAndKeywords(args,kwds,"OO",kwlist,
        &parts,
        &entry_points)) return NULL;
    
    self = (CompiledCode*)type->tp_alloc(type,0);
    if(self) {
        self->entry = NULL;

        self->entry_points = PyObject_CallFunctionObjArgs(
            (PyObject*)&PyTuple_Type,
            entry_points,NULL);
        if(!self->entry_points) goto error;

        i = PyTuple_GET_SIZE(self->entry_points);

        if(PyTuple_GET_SIZE(self->entry_points) < 1) {
            PyErr_SetString(PyExc_ValueError,"entry_points needs to have at least one entry");
            goto error;
        }

        for(i=0; i<PyTuple_GET_SIZE(self->entry_points); ++i) {
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

#ifdef GDB_JIT_SUPPORT
        if(PyObject_HasAttrString(parts,"rebase")) {
            int err;
            PyObject *buffobj;

            if(!(item = PyObject_GetAttrString(parts,"buff"))) goto error;

            buffobj = PyObject_CallMethod(item,"getbuffer",NULL);
            Py_DECREF(item);
            if(!buffobj) goto error;

            err = PyObject_GetBuffer(buffobj,&buff,PyBUF_SIMPLE);
            Py_DECREF(buffobj);
            if(err) goto error;

            if(alloc_for_func(self,buff.len)) goto error;

            if(!(item = PyObject_CallMethod(parts,"rebase","k",self->entry))) goto error;
            Py_DECREF(item);

            size = buff.len;
            memcpy(self->entry,buff.buf,(size_t)size);

            if(register_gdb(self->entry,size)) goto error;
            self->gdb_reg = 1;
        } else
#endif
        if(PyBytes_Check(parts)) {
            size = PyBytes_GET_SIZE(parts);
            if(alloc_for_func(self,size)) goto error;
            memcpy(self->entry,PyBytes_AS_STRING(parts),(size_t)size);
        } else {
            char *dest;

            lparts = PyObject_CallFunctionObjArgs((PyObject*)&PyList_Type,parts,NULL);
            if(!lparts) goto error;

            size = 0;
            for(i=0; i < PyList_GET_SIZE(lparts); ++i) {
                item = PyList_GET_ITEM(lparts,i);
                if(!PyBytes_Check(item)) {
                    PyErr_SetString(PyExc_TypeError,"an item in parts is not a bytes object");
                    goto error;
                }
                size += PyBytes_GET_SIZE(item);
            }

            if(alloc_for_func(self,size)) goto error;

            dest = (char*)self->entry;
            for(i=0; i < PyList_GET_SIZE(lparts); ++i) {
                item = PyList_GET_ITEM(lparts,i);
                memcpy(dest,PyBytes_AS_STRING(item),PyBytes_GET_SIZE(item));
                dest += PyBytes_GET_SIZE(item);
            }
        }

#ifdef USE_POSIX
        if(mprotect(self->entry,size,PROT_READ|PROT_EXEC)) {
            PyErr_SetFromErrno(PyExc_OSError);
            goto error;
        }
    #ifdef __GNUC__
        __builtin___clear_cache(self->entry,(char*)self->entry + size);
    #else
        #warning "Don't know how to flush instruction cache on this compiler and OS"
    #endif
#elif defined(USE_WIN)
        if(!VirtualProtect(self->entry,size,PAGE_EXECUTE_READ)) {
            PyErr_SetFromWindowsErr(0);
            goto error;
        }
        FlushInstructionCache(GetCurrentProcess(),self->entry,size);
#endif

        goto end; /* skip over the error handling code */
    error:
        Py_DECREF(self);
        self = NULL;
    end:
        Py_XDECREF(lparts);
#ifdef GDB_JIT_SUPPORT
        if(buff.obj) PyBuffer_Release(&buff);
#endif
    }
        
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


PyDoc_STRVAR(read_address_doc,
"read_address(address,length) -> bytes\n\
\n\
Get the contents of memory at an arbitary address. Warning: this function is\n\
very unsafe. Improper use can easily cause a segmentation fault.");

static PyObject *read_address(PyObject *self,PyObject *args) {
    unsigned long addr;
    Py_ssize_t length;

    if(!PyArg_ParseTuple(args,"kn:read_address",&addr,&length)) return NULL;

    return PyBytes_FromStringAndSize((const char*)addr,length);
}




/* The following are modified versions of functions in Python/ceval.c */

#define C_TRACE(x, call) \
if (tstate->use_tracing && tstate->c_profilefunc) { \
    if (call_trace(tstate->c_profilefunc, \
        tstate->c_profileobj, \
        tstate->frame, PyTrace_C_CALL, \
        func)) { \
        x = NULL; \
    } \
    else { \
        x = call; \
        if (tstate->c_profilefunc != NULL) { \
            if (x == NULL) { \
                call_trace_protected(tstate->c_profilefunc, \
                    tstate->c_profileobj, \
                    tstate->frame, PyTrace_C_EXCEPTION, \
                    func); \
            } else { \
                if (call_trace(tstate->c_profilefunc, \
                    tstate->c_profileobj, \
                    tstate->frame, PyTrace_C_RETURN, \
                    func)) { \
                    Py_DECREF(x); \
                    x = NULL; \
                } \
            } \
        } \
    } \
} else { \
    x = call; \
    }


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
        PyThreadState *tstate = PyThreadState_GET();

        if (flags & (METH_NOARGS | METH_O)) {
            PyCFunction meth = PyCFunction_GET_FUNCTION(func);
            PyObject *self = PyCFunction_GET_SELF(func);
            if (flags & METH_NOARGS && na == 0) {
                C_TRACE(x, (*meth)(self,NULL));
            }
            else if (flags & METH_O && na == 1) {
                PyObject *arg = EXT_POP(pp_stack);
                C_TRACE(x, (*meth)(self,arg));
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
            C_TRACE(x, PyCFunction_Call(func,callargs,NULL));
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

        retval = call_ccode_or_evalframe((PyObject*)co,f,0);

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

    if (PyCFunction_Check(func)) {
        PyThreadState *tstate = PyThreadState_GET();
        C_TRACE(result, PyCFunction_Call(func, callargs, kwdict));
    }
    else
        result = PyObject_Call(func, callargs, kwdict);
call_fail:
    Py_XDECREF(callargs);
    Py_XDECREF(kwdict);
    return result;
}

static PyObject *
update_star_args(int nstack, int nstar, PyObject *stararg,
                 PyObject ***pp_stack)
{
    PyObject *callargs, *w;

    callargs = PyTuple_New(nstack + nstar);
    if (callargs == NULL) {
        return NULL;
    }
    if (nstar) {
        int i;
        for (i = 0; i < nstar; i++) {
            PyObject *a = PyTuple_GET_ITEM(stararg, i);
            Py_INCREF(a);
            PyTuple_SET_ITEM(callargs, nstack + i, a);
        }
    }
    while (--nstack >= 0) {
        w = EXT_POP(*pp_stack);
        PyTuple_SET_ITEM(callargs, nstack, w);
    }
    return callargs;
}

static PyObject *
ext_do_call(PyObject *func, PyObject **pp_stack, int flags, int na, int nk)
{
    int nstar = 0;
    PyObject *callargs = NULL;
    PyObject *stararg = NULL;
    PyObject *kwdict = NULL;
    PyObject *result = NULL;

    PyObject **end = pp_stack + na + nk*2;
    if(flags & CALL_FLAG_VAR) ++end;
    if(flags & CALL_FLAG_KW) ++end;

    if (flags & CALL_FLAG_KW) {
        kwdict = EXT_POP(pp_stack);
        if (!PyDict_Check(kwdict)) {
            PyObject *d;
            d = PyDict_New();
            if (d == NULL)
                goto ext_call_fail;
            if (PyDict_Update(d, kwdict) != 0) {
                Py_DECREF(d);
                /* PyDict_Update raises attribute
                 * error (percolated from an attempt
                 * to get 'keys' attribute) instead of
                 * a type error if its second argument
                 * is not a mapping.
                 */
                if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
                    PyErr_Format(PyExc_TypeError,
                                 "%.200s%.200s argument after ** "
                                 "must be a mapping, not %.200s",
                                 PyEval_GetFuncName(func),
                                 PyEval_GetFuncDesc(func),
                                 kwdict->ob_type->tp_name);
                }
                goto ext_call_fail;
            }
            Py_DECREF(kwdict);
            kwdict = d;
        }
    }
    if (flags & CALL_FLAG_VAR) {
        stararg = EXT_POP(pp_stack);
        if (!PyTuple_Check(stararg)) {
            PyObject *t = NULL;
            t = PySequence_Tuple(stararg);
            if (t == NULL) {
                if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                    PyErr_Format(PyExc_TypeError,
                                 "%.200s%.200s argument after * "
                                 "must be a sequence, not %200s",
                                 PyEval_GetFuncName(func),
                                 PyEval_GetFuncDesc(func),
                                 stararg->ob_type->tp_name);
                }
                goto ext_call_fail;
            }
            Py_DECREF(stararg);
            stararg = t;
        }
        nstar = PyTuple_GET_SIZE(stararg);
    }
    if (nk > 0) {
        kwdict = update_keyword_args(kwdict, nk, &pp_stack, func);
        if (kwdict == NULL)
            goto ext_call_fail;
    }
    callargs = update_star_args(na, nstar, stararg, &pp_stack);
    if (callargs == NULL)
        goto ext_call_fail;

    if (PyCFunction_Check(func)) {
        PyThreadState *tstate = PyThreadState_GET();
        C_TRACE(result, PyCFunction_Call(func, callargs, kwdict));
    }
    else
        result = PyObject_Call(func, callargs, kwdict);
ext_call_fail:
    Py_XDECREF(callargs);
    Py_XDECREF(kwdict);
    Py_XDECREF(stararg);
    while(pp_stack < end) {
        PyObject *o = EXT_POP(pp_stack);
        Py_DECREF(o);
    }
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

static PyObject *
format_exc_check_arg(PyObject *exc, const char *format_str, PyObject *obj)
{
    const char *obj_str;

    if (!obj)
        return NULL;

    obj_str = _PyUnicode_AsString(obj);
    if (!obj_str)
        return NULL;

    return PyErr_Format(exc, format_str, obj_str);
}

static PyObject *
format_exc_unbound(PyCodeObject *co, int oparg)
{
    PyObject *name;
    if (PyErr_Occurred())
        return NULL;
    if (oparg < PyTuple_GET_SIZE(co->co_cellvars)) {
        name = PyTuple_GET_ITEM(co->co_cellvars,
                                oparg);
        format_exc_check_arg(
            PyExc_UnboundLocalError,
            UNBOUNDLOCAL_ERROR_MSG,
            name);
    } else {
        name = PyTuple_GET_ITEM(co->co_freevars, oparg -
                                PyTuple_GET_SIZE(co->co_cellvars));
        format_exc_check_arg(PyExc_NameError,
                             UNBOUNDFREE_ERROR_MSG, name);
    }
    return NULL;
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

    retval = call_ccode_or_evalframe(_co,f,0);

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

static int
call_trace(Py_tracefunc func, PyObject *obj, PyFrameObject *frame,
           int what, PyObject *arg)
{
    register PyThreadState *tstate = frame->f_tstate;
    int result;
    if (tstate->tracing)
        return 0;
    tstate->tracing++;
    tstate->use_tracing = 0;
    result = func(obj, frame, what, arg);
    tstate->use_tracing = ((tstate->c_tracefunc != NULL)
                           || (tstate->c_profilefunc != NULL));
    tstate->tracing--;
    return result;
}

static int
call_trace_protected(Py_tracefunc func, PyObject *obj, PyFrameObject *frame,
                     int what, PyObject *arg)
{
    PyObject *type, *value, *traceback;
    int err;
    PyErr_Fetch(&type, &value, &traceback);
    err = call_trace(func, obj, frame, what, arg);
    if (err == 0)
    {
        PyErr_Restore(type, value, traceback);
        return 0;
    }
    else {
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        return -1;
    }
}

static void
call_exc_trace(Py_tracefunc func, PyObject *self, PyFrameObject *f)
{
    PyObject *type, *value, *traceback, *arg;
    int err;
    PyErr_Fetch(&type, &value, &traceback);
    if (value == NULL) {
        value = Py_None;
        Py_INCREF(value);
    }
    arg = PyTuple_Pack(3, type, value, traceback);
    if (arg == NULL) {
        PyErr_Restore(type, value, traceback);
        return;
    }
    err = call_trace(func, self, f, PyTrace_EXCEPTION, arg);
    Py_DECREF(arg);
    if (err == 0)
        PyErr_Restore(type, value, traceback);
    else {
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
    }
}

static PyObject *
unicode_concatenate(PyObject *v, PyObject *w, PyFrameObject *f,
                    unsigned char next_instr, int instr_arg)
{
    Py_ssize_t v_len = PyUnicode_GET_SIZE(v);
    Py_ssize_t w_len = PyUnicode_GET_SIZE(w);
    Py_ssize_t new_len = v_len + w_len;
    if (new_len < 0) {
        PyErr_SetString(PyExc_OverflowError,
                        "strings are too large to concat");
        return NULL;
    }

    if (Py_REFCNT(v) == 2) {
        switch (next_instr) {
        case STORE_FAST:
        {
            PyObject **fastlocals = f->f_localsplus;
            if (GETLOCAL(instr_arg) == v)
                SETLOCAL(instr_arg, NULL);
            break;
        }
        case STORE_DEREF:
        {
            PyObject **freevars = (f->f_localsplus +
                                   f->f_code->co_nlocals);
            PyObject *c = freevars[instr_arg];
            if (PyCell_GET(c) == v)
                PyCell_Set(c, NULL);
            break;
        }
        case STORE_NAME:
        {
            PyObject *names = f->f_code->co_names;
            PyObject *name = PyTuple_GET_ITEM(names, instr_arg);
            PyObject *locals = f->f_locals;
            if (PyDict_CheckExact(locals) &&
                PyDict_GetItem(locals, name) == v) {
                if (PyDict_DelItem(locals, name) != 0) {
                    PyErr_Clear();
                }
            }
            break;
        }
        }
    }

    if (Py_REFCNT(v) == 1 && !PyUnicode_CHECK_INTERNED(v)) {
        if (PyUnicode_Resize(&v, new_len) != 0) {
            return NULL;
        }

        memcpy(PyUnicode_AS_UNICODE(v) + v_len,
               PyUnicode_AS_UNICODE(w), w_len*sizeof(Py_UNICODE));
        return v;
    }
    else {
        w = PyUnicode_Concat(v, w);
        Py_DECREF(v);
        return w;
    }
}


/* The following are modified functions from Objects/genobject.c: */

static PyObject *gen_send_ex(PyGenObject *gen, PyObject *arg, int exc)
{
    PyThreadState *tstate = PyThreadState_GET();
    PyFrameObject *f = gen->gi_frame;
    PyObject *result;

    if (gen->gi_running) {
        PyErr_SetString(PyExc_ValueError,
                        "generator already executing");
        return NULL;
    }
    if (f==NULL || f->f_stacktop == NULL) {
        if (arg && !exc)
            PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    if (f->f_lasti == -1) {
        if (arg && arg != Py_None) {
            PyErr_SetString(PyExc_TypeError,
                            "can't send non-None value to a "
                            "just-started generator");
            return NULL;
        }
    } else {
        result = arg ? arg : Py_None;
        Py_INCREF(result);
        *(f->f_stacktop++) = result;
    }

    Py_XINCREF(tstate->frame);
    assert(f->f_back == NULL);
    f->f_back = tstate->frame;

    gen->gi_running = 1;
    result = call_ccode_or_evalframe(gen->gi_code,f,exc);
    gen->gi_running = 0;

    assert(f->f_back == tstate->frame);
    Py_CLEAR(f->f_back);

    if (result == Py_None && f->f_stacktop == NULL) {
        Py_DECREF(result);
        result = NULL;
        if (arg)
            PyErr_SetNone(PyExc_StopIteration);
    }

    if (!result || f->f_stacktop == NULL) {
        Py_DECREF(f);
        gen->gi_frame = NULL;
    }

    return result;
}

static PyObject *
gen_send(PyGenObject *gen, PyObject *arg)
{
    return gen_send_ex(gen, arg, 0);
}

static PyObject *
gen_close(PyGenObject *gen, PyObject *args)
{
    PyObject *retval;
    PyErr_SetNone(PyExc_GeneratorExit);
    retval = gen_send_ex(gen, Py_None, 1);
    if (retval) {
        Py_DECREF(retval);
        PyErr_SetString(PyExc_RuntimeError,
                        "generator ignored GeneratorExit");
        return NULL;
    }
    if (PyErr_ExceptionMatches(PyExc_StopIteration)
        || PyErr_ExceptionMatches(PyExc_GeneratorExit))
    {
        PyErr_Clear();
        Py_INCREF(Py_None);
        return Py_None;
    }
    return NULL;
}

static PyObject *
gen_throw(PyGenObject *gen, PyObject *args)
{
    PyObject *typ;
    PyObject *tb = NULL;
    PyObject *val = NULL;

    if (!PyArg_UnpackTuple(args, "throw", 1, 3, &typ, &val, &tb))
        return NULL;

    if (tb == Py_None)
        tb = NULL;
    else if (tb != NULL && !PyTraceBack_Check(tb)) {
        PyErr_SetString(PyExc_TypeError,
            "throw() third argument must be a traceback object");
        return NULL;
    }

    Py_INCREF(typ);
    Py_XINCREF(val);
    Py_XINCREF(tb);

    if (PyExceptionClass_Check(typ)) {
        PyErr_NormalizeException(&typ, &val, &tb);
    }

    else if (PyExceptionInstance_Check(typ)) {
        if (val && val != Py_None) {
            PyErr_SetString(PyExc_TypeError,
              "instance exception may not have a separate value");
            goto failed_throw;
        }
        else {
            Py_XDECREF(val);
            val = typ;
            typ = PyExceptionInstance_Class(typ);
            Py_INCREF(typ);
        }
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "exceptions must be classes or instances "
                     "deriving from BaseException, not %s",
                     typ->ob_type->tp_name);
            goto failed_throw;
    }

    PyErr_Restore(typ, val, tb);
    return gen_send_ex(gen, Py_None, 1);

failed_throw:
    Py_DECREF(typ);
    Py_XDECREF(val);
    Py_XDECREF(tb);
    return NULL;
}

static PyObject *
gen_iternext(PyGenObject *gen)
{
    return gen_send_ex(gen, NULL, 0);
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
    {"read_address",read_address,METH_VARARGS,read_address_doc},
    {NULL}
};

static void free_module(void *_) {
    assert(old_func_call);
    PyFunction_Type.tp_call = old_func_call;
    PyGen_Type.tp_methods[0].ml_meth = old_gen_send;
    PyGen_Type.tp_methods[1].ml_meth = old_gen_throw;
    PyGen_Type.tp_methods[2].ml_meth = old_gen_close;
    PyGen_Type.tp_iternext = old_gen_iternext;
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


#define M_OFFSET(type,member) {#member,offsetof(type,member)}

typedef struct {
    const char *name;
    int offset;
} OffsetMember;

typedef struct {
    const char *name;
    OffsetMember *members;
} OffsetStruct;

static OffsetStruct member_offsets[] = {
    {"PyObject",(OffsetMember[]){
            M_OFFSET(PyObject,ob_refcnt),
            M_OFFSET(PyObject,ob_type),
            {NULL}}},
    {"PyVarObject",(OffsetMember[]){M_OFFSET(PyVarObject,ob_size),{NULL}}},
    {"PyTypeObject",(OffsetMember[]){
            M_OFFSET(PyTypeObject,tp_dealloc),
            M_OFFSET(PyTypeObject,tp_iternext),
            M_OFFSET(PyTypeObject,tp_flags),
            {NULL}}},
    {"PyListObject",(OffsetMember[]){M_OFFSET(PyListObject,ob_item),{NULL}}},
    {"PyTupleObject",(OffsetMember[]){M_OFFSET(PyTupleObject,ob_item),{NULL}}},
    {"PyFrameObject",(OffsetMember[]){
            M_OFFSET(PyFrameObject,f_back),
            M_OFFSET(PyFrameObject,f_builtins),
            M_OFFSET(PyFrameObject,f_globals),
            M_OFFSET(PyFrameObject,f_locals),
            M_OFFSET(PyFrameObject,f_localsplus),
            M_OFFSET(PyFrameObject,f_valuestack),
            M_OFFSET(PyFrameObject,f_stacktop),
            M_OFFSET(PyFrameObject,f_exc_type),
            M_OFFSET(PyFrameObject,f_exc_value),
            M_OFFSET(PyFrameObject,f_exc_traceback),
            M_OFFSET(PyFrameObject,f_lasti),
            {NULL}}},
    {"PyThreadState",(OffsetMember[]){
            M_OFFSET(PyThreadState,frame),
            M_OFFSET(PyThreadState,exc_traceback),
            M_OFFSET(PyThreadState,exc_value),
            M_OFFSET(PyThreadState,exc_type),
            M_OFFSET(PyThreadState,c_tracefunc),
            M_OFFSET(PyThreadState,c_traceobj),
            {NULL}}},
    {"PyCellObject",(OffsetMember[]){M_OFFSET(PyCellObject,ob_ref),{NULL}}},
    {"PyMethodObject",(OffsetMember[]){
            M_OFFSET(PyMethodObject,im_self),
            M_OFFSET(PyMethodObject,im_func),
            {NULL}}},
    {NULL}
};

PyMODINIT_FUNC
PyInit_pyinternals(void) {
    PyObject *m, *addrs, *m_offsets, *m_dict, *tmp;
    OffsetStruct *structs;
    OffsetMember *members;
    int ret, i;
    
    if(PyType_Ready(&CompiledCodeType) < 0) return NULL;

    m = PyModule_Create(&this_module);
    if(!m) return NULL;

    if(PyModule_AddStringConstant(m,"ARCHITECTURE",ARCHITECTURE) == -1) return NULL;
    if(PyModule_AddObject(m,"REF_DEBUG",PyBool_FromLong(REF_DEBUG_VAL)) == -1) return NULL;
    if(PyModule_AddObject(m,"COUNT_ALLOCS",PyBool_FromLong(COUNT_ALLOCS_VAL)) == -1) return NULL;
#ifdef GDB_JIT_SUPPORT
    Py_INCREF(Py_True);
    if(PyModule_AddObject(m,"GDB_JIT_SUPPORT",Py_True) == -1) return NULL;
#endif


    if(!(m_offsets = PyDict_New())) return NULL;
    if(PyModule_AddObject(m,"member_offsets",m_offsets) == -1) return NULL;

    for(structs = member_offsets; structs->name; ++structs) {
        if(!(m_dict = PyDict_New())) return NULL;
        ret = PyDict_SetItemString(m_offsets,structs->name,m_dict);
        Py_DECREF(m_dict);
        if(ret == -1) return NULL;

        for(members = structs->members; members->name; ++members) {
            if(!(tmp = PyLong_FromLong(members->offset))) return NULL;
            ret = PyDict_SetItemString(m_dict,members->name,tmp);
            Py_DECREF(tmp);
            if(ret == -1) return NULL;
        }
    }

    
    if(!(addrs = PyDict_New())) return NULL;
    if(PyModule_AddObject(m,"raw_addresses",addrs) == -1) return NULL;

    AddrRec addr_records[] = {
        ADD_ADDR(Py_IncRef),
        ADD_ADDR(Py_DecRef),
        ADD_ADDR(Py_AddPendingCall),
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
        ADD_ADDR(PyEval_AcquireThread),
        ADD_ADDR(_PyEval_SignalAsyncExc),
        ADD_ADDR(PyErr_Occurred),
        ADD_ADDR(PyErr_ExceptionMatches),
        ADD_ADDR(PyErr_Clear),
        ADD_ADDR(PyErr_Format),
        ADD_ADDR(PyErr_SetString),
        ADD_ADDR(PyErr_Fetch),
        ADD_ADDR(PyErr_Restore),
        ADD_ADDR(PyErr_NormalizeException),
        ADD_ADDR(PyException_SetTraceback),
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
        ADD_ADDR(PyNumber_Positive),
        ADD_ADDR(PyNumber_Negative),
        ADD_ADDR(PyNumber_Invert),
        ADD_ADDR(PyNumber_Remainder),
        ADD_ADDR(PyNumber_Power),
        ADD_ADDR(PyLong_AsLong),
        ADD_ADDR(PyLong_FromLong),
        ADD_ADDR(PyList_New),
        ADD_ADDR(PyList_Append),
        ADD_ADDR(PyTuple_New),
        ADD_ADDR(PyTuple_Pack),
        ADD_ADDR(PySet_Add),
        ADD_ADDR(PySet_New),
        ADD_ADDR(PySlice_New),
        ADD_ADDR(PySequence_Contains),
        ADD_ADDR(PyTraceBack_Here),
        ADD_ADDR(PyUnicode_Format),
        ADD_ADDR(PyCell_Get),
        ADD_ADDR(PyCell_Set),
        ADD_ADDR(_EnterRecursiveCall),
        ADD_ADDR(_LeaveRecursiveCall),
        ADD_ADDR(call_function),
        ADD_ADDR(format_exc_check_arg),
        ADD_ADDR(format_exc_unbound),
        ADD_ADDR(_make_function),
        ADD_ADDR(_unpack_iterable),
        ADD_ADDR(_exception_cmp),
        ADD_ADDR(_do_raise),
        ADD_ADDR(import_all_from),
        ADD_ADDR(special_lookup),
        ADD_ADDR(call_exc_trace),
        ADD_ADDR(unicode_concatenate),
        ADD_ADDR(ext_do_call),
    
        ADD_ADDR(Py_True),
        ADD_ADDR(Py_False),
        ADD_ADDR(Py_None),
        ADD_ADDR_OF(PyDict_Type),
        ADD_ADDR_OF(PyList_Type),
        ADD_ADDR_OF(PyTuple_Type),
        ADD_ADDR_OF(PyUnicode_Type),
        ADD_ADDR_OF(PyMethod_Type),
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
        if(!(tmp = PyLong_FromUnsignedLong(addr_records[i].addr))) return NULL;
        ret = PyDict_SetItemString(addrs,addr_records[i].name,tmp);
        Py_DECREF(tmp);
        if(ret == -1) return NULL;
    }
    
    Py_INCREF(&CompiledCodeType);
    if(PyModule_AddObject(m,"CompiledCode",(PyObject*)&CompiledCodeType) == -1) return NULL;

    assert(!old_func_call);

#define backup_and_set(old,new,store) old = store; store = new

    backup_and_set(old_func_call,_function_call,PyFunction_Type.tp_call);
    backup_and_set(old_gen_send,(PyCFunction)gen_send,PyGen_Type.tp_methods[0].ml_meth);
    backup_and_set(old_gen_throw,(PyCFunction)gen_throw,PyGen_Type.tp_methods[1].ml_meth);
    backup_and_set(old_gen_close,(PyCFunction)gen_close,PyGen_Type.tp_methods[2].ml_meth);
    backup_and_set(old_gen_iternext,(iternextfunc)gen_iternext,PyGen_Type.tp_iternext);
    
    return m;
}

