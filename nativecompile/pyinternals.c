
#include <Python.h>
#include <structmember.h>

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


/* implements the CALL_FUNCTION bytecode, except using the real stack */
static PyObject *_call_function(unsigned int arg,...) {
    va_list stack;
    PyObject *kw;
    PyObject *pos;
    PyObject *func;
    PyObject *val;
    PyObject *key;
    int r;
    int titems;
    int pos_count = arg & 0xFF;
    int kw_count = (arg & 0xFF00) >> 8;
    kw = NULL;
    
    va_start(stack,arg);
    
    if(kw_count) {
        if(!(kw = PyDict_New())) goto err;
        
        while(kw_count--) {
            val = va_arg(stack,PyObject*);
            key = va_arg(stack,PyObject*);
            r = PyDict_SetItem(kw,key,val);
            Py_DECREF(val);
            Py_DECREF(key);
            if(r == -1) {
                Py_DECREF(kw);
                goto err;
            }
        }
    }
    
    if(!(pos = PyTuple_New(pos_count))) {
        Py_XDECREF(kw);
        goto err;
    }
    
    while(pos_count--) {
        val = va_arg(stack,PyObject*);
        Py_INCREF(val);
        PyTuple_SET_ITEM(pos,pos_count,val);
    }
    
    func = va_arg(stack,PyObject*);
    
    va_end(stack);
    
    val = PyObject_Call(func,pos,kw);
    
    Py_XDECREF(kw);
    Py_DECREF(pos);
    
    return val;

err:
    titems = kw_count * 2 + pos_count + 1;
    for(;kw_count > 0; --kw_count) {
        val = va_arg(stack,PyObject*);
        Py_DECREF(val);
    }
    
    va_end(stack);
    
    return NULL;
}

/* copied from Python/ceval.c */
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


typedef struct {
    PyObject_HEAD
    
    /* The code object is not used here but may be referenced by the machine
    code in 'entry', so it needs to be kept from being destroyed */
    PyObject *code;
    
    PyObject *(*entry)(void);
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

    if(!PyArg_ParseTupleAndKeywords(args,kwds,"O&O",kwlist,PyUnicode_FSConverter,&filename_o,&code)) return NULL;
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
        self->entry = (PyObject *(*)(void))mem;

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
    return self->entry();
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
    0,		                   /* tp_traverse */
    0,		                   /* tp_clear */
    0,		                   /* tp_richcompare */
    0,		                   /* tp_weaklistoffset */
    0,		                   /* tp_iter */
    0,		                   /* tp_iternext */
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


/*
PyDoc_STRVAR(execute_machine_code_doc,
"execute_machine_code(path) -> None\n\
\n\
Execute machine code from the file at path. \n\
\n\
The file must not have any sort of header and will be executed as-is. The code\n\
is expected to start with a function that takes no arguments and return a\n\
Python object (PyObject*), or NULL if an exception is set.\n\
\n\
Warning: this function is not even\n\
remotely safe. The code has to use the correct instruction-set for the current\n\
CPU and use the same calling convention that Python was compiled with. Any\n\
mistake will most likely cause Python to immediately terminate.\n\
");*/



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

    m = PyModule_Create(&this_module);
    if(PyModule_AddIntConstant(m,"refcnt_offset",offsetof(PyObject,ob_refcnt)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"type_offset",offsetof(PyObject,ob_type)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"type_dealloc_offset",offsetof(PyTypeObject,tp_dealloc)) == -1) return NULL;
    if(PyModule_AddIntConstant(m,"type_iternext_offset",offsetof(PyTypeObject,tp_iternext)) == -1) return NULL;
    if(PyModule_AddStringConstant(m,"architecture",ARCHITECTURE) == -1) return NULL;
    if(PyModule_AddObject(m,"ref_debug",PyBool_FromLong(REF_DEBUG_VAL)) == -1) return NULL;
    if(PyModule_AddObject(m,"count_allocs",PyBool_FromLong(COUNT_ALLOCS_VAL)) == -1) return NULL;
    
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
    ADD_ADDR(PyNumber_InPlaceSubtract)
    ADD_ADDR(PyNumber_InPlaceLshift)
    ADD_ADDR(PyNumber_InPlaceRshift)
    ADD_ADDR(PyNumber_InPlaceAnd)
    ADD_ADDR(PyNumber_InPlaceXor)
    ADD_ADDR(PyNumber_InPlaceOr)
    ADD_ADDR(_call_function)
    ADD_ADDR(format_exc_check_arg)
    
    ADD_ADDR_NAME(&PyDict_Type,"PyDict_Type")
    ADD_ADDR(PyExc_KeyError)
    ADD_ADDR(PyExc_NameError)
    ADD_ADDR(PyExc_StopIteration)
    ADD_ADDR(NAME_ERROR_MSG)
    ADD_ADDR(GLOBAL_NAME_ERROR_MSG)
    ADD_ADDR(UNBOUNDLOCAL_ERROR_MSG)
    ADD_ADDR(UNBOUNDFREE_ERROR_MSG)
    
    Py_INCREF(&CompiledCodeType);
    if(PyModule_AddObject(m,"CompiledCode",(PyObject*)&CompiledCodeType) == -1) return NULL;
    
    return m;
}

