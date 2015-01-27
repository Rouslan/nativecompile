/* Copyright 2015 Rouslan Korneychuk
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
#define CANNOT_IMPORT_MSG "cannot import name %R"
#define IMPORT_NOT_FOUND_MSG "__import__ not found"
#define BAD_EXCEPTION_MSG "'finally' pops bad exception"
#define UNEXPECTED_KW_ARG_MSG "%U() got an unexpected keyword argument '%S'"
#define DUPLICATE_VAL_MSG "%U() got multiple values for argument '%S'"


#define GETLOCAL(i)     (fastlocals[i])
#define SETLOCAL(i, value)      do { PyObject *tmp = GETLOCAL(i); \
                                     GETLOCAL(i) = value; \
                                     Py_XDECREF(tmp); } while (0)

#define CALL_FLAG_VAR 1
#define CALL_FLAG_KW 2


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


typedef struct {
    PyObject_HEAD

    void *data;
    Py_ssize_t size;

#ifdef GDB_JIT_SUPPORT
    unsigned char gdb_reg;
#endif
} CompiledCode;

static void CompiledCode_dealloc(CompiledCode *self) {
    if(self->data) {
#ifdef GDB_JIT_SUPPORT
        if(self->gdb_reg) unregister_gdb(self->data);
#endif

#ifdef USE_POSIX
        munmap(self->data,self->size);
#elif USE_WIN
        VirtualFree(self->data,0,MEM_RELEASE);
#else
        PyMem_Free(self->data);
#endif
    }
}

static int alloc_for_func(CompiledCode *self,long size) {
    assert(size >= 0);

    if(!size) {
        PyErr_SetString(PyExc_ValueError,"parts cannot be empty");
        return -1;
    }
    
    self->size = size;

#ifdef USE_POSIX
    if((self->data = mmap(0,size,PROT_READ|PROT_WRITE|PROT_EXEC,MAP_PRIVATE|MAP_ANON,-1,0)) == MAP_FAILED) {
    /*if(posix_memalign((void**)&self->entry,sysconf(_SC_PAGESIZE),size)) {*/
        self->data = NULL;
        PyErr_NoMemory();
        return -1;
    }
    
#elif defined(USE_WIN)
    if(!(self->data = VirtualAlloc(0,size,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE))) {
        PyErr_NoMemory();
        return -1;
    }
#else
    if(!(self->data = PyMem_Malloc(size))) {
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
    Py_ssize_t i;
    PyObject *lparts = NULL;
    PyObject *item;

#ifdef GDB_JIT_SUPPORT
    Py_buffer buff = {0};
#endif

    static char *kwlist[] = {"parts",NULL};

    if(!PyArg_ParseTupleAndKeywords(args,kwds,"O",kwlist,
        &parts)) return NULL;

    self = (CompiledCode*)type->tp_alloc(type,0);
    if(self) {
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

            if(!(item = PyObject_CallMethod(parts,"rebase","k",self->data))) goto error;
            Py_DECREF(item);

            size = buff.len;
            memcpy(self->data,buff.buf,(size_t)size);

            if(register_gdb(self->data,size)) goto error;
            self->gdb_reg = 1;
        } else
#endif
        if(PyBytes_Check(parts)) {
            size = PyBytes_GET_SIZE(parts);
            if(alloc_for_func(self,size)) goto error;
            memcpy(self->data,PyBytes_AS_STRING(parts),(size_t)size);
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

            dest = (char*)self->data;
            for(i=0; i < PyList_GET_SIZE(lparts); ++i) {
                item = PyList_GET_ITEM(lparts,i);
                memcpy(dest,PyBytes_AS_STRING(item),PyBytes_GET_SIZE(item));
                dest += PyBytes_GET_SIZE(item);
            }
        }

#ifdef USE_POSIX
        if(mprotect(self->data,size,PROT_READ|PROT_EXEC)) {
            PyErr_SetFromErrno(PyExc_OSError);
            goto error;
        }
    #ifdef __GNUC__
        __builtin___clear_cache(self->data,(char*)self->data + size);
    #else
        #warning "Don't know how to flush instruction cache on this compiler and OS"
    #endif
#elif defined(USE_WIN)
        if(!VirtualProtect(self->data,size,PAGE_EXECUTE_READ)) {
            PyErr_SetFromWindowsErr(0);
            goto error;
        }
        FlushInstructionCache(GetCurrentProcess(),self->data,size);
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
    0,                         /* tp_hash */
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
    CompiledCode_new,          /* tp_new */
};


typedef struct {
    PyObject_HEAD

    PyObject *name;
    
    /* The function parameter names are always at the beginning and in order.
       The positional parameter names are at the start, followed by the name of
       the * parameter if var_pos is 1, followed by the keyword-only parameters,
       followed by the ** parameter if var_kw if 1.
    
       All code that relies on this order (including python code) will have a
       comment containing "FUNCTION_BODY_NAME_ORDER". */
    PyObject *names;

    PyObject *free_names;
    
    PyObject *consts;

    CompiledCode *code;

    Py_ssize_t offset;
    
    int pos_params;
    int kwonly_params;
    int cells;
    char var_pos;
    char var_kw;
} FunctionBody;

static int FunctionBody_traverse(FunctionBody *self,visitproc visit,void *arg) {
    Py_VISIT(self->name);
    Py_VISIT(self->names);
    Py_VISIT(self->free_names);
    Py_VISIT(self->consts);
    Py_VISIT(self->code);
    
    return 0;
}

static int FunctionBody_clear(FunctionBody *self) {
    Py_CLEAR(self->name);
    Py_CLEAR(self->names);
    Py_CLEAR(self->free_names);
    Py_CLEAR(self->consts);
    Py_CLEAR(self->code);
    
    return 0;
}

static void FunctionBody_dealloc(FunctionBody *self) {
    FunctionBody_clear(self);
    
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* There is no reason to modify a FunctionBody instance once it's initialized,
   but initialization is done in tp_init and not tp_new so that an instance can
   be created and have an address in memory, before it is initialized. This is
   so the function's address can be hard-coded into its own code object. */
static int FunctionBody_init(FunctionBody *self,PyObject *args,PyObject *kwds) {
    CompiledCode *code;
    PyObject *name, *names, *free_names, *consts;
    int var_pos, var_kw;
    
    static char *kwlist[] = {"code","offset","name","names","pos_params","var_pos","kwonly_params","var_kw","free_names","cells","consts",NULL};
    
    if(self->code) {
        PyErr_SetString(PyExc_ValueError,"an instance of FunctionBody can only be initialized once");
        return -1;
    }

    if(!PyArg_ParseTupleAndKeywords(args,kwds,"O!nUOipipOiO",kwlist,
        &CompiledCodeType,&code,
        &self->offset,
        &name,
        &names,
        &self->pos_params,
        &var_pos,
        &self->kwonly_params,
        &var_kw,
        &free_names,
        &self->cells,
        &consts)) return -1;
    
    /* in case of prior incomplete initialization due to an error */
    FunctionBody_clear(self);
    
    self->names = PySequence_Tuple(names);
    if(!self->names) return -1;
    
    if(self->pos_params + var_pos + self->kwonly_params + var_kw > PyTuple_GET_SIZE(self->names)) {
        PyErr_SetString(PyExc_ValueError,"there cannot be more parameters than names");
        return -1;
    }
    
    self->free_names = PySequence_Tuple(free_names);
    if(!self->free_names) return -1;
    
    if(self->cells > PyTuple_GET_SIZE(self->free_names)) {
        PyErr_SetString(PyExc_ValueError,"there cannot be more cells than free names");
        return -1;
    }
    
    self->consts = PySequence_Tuple(consts);
    if(!self->consts) return -1;
    
    self->name = name;
    Py_INCREF(name);
    
    self->code = code;
    Py_INCREF(code);
    
    self->var_pos = var_pos;
    self->var_kw = var_kw;

    return 0;
}

static PyMemberDef FunctionBody_members[] = {
    {"name",T_OBJECT_EX,offsetof(FunctionBody,name),READONLY,NULL},
    {"names",T_OBJECT_EX,offsetof(FunctionBody,names),READONLY,NULL},
    {"free_names",T_OBJECT_EX,offsetof(FunctionBody,free_names),READONLY,NULL},
    {"consts",T_OBJECT_EX,offsetof(FunctionBody,consts),READONLY,NULL},
    {"code",T_OBJECT_EX,offsetof(FunctionBody,code),READONLY,NULL},
    {"pos_params",T_INT,offsetof(FunctionBody,pos_params),READONLY,NULL},
    {"kwonly_params",T_INT,offsetof(FunctionBody,kwonly_params),READONLY,NULL},
    {"cells",T_INT,offsetof(FunctionBody,cells),READONLY,NULL},
    {"var_pos",T_BOOL,offsetof(FunctionBody,var_pos),READONLY,NULL},
    {"var_kw",T_BOOL,offsetof(FunctionBody,var_kw),READONLY,NULL},
    {NULL}
};

PyDoc_STRVAR(FunctionBody_doc,
"FunctionBody(code : CompiledCode,offset : int,name : str,names,pos_params : int,var_pos : bool,kwonly_params : int,var_kw : bool,free_names,cells : int,consts)\n\
\n\
Compiled Function Body\n\
");

static PyTypeObject FunctionBodyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "nativecompile.pyinternals.FunctionBody", /* tp_name */
    sizeof(FunctionBody),      /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)FunctionBody_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
    FunctionBody_doc,          /* tp_doc */
    (traverseproc)FunctionBody_traverse, /* tp_traverse */
    (inquiry)FunctionBody_clear, /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    FunctionBody_members,      /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)FunctionBody_init, /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};


void free_pyobj_array(PyObject **array,Py_ssize_t size) {
    Py_ssize_t i;
    for(i=0; i<size; ++i) Py_XDECREF(array[i]);
    PyMem_Free(array);
}

typedef struct {
    PyObject_HEAD

    PyObject *name;
    PyObject *doc;
    PyObject *globals;
    PyObject *defaults;
    PyObject **kwdefaults;
    PyObject **closure;
    PyObject *annotations;
    PyObject *module;
    PyObject *dict;
    PyObject *weakrefs;

    FunctionBody *body;
} Function;

static PyObject *Function_call(Function *self,PyObject *args,PyObject *kwds) {
    PyFrameObject *f;
    PyCodeObject *dummy_code;
    PyThreadState *tstate = PyThreadState_GET();
    PyObject *r = NULL;


    assert(self->body && self->globals);
    
    dummy_code = PyCode_NewEmpty("<string>","generated",0);
    if(!dummy_code) return NULL;
    
    f = PyFrame_New(tstate,dummy_code,self->globals,NULL);
    Py_DECREF(dummy_code);
    
    if(f) {
        r = (*(PyObject *(*)(PyFrameObject*,Function*,PyObject*,PyObject*))(self->body->code->data + self->body->offset))(f,self,args,kwds);
        Py_DECREF(f);
    }
    
    return r;
}

#define FOR_FUNCTION_ATTR(X) \
    X(self->name); \
    X(self->doc); \
    X(self->globals); \
    X(self->defaults); \
    if(self->kwdefaults) { \
        int i; \
        for(i=0; i<self->body->kwonly_params; ++i) X(self->kwdefaults[i]); \
    } \
    if(self->closure) { \
        int i; \
        for(i=0; i<PyTuple_GET_SIZE(self->body->free_names); ++i) X(self->closure[i]); \
    } \
    X(self->annotations); \
    X(self->module); \
    X(self->dict); \
    X(self->body)

static int Function_traverse(Function *self,visitproc visit,void *arg) {
    FOR_FUNCTION_ATTR(Py_VISIT);
    
    return 0;
}

static void Function_dealloc(Function *self) {
    if(self->weakrefs) PyObject_ClearWeakRefs((PyObject*)self);
    
    FOR_FUNCTION_ATTR(Py_XDECREF);
    
    PyMem_Free(self->kwdefaults);
    PyMem_Free(self->closure);
    
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *_new_function(Function *self,FunctionBody *body,PyObject *name,PyObject *globals,PyObject *doc,PyObject *defaults,PyObject **kwdefaults,PyObject **closure,PyObject *annotations) {
    self->kwdefaults = kwdefaults;
    self->closure = closure;
    self->dict = self->weakrefs = NULL;

    if(defaults) {
        if(PyTuple_GET_SIZE(defaults) > body->pos_params) {
            PyErr_SetString(PyExc_ValueError,"there cannot be more default positional arguments than positional parameters");
            Py_DECREF(self);
            return NULL;
        }
        self->defaults = defaults;
        Py_INCREF(defaults);
    } else {
        self->defaults = NULL;
        if(body->pos_params) {
            self->defaults = PyTuple_New(0);
            if(!self->defaults) {
                Py_DECREF(self);
                return NULL;
            }
        }
    }
    
    self->body = body;
    Py_INCREF(body);
    self->name = name;
    Py_INCREF(name);
    self->globals = globals;
    Py_INCREF(globals);
    self->doc = doc;
    Py_XINCREF(doc);
    self->annotations = annotations;
    Py_XINCREF(annotations);
    
    /* TODO: store interned unicode object in this module */
    self->module = PyDict_GetItemString(globals,"__name__");
    Py_XINCREF(self->module);

    return (PyObject*)self;
}

static PyObject *Function_new(PyTypeObject *type,PyObject *args,PyObject *kwds) {
    Py_ssize_t closure_size;
    int i;
    Function *self;
    FunctionBody *body;
    PyObject **kwarray=NULL, **clarray=NULL;
    PyObject *name, *globals, *doc=NULL, *defaults=NULL, *kwdefaults=NULL, *closure=NULL, *annotations=NULL;
    
    static char *kwlist[] = {"body","name","globals","doc","defaults","kwdefaults","closure","annotations",NULL};

    if(!PyArg_ParseTupleAndKeywords(args,kwds,"O!UO!|OO!O!O!",kwlist,
            &FunctionBodyType,&body,
            &name,
            &PyDict_Type,&globals,
            &doc,
            &PyTuple_Type,&defaults,
            &PyDict_Type,&kwdefaults,
            &PyTuple_Type,&closure,
            &PyDict_Type,&annotations)) {
        return NULL;
    }
    
    if(!body->code) {
        PyErr_SetString(PyExc_ValueError,"\"body\" was not initialized");
        return NULL;
    }
    
    self = (Function*)type->tp_alloc(type,0);
    if(!self) return NULL;
    
    if(kwdefaults) {
        PyObject *k, *v;
        Py_ssize_t pos = 0;
        
        if(body->kwonly_params) {
            kwarray = PyMem_Malloc(sizeof(PyObject*) * body->kwonly_params);
        }
        
        while(PyDict_Next(kwdefaults,&pos,&k,&v)) {
            int i;
            int kwoffset = body->pos_params + body->var_pos;
            
            for(i=0; i<body->kwonly_params; ++i) {
                PyObject *kwname = PyTuple_GET_ITEM(body->names,kwoffset+i);
                int cmp = PyObject_RichCompareBool(kwname,k,Py_EQ);
                
                if(cmp < 0) goto fail;
                if(cmp) {
                    kwarray[i] = v;
                    Py_INCREF(v);
                    goto next;
                }
            }
            
            PyErr_Format(PyExc_ValueError,"\"body\" does not have a keyword-only parameter named \"%S\"",v);
            goto fail;
            
        next:
            ;
        }
    }
    
    closure_size = 0;
    if(closure) closure_size = PyTuple_GET_SIZE(closure);
    if(PyTuple_GET_SIZE(body->free_names) != closure_size) {
        if(PyTuple_GET_SIZE(body->free_names))
            PyErr_Format(PyExc_ValueError,"\"closure\" must have exactly %d elements",PyTuple_GET_SIZE(body->free_names));
        else
            PyErr_SetString(PyExc_ValueError,"\"body\" does not have any free variables (\"closure\" should be empty)");
        goto fail;
    }
    if(closure) {
        for(i=0; i<body->cells; ++i) {
            if(!PyCell_Check(PyTuple_GET_ITEM(closure,i))) {
                PyErr_Format(PyExc_TypeError,"the first %d elements of \"closure\" must be cells");
                goto fail;
            }
        }
        
        clarray = PyMem_Malloc(sizeof(PyObject*) * closure_size);
        for(i=0; i<closure_size; ++i) {
            clarray[i] = PyTuple_GET_ITEM(closure,i);
            Py_INCREF(clarray[i]);
        }
    }
    
    return _new_function(self,body,name,globals,doc,defaults,kwarray,clarray,annotations);
    
fail:
    if(kwarray) free_pyobj_array(kwarray,body->kwonly_params);
    if(clarray) free_pyobj_array(clarray,PyTuple_GET_SIZE(body->free_names));
    Py_DECREF(self);
    return NULL;
}

static PyObject *Function_descr_get(PyObject *self,PyObject *obj,PyObject *type) {
    if(!obj || obj == Py_None) {
        Py_INCREF(self);
        return self;
    }
    return PyMethod_New(self,obj);
}

static PyMemberDef Function_members[] = {
    {"__name__",T_OBJECT_EX,offsetof(Function,name),READONLY,NULL},
    {"__doc__",T_OBJECT_EX,offsetof(Function,doc),READONLY,NULL},
    {"defaults",T_OBJECT_EX,offsetof(Function,defaults),READONLY,NULL},
    {"annotations",T_OBJECT_EX,offsetof(Function,annotations),READONLY,NULL},
    {"__module__",T_OBJECT_EX,offsetof(Function,module),READONLY,NULL},
    {"body",T_OBJECT_EX,offsetof(Function,body),READONLY,NULL},
    {NULL}
};

PyDoc_STRVAR(Function_doc,
"Function(body : FunctionBody,name : str,globals : dict[,doc,defaults : tuple,kwdefaults : dict,closure : tuple,annotations : dict])\n\
\n\
Compiled Function\n\
");

static PyTypeObject FunctionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "nativecompile.pyinternals.Function", /* tp_name */
    sizeof(Function),          /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)Function_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    (ternaryfunc)Function_call, /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
    Function_doc,              /* tp_doc */
    (traverseproc)Function_traverse, /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    offsetof(Function,weakrefs), /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    Function_members,          /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    Function_descr_get,        /* tp_descr_get */
    0,                         /* tp_descr_set */
    offsetof(Function,dict),   /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    Function_new,              /* tp_new */
};

static PyObject *new_function(FunctionBody *body,PyObject *name,PyObject *globals,PyObject *doc,PyObject *defaults,PyObject **kwdefaults,PyObject **closure,PyObject *annotations) {
    Function *self = PyObject_GC_New(Function,&FunctionType);
    if(!self) return NULL;
    return _new_function(self,body,name,globals,doc,defaults,kwdefaults,closure,annotations);
}



PyDoc_STRVAR(read_address_doc,
"read_address(address : int,length : int) -> bytes\n\
\n\
Get the contents of memory at an arbitary address.\n\
\n\
Warning: this function is very unsafe. Improper use can easily cause a\n\
segmentation fault.");

static PyObject *read_address(PyObject *self,PyObject *args) {
    unsigned long addr;
    Py_ssize_t length;

    if(!PyArg_ParseTuple(args,"kn:read_address",&addr,&length)) return NULL;

    return PyBytes_FromStringAndSize((const char*)addr,length);
}


PyDoc_STRVAR(create_cell_doc,
"create_cell([value]) -> cell\n\
\n\
Create a cell.\n\
");

static PyObject *create_cell(PyObject *self,PyObject *args) {
    PyObject *value=NULL;

    if(!PyArg_ParseTuple(args,"|O:create_cell",&value)) return NULL;

    return PyCell_New(value);
}


/* This function is based on ext_do_call from Python/ceval.c. This steals a
   reference to args if it's not NULL. */
static PyObject *append_tuple_for_call(PyObject *func,PyObject *args,PyObject *star) {
    Py_ssize_t i, oldlen;
    PyObject *tmp = NULL;
    
    if(!PyTuple_Check(star)) {
        tmp = PySequence_Tuple(star);
        if(!tmp) {
            if(PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Format(PyExc_TypeError,
                    "%.200s%.200s argument after * must be a sequence, not %.200s",
                    PyEval_GetFuncName(func),
                    PyEval_GetFuncDesc(func),
                    star->ob_type->tp_name);
            }
            Py_XDECREF(args);
            return NULL;
        }
        star = tmp;
    }
    
    if(args) {
        oldlen = PyTuple_GET_SIZE(args);
        if(_PyTuple_Resize(&args,oldlen + PyTuple_GET_SIZE(star))) {
            Py_DECREF(args);
            args = NULL;
        } else {
            for(i=0; i<PyTuple_GET_SIZE(star); ++i) {
                PyTuple_SET_ITEM(args,oldlen + i,PyTuple_GET_ITEM(star,i));
                Py_INCREF(PyTuple_GET_ITEM(star,i));
            }
        }
    } else  {
        Py_INCREF(star);
        args = star;
    }
    
    Py_XDECREF(tmp);
    return args;
}

/* This function is based on ext_do_call and update_keyword_args from
   Python/ceval.c. This steals a reference to args if it's not NULL. */
static PyObject *append_dict_for_call(PyObject *func,PyObject *args,PyObject *star) {
    PyObject *k, *v;
    PyObject *tmp = NULL;
    Py_ssize_t pos = 0;
    
    if(!PyDict_Check(star)) {
        if(!(tmp = PyDict_New())) {
            Py_XDECREF(args);
            return NULL;
        }
        
        if(PyDict_Update(tmp,star)) {
            if(PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Format(PyExc_TypeError,
                    "%.200s%.200s argument after * must be a mapping, not %.200s",
                    PyEval_GetFuncName(func),
                    PyEval_GetFuncDesc(func),
                    star->ob_type->tp_name);
            }
            Py_XDECREF(args);
            return NULL;
        }
        star = tmp;
    }
    
    if(args) {
        while(PyDict_Next(star,&pos,&k,&v)) {
            if(PyDict_GetItem(args,k)) {
                PyErr_Format(PyExc_TypeError,
                    "%.200s%s got multiple values for keyword argument '%U'",
                    PyEval_GetFuncName(func),
                    PyEval_GetFuncDesc(func),
                    k);
                Py_DECREF(args);
                args = NULL;
                break;
            }
            if(PyDict_SetItem(args,k,v)) {
                Py_DECREF(args);
                args = NULL;
                break;
            }
        }
    } else {
        Py_INCREF(star);
        args = star;
    }
    
    Py_XDECREF(tmp);
    return args;
}

static void too_many_positional(Function *f,long excess,PyObject *kwds) {
    PyObject *sig, *kw_sig;
    int plural, given, start, i;
    int kwonly = 0;
    
    assert(!f->body->var_pos);
    
    if(PyTuple_GET_SIZE(f->defaults)) {
        sig = PyUnicode_FromFormat(
            "from %d to %d",
            f->body->pos_params - PyTuple_GET_SIZE(f->defaults),
            f->body->pos_params);
        plural = 1;
    } else {
        sig = PyUnicode_FromFormat("%d",f->body->pos_params);
        plural = f->body->pos_params != 1;
    }
    if(!sig) return;
    
    start = f->body->pos_params + f->body->var_pos;
    for(i=start; i<start+f->body->kwonly_params; ++i) {
        int c = PyDict_Contains(kwds,PyTuple_GET_ITEM(f->body->names,i));
        if(c < 0) return;
        kwonly += c;
    }
    
    given = f->body->pos_params + excess;
    
    if(kwonly)
        kw_sig = PyUnicode_FromFormat(
            " positional argument%s (and %d keyword-only argument%s)",
            given != 1 ? "s" : "",
            kwonly,
            kwonly != 1 ? "s" : "");
    else
        kw_sig = PyUnicode_FromString("");
    
    if(!kw_sig) {
        Py_DECREF(sig);
        return;
    }
    
    PyErr_Format(PyExc_TypeError,
        "%U() takes %U positional argument%s but %d%U %s given",
        f->name,
        sig,
        plural ? "s" : "",
        given,
        kw_sig,
        given == 1 && !kwonly ? "was" : "were");
    
    Py_DECREF(kw_sig);
    Py_DECREF(sig);
}

static void excess_keyword(Function *f,PyObject *kwds) {
    PyObject *k, *v;
    int i, r;
    Py_ssize_t pos = 0;
    
    while(PyDict_Next(kwds,&pos,&k,&v)) {
        for(i=0; i<f->body->pos_params+f->body->var_pos+f->body->kwonly_params; ++i) {
            if(f->body->var_pos && i == f->body->pos_params) continue;
            r = PyObject_RichCompareBool(k,PyTuple_GET_ITEM(f->body->names,i),Py_EQ);
            if(r < 0) return;
            if(r) goto keep_looking;
        }
        
        PyErr_Format(
            PyExc_TypeError,
            "%U() got an unexpected keyword argument '%S'",
            f->name,
            k);
        return;
        
    keep_looking:
        ;
    }
    assert(0);
}

/* Note: we report missing arguments slightly differently from CPython. CPython
   first checks positional arguments and then keyword-only arguments and
   mentions whether the arguments are positional or keyword-only in the error
   message. Our code will report all missing arguments together, and not mention
   which kind they are (since the list could have both). */
static void missing_arguments(Function *f,PyObject **locals) {
    int i, len;
    PyObject *name_str, *comma, *tail, *tmp;
    
    int n_count = f->body->pos_params + f->body->var_pos + f->body->kwonly_params;
    PyObject *names = PyList_New(0);
    if(!names) return;
    
    for(i=0; i<n_count; ++i) {
        /* FUNCTION_BODY_NAME_ORDER
           it doesn't matter if the "*" parameter is set */
        if(!locals[i] && !(f->body->var_pos && i == f->body->pos_params)) {
            int r;
            
            PyObject *name = PyObject_Repr(PyTuple_GET_ITEM(f->body->names,i));
            if(!name) {
                Py_DECREF(names);
                return;
            }
            
            r = PyList_Append(names,name);
            Py_DECREF(name);
            if(r < 0) {
                Py_DECREF(names);
                return;
            }
        }
    }
    
    /* the rest of the code here is adapted from format_missing from
       Python/ceval.c */
    
    len = PyList_GET_SIZE(names);
    assert(len);

    switch(len) {
    case 1:
        name_str = PyList_GET_ITEM(names,0);
        Py_INCREF(name_str);
        break;
    case 2:
        name_str = PyUnicode_FromFormat(
            "%U and %U",
            PyList_GET_ITEM(names,0),
            PyList_GET_ITEM(names,1));
        break;
    default:
        name_str = NULL;

        if(!(tail = PyUnicode_FromFormat(
            ", %U, and %U",
            PyList_GET_ITEM(names,len - 2),
            PyList_GET_ITEM(names,len - 1)))) break;

        if(PyList_SetSlice(names,len - 2,len,NULL) < 0) {
            Py_DECREF(tail);
            break;
        }
        
        if(!(comma = PyUnicode_FromString(", "))) {
            Py_DECREF(tail);
            break;
        }
        
        tmp = PyUnicode_Join(comma,names);
        Py_DECREF(comma);
        if(!tmp) {
            Py_DECREF(tail);
            break;
        }
        
        name_str = PyUnicode_Concat(tmp,tail);
        Py_DECREF(tmp);
        Py_DECREF(tail);
        break;
    }
    
    Py_DECREF(names);
    
    if(!name_str) return;
    
    PyErr_Format(
        PyExc_TypeError,
        "%U() missing %i required argument%s: %U",
        f->name,
        len,
        len == 1 ? "" : "s",
        name_str);
    Py_DECREF(name_str);
}



/* The following are copied from Python/ceval.c */

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
        if (!PyExceptionInstance_Check(value)) {
            PyErr_Format(PyExc_TypeError,
                         "calling %R should have returned an instance of "
                         "BaseException, not %R",
                         type, Py_TYPE(value));
            goto raise_error;
        }
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
        else if (cause == Py_None) {
            Py_DECREF(cause);
            fixed_cause = NULL;
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

static int
import_all_from(PyObject *locals, PyObject *v)
{
    _Py_IDENTIFIER(__all__);
    _Py_IDENTIFIER(__dict__);
    PyObject *all = _PyObject_GetAttrId(v, &PyId___all__);
    PyObject *dict, *name, *value;
    int skip_leading_underscores = 0;
    int pos, err;

    if (all == NULL) {
        if (!PyErr_ExceptionMatches(PyExc_AttributeError))
            return -1;
        PyErr_Clear();
        dict = _PyObject_GetAttrId(v, &PyId___dict__);
        if (dict == NULL) {
            if (!PyErr_ExceptionMatches(PyExc_AttributeError))
                return -1;
            PyErr_SetString(PyExc_ImportError,
            "from-import-* object has no __dict__ and no __all__");
            return -1;
        }
        all = PyMapping_Keys(dict);
        Py_DECREF(dict);
        if (all == NULL)
            return -1;
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
            PyUnicode_READY(name) != -1 &&
            PyUnicode_READ_CHAR(name, 0) == '_')
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
    return err;
}

static PyObject *
special_lookup(PyObject *o, _Py_Identifier *id)
{
    PyObject *res;
    res = _PyObject_LookupSpecial(o, id);
    if (res == NULL && !PyErr_Occurred()) {
        PyErr_SetObject(PyExc_AttributeError, id->object);
        return NULL;
    }
    return res;
}

static int
call_trace(Py_tracefunc func, PyObject *obj,
           PyThreadState *tstate, PyFrameObject *frame,
           int what, PyObject *arg)
{
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
call_trace_protected(Py_tracefunc func, PyObject *obj,
                     PyThreadState *tstate, PyFrameObject *frame,
                     int what, PyObject *arg)
{
    PyObject *type, *value, *traceback;
    int err;
    PyErr_Fetch(&type, &value, &traceback);
    err = call_trace(func, obj, tstate, frame, what, arg);
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
call_exc_trace(Py_tracefunc func, PyObject *self,
               PyThreadState *tstate, PyFrameObject *f)
{
    PyObject *type, *value, *traceback, *orig_traceback, *arg;
    int err;
    PyErr_Fetch(&type, &value, &orig_traceback);
    if (value == NULL) {
        value = Py_None;
        Py_INCREF(value);
    }
    PyErr_NormalizeException(&type, &value, &orig_traceback);
    traceback = (orig_traceback != NULL) ? orig_traceback : Py_None;
    arg = PyTuple_Pack(3, type, value, traceback);
    if (arg == NULL) {
        PyErr_Restore(type, value, orig_traceback);
        return;
    }
    err = call_trace(func, self, tstate, f, PyTrace_EXCEPTION, arg);
    Py_DECREF(arg);
    if (err == 0)
        PyErr_Restore(type, value, orig_traceback);
    else {
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(orig_traceback);
    }
}

static PyObject *
unicode_concatenate(PyObject *v, PyObject *w, PyFrameObject *f,
                    unsigned char next_instr, int instr_arg)
{
    PyObject *res;
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
    res = v;
    PyUnicode_Append(&res, w);
    return res;
}

static int _print_expr(PyObject *expr) {
    PyObject *args, *hook, *eval_ret;
    int ret = 0;

    if(!(hook = PySys_GetObject("displayhook"))) {
        PyErr_SetString(PyExc_RuntimeError,"lost sys.displayhook");
        ret = -1;
        goto err1;
    }

    if(!(args = PyTuple_Pack(1, expr))) {
        ret = -1;
        goto err1;
    }

    if(!(eval_ret = PyEval_CallObject(hook, args))) {
        ret = -1;
        goto err2;
    }

    Py_DECREF(eval_ret);
err2:
    Py_DECREF(args);
err1:
    Py_DECREF(expr);
    return ret;
}

static PyObject *_load_build_class(PyObject *f_builtins) {
    PyObject *x;
    _Py_IDENTIFIER(__build_class__);

    if (PyDict_CheckExact(f_builtins)) {
        x = _PyDict_GetItemId(f_builtins, &PyId___build_class__);
        if (x == NULL) {
            PyErr_SetString(PyExc_NameError,
                            "__build_class__ not found");
            return NULL;
        }
        Py_INCREF(x);
    }
    else {
        PyObject *build_class_str = _PyUnicode_FromId(&PyId___build_class__);
        if (build_class_str == NULL)
            return NULL;
        x = PyObject_GetItem(f_builtins, build_class_str);
        if (x == NULL) {
            if (PyErr_ExceptionMatches(PyExc_KeyError))
                PyErr_SetString(PyExc_NameError,
                                "__build_class__ not found");
            return NULL;
        }
    }
    return x;
}

/* end of functions copied from Python/ceval.c */



/* Py_EnterRecursiveCall and Py_LeaveRecursiveCall are somewhat complicated
 * macros so they are wrapped in the following two functions */

static int _EnterRecursiveCall(void) {
    return Py_EnterRecursiveCall("");
}

static void _LeaveRecursiveCall(void) {
    Py_LeaveRecursiveCall();
}



_Py_IDENTIFIER(__import__);
_Py_IDENTIFIER(__enter__);
_Py_IDENTIFIER(__exit__);
_Py_IDENTIFIER(send);



static PyMethodDef functions[] = {
    {"read_address",read_address,METH_VARARGS,read_address_doc},
    {"create_cell",create_cell,METH_VARARGS,create_cell_doc},
    {NULL}
};

static struct PyModuleDef this_module = {
    PyModuleDef_HEAD_INIT,
    "pyinternals",
    NULL,
    0,
    functions,
    NULL,
    NULL,
    NULL,
    NULL
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
            M_OFFSET(PyFrameObject,f_code),
            M_OFFSET(PyFrameObject,f_builtins),
            M_OFFSET(PyFrameObject,f_globals),
            M_OFFSET(PyFrameObject,f_locals),
            M_OFFSET(PyFrameObject,f_localsplus),
            M_OFFSET(PyFrameObject,f_valuestack),
            M_OFFSET(PyFrameObject,f_stacktop),
            M_OFFSET(PyFrameObject,f_trace),
            M_OFFSET(PyFrameObject,f_exc_type),
            M_OFFSET(PyFrameObject,f_exc_value),
            M_OFFSET(PyFrameObject,f_exc_traceback),
            M_OFFSET(PyFrameObject,f_lasti),
            {NULL}}},
    {"PyThreadState",(OffsetMember[]){
            M_OFFSET(PyThreadState,frame),
            M_OFFSET(PyThreadState,curexc_traceback),
            M_OFFSET(PyThreadState,curexc_value),
            M_OFFSET(PyThreadState,curexc_type),
            M_OFFSET(PyThreadState,exc_traceback),
            M_OFFSET(PyThreadState,exc_value),
            M_OFFSET(PyThreadState,exc_type),
            M_OFFSET(PyThreadState,c_tracefunc),
            M_OFFSET(PyThreadState,c_traceobj),
            M_OFFSET(PyThreadState,dict),
            M_OFFSET(PyThreadState,async_exc),
            M_OFFSET(PyThreadState,thread_id),
            {NULL}}},
    {"PyCellObject",(OffsetMember[]){M_OFFSET(PyCellObject,ob_ref),{NULL}}},
    {"PyMethodObject",(OffsetMember[]){
            M_OFFSET(PyMethodObject,im_self),
            M_OFFSET(PyMethodObject,im_func),
            {NULL}}},
    {"Function",(OffsetMember[]){
            M_OFFSET(Function,name),
            M_OFFSET(Function,defaults),
            M_OFFSET(Function,kwdefaults),
            M_OFFSET(Function,closure),
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
    
    FunctionBodyType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&FunctionBodyType) < 0) return NULL;
    
    if(PyType_Ready(&FunctionType) < 0) return NULL;

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
        ADD_ADDR(PyMem_Malloc),
        ADD_ADDR(Py_IncRef),
        ADD_ADDR(Py_DecRef),
        ADD_ADDR(Py_AddPendingCall),
        ADD_ADDR(PyDict_GetItem),
        ADD_ADDR(PyDict_SetItem),
        ADD_ADDR(PyDict_GetItemString),
        ADD_ADDR(PyDict_Size),
        ADD_ADDR(PyDict_Copy),
        ADD_ADDR(PyDict_New),
        ADD_ADDR(_PyDict_NewPresized),
        ADD_ADDR(_PyDict_LoadGlobal),
        ADD_ADDR(_PyDict_GetItemId),
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
        ADD_ADDR(_PyObject_CallMethodId),
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
        ADD_ADDR(_PyGen_FetchStopIterationValue),
        ADD_ADDR(_EnterRecursiveCall),
        ADD_ADDR(_LeaveRecursiveCall),
        
        ADD_ADDR(new_function),
        ADD_ADDR(free_pyobj_array),
        ADD_ADDR(missing_arguments),
        ADD_ADDR(too_many_positional),
        ADD_ADDR(excess_keyword),
        ADD_ADDR(append_tuple_for_call),
        ADD_ADDR(append_dict_for_call),
        ADD_ADDR(format_exc_check_arg),
        ADD_ADDR(format_exc_unbound),
        ADD_ADDR(_unpack_iterable),
        ADD_ADDR(_exception_cmp),
        ADD_ADDR(_do_raise),
        ADD_ADDR(import_all_from),
        ADD_ADDR(special_lookup),
        ADD_ADDR(call_exc_trace),
        ADD_ADDR(unicode_concatenate),
        ADD_ADDR(_print_expr),
        ADD_ADDR(_load_build_class),

        ADD_ADDR(Py_True),
        ADD_ADDR(Py_False),
        ADD_ADDR(Py_None),
        ADD_ADDR_OF(PyDict_Type),
        ADD_ADDR_OF(PyList_Type),
        ADD_ADDR_OF(PyTuple_Type),
        ADD_ADDR_OF(PyUnicode_Type),
        ADD_ADDR_OF(PyMethod_Type),
        ADD_ADDR_OF(PyGen_Type),
        ADD_ADDR(PyExc_KeyError),
        ADD_ADDR(PyExc_NameError),
        ADD_ADDR(PyExc_StopIteration),
        ADD_ADDR(PyExc_UnboundLocalError),
        ADD_ADDR(PyExc_SystemError),
        ADD_ADDR(PyExc_ImportError),
        ADD_ADDR(PyExc_AttributeError),
        ADD_ADDR(PyExc_TypeError),
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
        ADD_ADDR(UNEXPECTED_KW_ARG_MSG),
        ADD_ADDR(DUPLICATE_VAL_MSG),
        ADD_ADDR_STR("__build_class__"),
        ADD_ADDR_OF(PyId___import__),
        ADD_ADDR_OF(PyId___exit__),
        ADD_ADDR_OF(PyId___enter__),
        ADD_ADDR_OF(PyId_send)
    };

    for(i=0; i<sizeof(addr_records)/sizeof(AddrRec); ++i) {
        if(!(tmp = PyLong_FromUnsignedLong(addr_records[i].addr))) return NULL;
        ret = PyDict_SetItemString(addrs,addr_records[i].name,tmp);
        Py_DECREF(tmp);
        if(ret == -1) return NULL;
    }

    Py_INCREF(&CompiledCodeType);
    if(PyModule_AddObject(m,"CompiledCode",(PyObject*)&CompiledCodeType) == -1) return NULL;
    
    Py_INCREF(&FunctionBodyType);
    if(PyModule_AddObject(m,"FunctionBody",(PyObject*)&FunctionBodyType) == -1) return NULL;
    
    Py_INCREF(&FunctionType);
    if(PyModule_AddObject(m,"Function",(PyObject*)&FunctionType) == -1) return NULL;

    return m;
}

