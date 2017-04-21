/* Copyright 2017 Rouslan Korneychuk
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

#ifndef PYINTERNALS_H
#define PYINTERNALS_H

#include <Python.h>


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


typedef struct {
    PyObject_HEAD

    void *data;
    Py_ssize_t size;

#ifdef GDB_JIT_SUPPORT
    unsigned char gdb_reg;
#endif
} CompiledCode;

struct _Function;
typedef PyObject *(*function_entry)(PyFrameObject*,struct _Function*,PyObject*,PyObject*);

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

    function_entry entry;

    int pos_params;
    int kwonly_params;
    int cells;
    char var_pos;
    char var_kw;
} FunctionBody;

typedef struct _Function {
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

enum generator_state {
    GEN_INITIAL,
    GEN_RUNNING,
    GEN_PAUSED,
    GEN_FINISHED
} state;

typedef struct {
    PyObject_HEAD

    size_t stack_size;
    void **stack;
    enum generator_state state;
    size_t offset;
    PyObject *name;
    PyObject **closure;
    FunctionBody *body;
    PyObject *sub_generator;
    PyObject *weakrefs;
} Generator;

#endif
