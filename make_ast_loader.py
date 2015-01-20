#  Copyright 2015 Rouslan Korneychuk
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


"""Unfortunately, the CPython function to convert an AST tree into a Python
object does not preserve the addresses of the original objects, which we need
in order to match AST nodes to symbol table namespaces, so we create our own
version that adds "_raw_id" attributes containing the original addresses.

"""

import sys
import os.path

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,os.path.join(base_dir,'cpython'))

import asdl
import asdl_c


MOD_NAME = 'astloader'


class PyTypesDeclareVisitor(asdl_c.PickleVisitor):
    def type_def(self,name):
        self.emit("static PyTypeObject *{}_type = NULL;".format(name),0)
    
    def emit_identifiers(self,items):
        for item in items:
            self.emit_identifier(item.name)
    
    def visitProduct(self,prod,name):
        self.type_def(name)
        self.emit("static PyObject* ast2obj_{}(void*);".format(name),0)
        self.emit_identifiers(prod.fields)

    def visitSum(self,sum,name):
        self.emit_identifiers(sum.attributes)
        
        ptype = "void*"
        if asdl_c.is_simple(sum):
            ptype = asdl_c.get_c_type(name)
            self.emit("static PyObject *{};".format(
                ", *".join(str(t.name)+"_singleton=NULL" for t in sum.types)),0)
        else:
            for t in sum.types:
                self.visitConstructor(t,name)
        
        self.emit("static PyObject* ast2obj_{}({});".format(name,ptype),0)

    def visitConstructor(self,cons,name):
        self.type_def(cons.name)
        self.emit_identifiers(cons.fields)

class ObjVisitor(asdl_c.ObjVisitor):
    def func_end(self):
        self.emit('value = PyLong_FromVoidPtr(_o);',1)
        self.emit('if(!value || _PyObject_SetAttrId(result,&PyId__raw_id,value) < 0) goto failed;',1)
        self.emit('Py_DECREF(value);',1)
        
        super().func_end()

class ConcreteTypeVisitor(asdl_c.PickleVisitor):
    def visitModule(self,mod):
        for d in mod.dfns:
            self.visit(d)
    
    def visitProduct(self,prod,name):
        self.process(name,False)
    
    def visitSum(self,sum,name):
        simple = asdl_c.is_simple(sum)
        for t in sum.types:
            self.process(t.name,simple)

class CleanupVisitor(ConcreteTypeVisitor):
    def visitModule(self,mod):
        self.emit('static void cleanup(void *ptr) {',0)
        super().visitModule(mod)
        self.emit('}',0)
    
    def process(self,name,simple):
        self.emit('Py_XDECREF({}_{});'.format(name,('type','singleton')[simple]),1)

class ModuleVisitor(ConcreteTypeVisitor):
    def visitModule(self,mod):
        self.emit(r"""
PyDoc_STRVAR(symtable_doc,
"symtable(code,filename,compile_type) -> tuple\n\
\n\
Return the abstract syntax tree and a mapping of AST nodes to symbol tables for\n\
the Python source \"code\".\n\
\n\
The return value is a 2-item tuple containing the an instance of _ast.AST and\n\
dict object mapping AST node addresses to namespaces, respectively. Non-trivial\n\
AST nodes get an extra parameter named \"_raw_id\" which holds the address of the\n\
original internal object, to which the symtable dict maps to.\n\
\n\
The input is the same as to the built-in function symtable.symtable.");

static PyObject *symtable(PyObject *self,PyObject *args) {{
    struct symtable *st;
    PyObject *tup, *ast, *tab;
    mod_ty mod;
    PyCompilerFlags flags;
    PyArena *arena;
    char *str, *filename, *startstr;
    int start;

    if(!PyArg_ParseTuple(args,"sss:symtable",&str,&filename,&startstr))
        return NULL;
    
    if(strcmp(startstr, "exec") == 0)
        start = Py_file_input;
    else if(strcmp(startstr, "eval") == 0)
        start = Py_eval_input;
    else if(strcmp(startstr, "single") == 0)
        start = Py_single_input;
    else {{
        PyErr_SetString(
            PyExc_ValueError,
            "symtable() arg 3 must be 'exec', 'eval' or 'single'");
        return NULL;
    }}
    
    if(!(arena = PyArena_New())) return NULL;
    
    flags.cf_flags = 0;
    if(!(mod = PyParser_ASTFromString(str,filename,start,&flags,arena))) {{
        PyArena_Free(arena);
        return NULL;
    }}
    
    if(!(st = PySymtable_Build(mod,filename,NULL))) {{
        PyArena_Free(arena);
        return NULL;
    }}
    
    ast = ast2obj_mod(mod);
    
    PyArena_Free(arena);
    
    tab = st->st_blocks;
    Py_INCREF(tab);
    PyMem_Free(st->st_future);
    PySymtable_Free(st);
    
    if(!ast) {{
        Py_DECREF(tab);
        return NULL;
    }}
    
    tup = PyTuple_Pack(2,ast,tab);
    Py_DECREF(ast);
    Py_DECREF(tab);

    return tup;
}}

static PyMethodDef functions[] = {{
    {{"symtable",symtable,METH_VARARGS,symtable_doc}},
    {{NULL}}
}};

static PyModuleDef moddef = {{
    PyModuleDef_HEAD_INIT,
    "{0}",
    NULL,
    0,
    functions,
    NULL,
    NULL,
    NULL,
    cleanup
}};

PyTypeObject *get_type(PyObject *mod,const char *name) {{
    PyObject *r = PyObject_GetAttrString(mod,name);
    if(!r) {{
        cleanup(NULL);
        return NULL;
    }}
    if(!PyType_Check(r)) {{
        Py_DECREF(r);
        PyErr_Format(PyExc_TypeError,"_ast.%s is supposed to be a class",name);
        cleanup(NULL);
        return NULL;
    }}
    return (PyTypeObject*)r;
}}
PyObject *get_singleton(PyObject *mod,const char *name) {{
    PyObject *r;
    PyTypeObject *t = get_type(mod,name);
    if(!t) return NULL;

    r = PyType_GenericNew(t,NULL,NULL);
    Py_DECREF(t);
    if(!r) {{
        cleanup(NULL);
        return NULL;
    }}
    return r;
}}

PyMODINIT_FUNC PyInit_{0}(void) {{
    PyObject *m, *astmod;
    
    m = NULL;
    
    if(!(astmod = PyImport_ImportModule("_ast"))) return NULL;
""".format(MOD_NAME),0,reflow=False)
        super().visitModule(mod)
        self.emit("""
    m = PyModule_Create(&moddef);
end:
    Py_DECREF(astmod);
    return m;
}
""",0,reflow=False)
    
    def process(self,name,simple):
        self.emit('    if(!({0}_{1} = get_{1}(astmod,"{0}"))) goto end;'.format(
            name,
            ('type','singleton')[simple]),0,reflow=False)

def create(infile,outfile):
    mod = asdl.parse(infile)
    
    with open(outfile,'w') as output:
        print("""/* THIS FILE WAS GENERATED BY {} */

#include <Python.h>
#include <structmember.h>
#include <Python-ast.h>
#include <symtable.h>


/* the next three functions are copied from Parser/asdl_c.py of the Python
   source (which is included in the "cpython" directory) */

static PyObject* ast2obj_list(asdl_seq *seq, PyObject* (*func)(void*))
{{
    Py_ssize_t i, n = asdl_seq_LEN(seq);
    PyObject *result = PyList_New(n);
    PyObject *value;
    if (!result)
        return NULL;
    for (i = 0; i < n; i++) {{
        value = func(asdl_seq_GET(seq, i));
        if (!value) {{
            Py_DECREF(result);
            return NULL;
        }}
        PyList_SET_ITEM(result, i, value);
    }}
    return result;
}}

static PyObject* ast2obj_object(void *o)
{{
    if (!o)
        o = Py_None;
    Py_INCREF((PyObject*)o);
    return (PyObject*)o;
}}
#define ast2obj_singleton ast2obj_object
#define ast2obj_identifier ast2obj_object
#define ast2obj_string ast2obj_object
#define ast2obj_bytes ast2obj_object

static PyObject* ast2obj_int(long b)
{{
    return PyLong_FromLong(b);
}}


_Py_IDENTIFIER(_raw_id);""".format(os.path.basename(__file__)),file=output)
        
        asdl_c.ChainOfVisitors(
            PyTypesDeclareVisitor(output),
            ObjVisitor(output),
            CleanupVisitor(output),
            ModuleVisitor(output)).visit(mod)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        create(os.path.join(base_dir,'cpython','Python.asdl'),sys.argv[1])
    else:
        print('Usage: {} <output file>'.format(os.path.basename(__file__)))
