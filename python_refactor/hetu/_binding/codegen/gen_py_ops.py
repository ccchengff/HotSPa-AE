import os, sys, traceback
import yaml
from collections import OrderedDict, defaultdict
import argparse

from args_bridge import ArgType, parse_args

cpp_indent = " " * 2

op_file_fmt = """\
// Auto-generated. Do NOT edit!

#include "hetu/_binding/graph/ops/python_headers.h"

namespace hetu {{
namespace graph {{

{file_body}

}} // namespace graph
}} // namespace hetu
"""

op_fmts = {}

op_fmts['class'] = """\
PyObject* Tensor_class_{py_fn_name}(PyObject*, PyObject* args, PyObject* kwargs) {{
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({parser_fmts});
  auto parsed_args = parser.parse(args, kwargs);
  PyObject* ret = nullptr;
  {pre_processing}
  switch (parsed_args.signature_index()) {{
    {signature_handlers}
    default: {{
      HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
      __builtin_unreachable();
    }}
  }}
  {post_processing}
  return ret;
  HT_PY_FUNC_END
}}

REGISTER_TENSOR_CLASS_METHOD(
  {py_fn_name}, 
  (PyCFunction) Tensor_class_{py_fn_name}, 
  {py_method_flags}, 
  {py_method_doc});
"""

op_fmts['member'] = """\
PyObject* Tensor_member_{py_fn_name}(PyTensor* self, PyObject* args, PyObject* kwargs) {{
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({parser_fmts});
  auto parsed_args = parser.parse(args, kwargs);
  auto& self_arg = self->tensor;
  Operator op;
  PyObject* ret = nullptr;
  {pre_processing}
  switch (parsed_args.signature_index()) {{
    {signature_handlers}
    default: {{
      HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
      __builtin_unreachable();
    }}
  }}
  {post_processing}
  return ret;
  HT_PY_FUNC_END
}}

REGISTER_TENSOR_METHOD(
  {py_fn_name}, 
  (PyCFunction) Tensor_member_{py_fn_name}, 
  {py_method_flags}, 
  {py_method_doc});
"""

op_signature_handler_case_fmt = """
    case {signature_index}: {{
      {cpp_function}
      break;
    }}
"""

def gen_ops(input_file, output_dir):
    with open(input_file) as stream:
        loaded_functions = yaml.safe_load(stream)
        functions = OrderedDict()
        unique_ops = set()
        for loaded_func in loaded_functions:
            assert (('name' in loaded_func) and 
                    ('op' in loaded_func) and 
                    ('args' in loaded_func)), \
                f"Invalid operator: {loaded_func}"
            py_fn_name = loaded_func['name']
            cpp_op_name = loaded_func['op']
            args = loaded_func['args']
            if py_fn_name not in functions:
                functions[py_fn_name] = []
            functions[py_fn_name].append((
                cpp_op_name, 
                args, 
                loaded_func.get('self')
            ))
            unique_ops.add(cpp_op_name)
    print(
        f"Generating {len(loaded_functions)} variants of {len(functions)} functions "
        f"(for {len(unique_ops)} operators defined in {input_file})...")
    
    num_generated = 0
    for py_fn_name, py_fn_descriptions in functions.items():
        parser_fmts = defaultdict(list)
        signature_handlers = defaultdict(list)
        signature_indexes = defaultdict(int)
        for (cpp_op_name, args, self_arg_name) in py_fn_descriptions:
            method_types = ['class']
            if self_arg_name is not None:
                method_types.append('member')
            for method_type in method_types:
                parsed_results = parse_args(
                    args, 'operator', 
                    self_arg_name=(self_arg_name if method_type == 'member' else None))
                arg_getters = parsed_results['arg_getters']
                
                # parse op arguments
                cpp_function = f"ret = PyObject_FromOperatorOutputs(Make{cpp_op_name}("
                # 1. operator-specific arguments
                if len(arg_getters) == 0:
                    if method_type == 'member':
                        assert parsed_results['self_pos'] == 0
                        cpp_function += "\n" + (cpp_indent * 4) + "self_arg, "
                else:
                    for arg_pos, arg_getter in enumerate(arg_getters):
                        if method_type == 'member' and arg_pos == parsed_results['self_pos']:
                            cpp_function += "\n" + (cpp_indent * 4) + "self_arg, "
                        cpp_function += "\n" + (cpp_indent * 4) + arg_getter + ", "
                    if method_type == 'member' and parsed_results['self_pos'] == len(arg_getters):
                        cpp_function += "\n" + (cpp_indent * 4) + "self_arg, "
                    
                # 2. common arguments for op_meta
                cpp_function += "\n" + (cpp_indent * 4) + \
                    f"parse_op_meta(parsed_args, {len(arg_getters)})"
                cpp_function += "\n" + (cpp_indent * 3) + "));"
                
                op_specific_args = args if method_type == 'class' \
                                        else parsed_results['args_excluding_self']
                if op_specific_args == "":
                    signature = f"\"{py_fn_name}(\" OP_META_ARGS \")\""
                else:
                    signature = f"\"{py_fn_name}({op_specific_args}, \" OP_META_ARGS \")\""
                signature_handler = op_signature_handler_case_fmt.format(
                    signature_index=signature_indexes[method_type], 
                    cpp_function=cpp_function
                )

                parser_fmts[method_type].append(signature)
                signature_handlers[method_type].append(signature_handler)
                signature_indexes[method_type] += 1

                if method_type == 'class':
                    print(f"Processed {py_fn_name}({op_specific_args}) ==> {cpp_op_name}...")
                else:
                    print(f"Processed {self_arg_name}.{py_fn_name}({op_specific_args}) ==> {cpp_op_name}...")
        
        gen_codes = []
        for method_type in parser_fmts.keys():
            parser_fmts_of_type = parser_fmts[method_type]
            signature_handlers_of_type = signature_handlers[method_type]
            assert len(parser_fmts_of_type) == len(signature_handlers_of_type)
            
            parser_fmts_indent = cpp_indent * 2
            packed_parser_fmts = "{"
            for signature_index, parser_fmt in enumerate(parser_fmts_of_type):
                packed_parser_fmts += "\n" + (cpp_indent * 2) + parser_fmt
                if signature_index + 1 != len(parser_fmts_of_type):
                    packed_parser_fmts += ", "
            packed_parser_fmts += "\n" + cpp_indent + "}"
            packed_signature_handlers = "".join(signature_handlers_of_type)
            
            one_gen = op_fmts[method_type].format(
                py_fn_name=py_fn_name, 
                parser_fmts=packed_parser_fmts, 
                signature_handlers=packed_signature_handlers, 
                pre_processing="", 
                post_processing="",
                py_method_flags="METH_VARARGS | METH_KEYWORDS", 
                py_method_doc="nullptr", 
            )
            gen_codes.append(one_gen)
        
        op_file_content = op_file_fmt.format(
            file_body="\n".join(gen_codes), 
        )

        filename = f"generated_{py_fn_name}.cc"
        with open(os.path.join(output_dir, filename), "w") as writer:
            writer.write(op_file_content)
        num_generated += 1
        print(f"Generated {py_fn_name} ({num_generated}/{len(functions)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input yaml")
    parser.add_argument("--output-dir", type=str, help="Path to output directory")
    args = parser.parse_args()
    gen_ops(args.input, args.output_dir)
