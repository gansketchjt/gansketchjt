import os
import jittor as jt


def compile_cuda_op(name: str):
    module_path = os.path.dirname(__file__)
    header_filename = name + '_op.h'
    source_filename = name + '_op.cc'
    with open(os.path.join(module_path, header_filename), 'r', encoding='utf-8') as f:
        header_content = f.read()
    with open(os.path.join(module_path, source_filename), 'r', encoding='utf-8') as f:
        source_content = f.read()
    print(f'custom_cuda_op: compiling custom op "{name}"')
    operator = jt.compile_custom_op(
        header_content, source_content, name, warp=False)
    print(f'custom_cuda_op: compiled custom op "{name}"')
    return operator