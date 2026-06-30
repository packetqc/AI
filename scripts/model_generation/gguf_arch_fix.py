#!/usr/bin/env python3
"""Patch a trained GGUF's ARCH metadata in place — no retraining.

Two fixes for the multi-arch nocode models on an AVX-only CPU (where qwen3 retraining SIGILLs):
  * --head-dim N   : inject `<arch>.attention.key_length` + `.value_length` (qwen3 decouples
                     head_dim from hidden//heads; llama.cpp needs these or it derives the wrong dim).
  * --dst-arch A   : rename the arch tag + every `<src>.*` KV prefix to `A.*` (mistral is
                     structurally llama; ships under the 'llama' loader the legacy engine has).

Copies all other KV + every tensor verbatim.
"""
import argparse
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType


def _arch_of(reader):
    f = reader.fields.get("general.architecture")
    return str(bytes(f.parts[f.data[0]]).decode()) if f else None


def patch(inp, out, dst_arch=None, head_dim=None):
    r = GGUFReader(inp)
    src = _arch_of(r)
    dst = dst_arch or src
    w = GGUFWriter(out, arch=dst)
    for field in r.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue                                   # writer re-emits arch from the `arch=` param
        name = field.name
        if src and name.startswith(src + "."):
            name = dst + "." + name[len(src) + 1:]      # rename arch-scoped keys (mistral.* -> llama.*)
        vt = field.types[0]
        sub = field.types[-1] if vt == GGUFValueType.ARRAY else None
        w.add_key_value(name, field.contents(), vt, sub_type=sub)
    if head_dim:
        w.add_key_value(dst + ".attention.key_length", int(head_dim), GGUFValueType.UINT32)
        w.add_key_value(dst + ".attention.value_length", int(head_dim), GGUFValueType.UINT32)
    for t in r.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print("patched %s -> %s (arch %s%s%s)" % (
        inp, out, src, "" if dst == src else " -> " + dst,
        "" if not head_dim else ", +key/value_length=%s" % head_dim))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--dst-arch", default=None)
    ap.add_argument("--head-dim", type=int, default=None)
    a = ap.parse_args()
    patch(a.input, a.output, a.dst_arch, a.head_dim)
