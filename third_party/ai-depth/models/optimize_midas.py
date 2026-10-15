# optimize_midas.py <fp32 in> <out prefix>
# 1. drops the in-graph ImageNet normalisation (the app's shader does it),
# 2. folds the zero Pads into the following Conv's pads,
# 3. folds BatchNormalization into the preceding Conv,
# 4. converts to FP16 including Resize (fp32 io and fp16 io variants).
import sys, onnx, numpy as np
from onnx import numpy_helper, helper
from onnxconverter_common import float16
src, prefix = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
nodes = list(g.node)
inits = {i.name: i for i in g.initializer}
consumers = lambda name: [n for n in g.node if name in n.input]
def attr(n, name, default=None):
    for a in n.attribute:
        if a.name == name: return helper.get_attribute_value(a)
    return default
def set_attr(n, name, value):
    for a in list(n.attribute):
        if a.name == name: n.attribute.remove(a)
    n.attribute.append(helper.make_attribute(name, value))
def rename_input(old, new):
    for n in g.node:
        for i, x in enumerate(n.input):
            if x == old: n.input[i] = new
    for o in g.output:
        if o.name == old: o.name = new

# 1. normalisation
sub = next(n for n in g.node if n.op_type == "Sub" and n.input[0] == g.input[0].name)
div = next(n for n in g.node if n.op_type == "Div" and n.input[0] == sub.output[0])
consts = [n for n in g.node if n.op_type == "Constant" and n.output[0] in (sub.input[1], div.input[1])]
rename_input(div.output[0], g.input[0].name)
for n in [sub, div] + consts: g.node.remove(n)

# 2. Pad -> Conv
pads_removed = 0
for p in [n for n in g.node if n.op_type == "Pad"]:
    if attr(p, "mode", b"constant") != b"constant" or attr(p, "value", 0.0) != 0.0 or len(p.input) > 1: continue
    cs = consumers(p.output[0])
    if len(cs) != 1 or cs[0].op_type != "Conv": continue
    c = cs[0]
    pp = attr(p, "pads")   # [n0,c0,h0,w0,n1,c1,h1,w1]
    if any(pp[i] for i in (0, 1, 4, 5)): continue
    if attr(c, "auto_pad", b"NOTSET") not in (b"NOTSET",): continue
    cp = attr(c, "pads", [0, 0, 0, 0])
    set_attr(c, "pads", [cp[0] + pp[2], cp[1] + pp[3], cp[2] + pp[6], cp[3] + pp[7]])
    c.input[0] = p.input[0]
    g.node.remove(p); pads_removed += 1

# 3. BN -> Conv
bn_folded = 0
for b in [n for n in g.node if n.op_type == "BatchNormalization"]:
    prod = [n for n in g.node if b.input[0] in n.output]
    if len(prod) != 1 or prod[0].op_type != "Conv" or len(consumers(b.input[0])) != 1: continue
    c = prod[0]
    W = numpy_helper.to_array(inits[c.input[1]]).astype(np.float64)
    B = numpy_helper.to_array(inits[c.input[2]]).astype(np.float64) if len(c.input) > 2 else np.zeros(W.shape[0])
    s, bb, mean, var = [numpy_helper.to_array(inits[x]).astype(np.float64) for x in b.input[1:5]]
    k = s / np.sqrt(var + attr(b, "epsilon", 1e-5))
    W2 = (W * k[:, None, None, None]).astype(np.float32)
    B2 = ((B - mean) * k + bb).astype(np.float32)
    wn, bn_ = c.output[0] + "_W", c.output[0] + "_B"
    g.initializer.extend([numpy_helper.from_array(W2, wn), numpy_helper.from_array(B2, bn_)])
    c.input[1] = wn
    if len(c.input) > 2: c.input[2] = bn_
    else: c.input.append(bn_)
    out = b.output[0]
    g.node.remove(b)
    c.output[0] = out
    bn_folded += 1

# unused initializers
used = {x for n in g.node for x in n.input}
for i in list(g.initializer):
    if i.name not in used: g.initializer.remove(i)
del g.value_info[:]
m = onnx.shape_inference.infer_shapes(m); onnx.checker.check_model(m)
onnx.save(m, prefix + "_fp32.onnx")
import collections
print("pads folded", pads_removed, "bn folded", bn_folded, collections.Counter(n.op_type for n in m.graph.node))
for keep, tag in ((True, "fp16"), (False, "fp16io")):
    h = float16.convert_float_to_float16(m, keep_io_types=keep, op_block_list=[])
    # Resize's scales stay float32 (the spec), as an initializer of their own.
    hi = {i.name: i for i in h.graph.initializer}
    for n in h.graph.node:
        if n.op_type == "Resize":
            for k in range(1, len(n.input)):
                x = n.input[k]
                if x in hi and hi[x].data_type == onnx.TensorProto.FLOAT16:
                    a = numpy_helper.to_array(hi[x]).astype(np.float32)
                    h.graph.initializer.remove(hi[x]); hi[x] = numpy_helper.from_array(a, x); h.graph.initializer.append(hi[x])
    onnx.checker.check_model(h, full_check=True)
    onnx.save(h, prefix + "_" + tag + ".onnx")
    print(tag, collections.Counter(n.op_type for n in h.graph.node))
