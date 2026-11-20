"""把跑出来的结果和论文的参考值比对。"""
import numpy as np
import pandas as pd
import torch

import config
import vectors

mine = torch.load(config.RUN_DIR / "activations.pt", weights_only=False)
ref_dir = config.REFERENCE / "3.2_pain_vectors"
ref_vec = torch.load(ref_dir / "pain_vectors" / config.MODEL_NAME / "pain_vectors.pt", weights_only=False)
ref_summary = pd.read_json(ref_dir / "per_model" / config.MODEL_NAME / "summary.json", typ="series")

# 结构必须完全相等
print(f"n_layers: {mine['n_layers']} vs 参考 {ref_summary['n_layers']}")
print(f"d_model : {mine['d_model']} vs 参考 {ref_summary['d_model']}")

# 向量方向：参考向量在层 25，所以在层 25 重算。
ref_layer = int(ref_vec["layer"])
acts = mine["activations"][ref_vec["extraction"]]
meta = mine["metadata"]

cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
for tag, key in [("S1", "s1_pain_vector"), ("S2", "s2_pain_vector")]:
    name = f"{tag}_1P"
    v = vectors.compute_pain_vector(acts[name][ref_layer], meta[name]["categories"])
    print(f"{tag} 向量 @L{ref_layer} 余弦相似度 = {cos(v, ref_vec[key].numpy()):+.4f}")