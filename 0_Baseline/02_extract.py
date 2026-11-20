"""提取激活、算 S1/S2 pain vector、算每层 AUC，输出到 runs/<RUN_NAME>/。

Tips: 先 SMOKE = True 跑一遍（20 句，一分钟）确认文件都生成了，再改 False 跑全量。
输出会直接覆盖上一次，想并存多个run需要在config.RUN_NAME修改目录。
"""
import config
import data
import activations as acts
import vectors

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import KFold

SMOKE = False       # True时只取set<=2的20句，用来验证代码能跑通

SETS = ["S1_1P", "S2_1P", "S2_3P"]
config.RUN_DIR.mkdir(parents=True, exist_ok=True)

model = acts.load_model()
n_layers, d_model = model.cfg.n_layers, model.cfg.d_model
print(f"n_layers={n_layers}  d_model={d_model}")

all_acts = {"final_token": {}, "mean": {}}
all_meta = {}

for name in SETS:
    prompts, categories, sets = data.load_data(name)
    if SMOKE:
        keep = [i for i, s in enumerate(sets) if s <= 5] # set字段每组含10类别各1，set<=5就是10类各5句。
        prompts = [prompts[i] for i in keep]
        categories = [categories[i] for i in keep]
        sets = [sets[i] for i in keep]
    print(f"{name}: {len(prompts)} 句")
    assert set(categories) >= set(config.PAIN_CATEGORIES + config.CONTROL_CATEGORIES), \
    f"切片后只剩 {sorted(set(categories))}，缺对照类，算不出 pain vector"

    a_final, a_mean = acts.extract_activations(model, prompts)
    all_acts["final_token"][name] = a_final
    all_acts["mean"][name] = a_mean
    all_meta[name] = {"categories": categories, "sets": sets}

del model

# 存全部 26 层的激活，方便后续调选层、画图、算 z-score 时不用重新加载模型。
# 结构照 01_extract_activations_and_pain_vectors.py:450-457，多存了 revision。
torch.save({
    "activations": all_acts,
    "metadata": all_meta,
    "n_layers": n_layers,
    "d_model": d_model,
    "model": config.MODEL,
    "revision": config.MODEL_REVISION,
}, config.RUN_DIR / "activations.pt")

EXT = "final_token"          # 对应 summary.json 的 best_layer_final_token

# 论文用 5-fold 交叉验证选层（训练折算向量、测试折算 AUC，见论文代码01_extract_activations_and_pain_vectors.py:245-291）
N_FOLDS = 5
rows = []
for L in range(n_layers):
    row = {"layer": L}
    for name in SETS:
        acts_np = all_acts[EXT][name][L].numpy()
        cats = np.array(all_meta[name]["categories"])
        sets = np.array(all_meta[name]["sets"])
        unique_sets = sorted(set(sets))
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

        fold_aucs = []
        for train_idx, test_idx in kf.split(unique_sets):
            train_mask = np.isin(sets, [unique_sets[i] for i in train_idx])
            test_mask = np.isin(sets, [unique_sets[i] for i in test_idx])
            v = vectors.compute_pain_vector(acts_np[train_mask], cats[train_mask])
            auc = vectors.compute_auc(acts_np[test_mask], cats[test_mask], v)
            if not np.isnan(auc):
                fold_aucs.append(auc)

        row[f"auc_{name}"] = np.mean(fold_aucs) if fold_aucs else np.nan
        row[f"std_{name}"] = np.std(fold_aucs) if fold_aucs else np.nan
    rows.append(row)

curves = pd.DataFrame(rows)
curves.to_csv(config.RUN_DIR / "layer_curves.csv", index=False)

best_layer = int(curves.loc[curves[["auc_S2_1P","auc_S2_3P"]].mean(axis=1).idxmax(), "layer"]) # idxmax 返回的是行索引，现在正好等于层号。这里对应一下到layer列。
print(f"\n最佳层 = {best_layer}  (参考值 25)")
print(f"该层 S2 AUC = {curves['auc_S2_1P'].max():.4f}")

acts_best = {name: all_acts[EXT][name][best_layer] for name in SETS}

# 参考论文代码01_extract_activations_and_pain_vectors.py:500-501。
s1_vec = vectors.compute_pain_vector(acts_best["S1_1P"], all_meta["S1_1P"]["categories"])
s2_vec = vectors.compute_pain_vector(acts_best["S2_1P"], all_meta["S2_1P"]["categories"])

torch.save({
    "s1_pain_vector": torch.tensor(s1_vec),
    "s2_pain_vector": torch.tensor(s2_vec),
    "layer": best_layer,
    "extraction": EXT,
}, config.RUN_DIR / "pain_vectors.pt")