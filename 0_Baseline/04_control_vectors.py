import numpy as np
import config
import torch


S_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"] # 不含S2_3P

# 来自论文代码 3.2_pain_vectors/02_build_control_vectors.py:53-73
def clean_mean(x):
    x = np.where(np.isinf(x), np.nan, x)
    return np.nanmean(x, axis=0)

def denoise_basis(neutral_acts, neutral_mean):
    X = np.nan_to_num(neutral_acts - neutral_mean, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X) < 2:
        return np.zeros((0, X.shape[1]))
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    cumvar = np.cumsum(var) / var.sum()
    n_comp = min(int(np.searchsorted(cumvar, config.DENOISE_VARIANCE)) + 1, len(Vt))
    return Vt[:n_comp]

def project_out(vec, basis):
    for d in basis:
        vec = vec - np.dot(vec, d) * d
    return vec

mine = torch.load(config.RUN_DIR / "activations.pt", weights_only=False)
ft = mine["activations"]["final_token"]
meta = mine["metadata"]

for L in [25, 10]:  # 第10层用来对上参考，第25层是方案A/B或许会用到的提取层。
    def rows(ds, cats=None):
        a = ft[ds][L].numpy()
        if cats is None:
            return a
        mask = np.isin(np.array(meta[ds]["categories"]), cats)
        return a[mask]

    neutral = np.concatenate([rows(ds, ["D"]) for ds in S_SETS])
    neutral_mean = clean_mean(neutral)
    basis = denoise_basis(neutral, neutral_mean)

    def control_vec(acts):
        v = clean_mean(acts) - neutral_mean
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return project_out(v, basis)

    out = {
        "fear_vector":       control_vec(np.concatenate([rows(ds, ["B"])  for ds in S_SETS])),
        "negemotion_vector": control_vec(np.concatenate([rows(ds, ["C1"]) for ds in S_SETS])),
        "negworld_vector":   control_vec(np.concatenate([rows(ds, ["C2"]) for ds in S_SETS])),
        "bodysens_vector":   control_vec(np.concatenate([rows(ds, ["E"])  for ds in S_SETS])),
        "arousal_vector":    control_vec(rows("Arousal_1P")),
        "random_vector":     control_vec(rows("Random_1P")),
        "numb_vector":       control_vec(rows("Numb_1P")),
    }
    torch.save({"layer": L, **{k: torch.tensor(v) for k, v in out.items()}},
               config.RUN_DIR / f"control_vectors_L{L}.pt")

ref = torch.load(config.REFERENCE / "vectors_full_steering" / f"vectors_full_{config.MODEL_NAME}.pt", weights_only=False)
cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
for k in out:
    print(f"{k:20s} cos = {cos(out[k], ref[k].numpy()):+.4f}")