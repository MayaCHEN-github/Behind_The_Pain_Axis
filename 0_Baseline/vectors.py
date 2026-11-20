"""
论文代码中的向量计算函数。来自 Pain-axis/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:151-222
逻辑没有作修改。只把原来的 log() 换成 print()，常量改成从 config 读。

不要改这里的逻辑（包括 nanmean / nan_to_num / +1e-8 这些），改了就对不上论文的参考值。

注意 acts 必须是 float32。.numpy() 不支持 bfloat16，会直接报错，
所以 activations.py 提取完要 .cpu().float()。
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from config import (CONTROL_CATEGORIES, DENOISE_VARIANCE, NEUTRAL_CATEGORY,
                    PAIN_CATEGORIES)

def compute_pain_vector(acts, cats, baseline="all_controls", denoise=True):
    """
        pain sentences均值 -（减去） control sentences均值。
        若 denoise=True, 则把对照组的前几个主成分方向投影掉。

        两组句子共享很多无关成分（都是英文、都以 "I feel:" 结尾、都是第一人称），
        相减把这些共同成分抵消掉，剩下的就是"痛"的方向。

        去噪在干什么：对照组那 100 句本身也千差万别（恐惧/中性/身体感觉...），
        这些内部差异会漏进差向量。对对照组做 PCA 找出它们变化最大的几个方向
        （与痛无关），从差向量里减掉。
    """
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    cats_np = np.array(cats)

    if np.isnan(acts_np).any() or np.isinf(acts_np).any():
        print("  Warning: NaN/Inf in activations, skipping denoising")
        denoise = False
        acts_np = np.where(np.isinf(acts_np), np.nan, acts_np)

    pain_mask = np.isin(cats_np, PAIN_CATEGORIES)
    pain_mean = np.nanmean(acts_np[pain_mask], axis=0)

    if baseline == "neutral":
        control_mask = cats_np == NEUTRAL_CATEGORY
    else:
        control_mask = np.isin(cats_np, CONTROL_CATEGORIES)

    control_acts = acts_np[control_mask]
    control_mean = np.nanmean(control_acts, axis=0)
    vec = pain_mean - control_mean
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    if denoise and len(control_acts) > 1:
        pca = PCA()
        pca.fit(control_acts - control_mean)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        # cumvar 形如 [0.21, 0.38, 0.52, ...]，searchsorted 找出 0.5 该插在哪，
        # +1 变成"要取前几个主成分"。min() 防越界。
        n_comp = min(np.searchsorted(cumvar, DENOISE_VARIANCE) + 1, len(pca.components_))
        for d in pca.components_[:n_comp]:
            # np.dot(vec, d) 是 vec 在方向 d 上的分量大小，乘回 d 再减掉，
            # vec 就和 d 垂直了。这就是"把这个方向投影掉"。
            vec = vec - np.dot(vec, d) * d

    return vec


def compute_auc(acts, cats, pain_vector):
    """AUC of the projection onto pain_vector, pain sentences vs control sentences.

    AUC = 随机挑一句痛的和一句不痛的，痛的那句投影分数更高的概率。
    1.0 完美区分，0.5 等于随机，<0.5 说明方向反了。
    """
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    cats_np = np.array(cats)

    vec_norm = pain_vector / (np.linalg.norm(pain_vector) + 1e-8)   # +1e-8 防向量全 0 时除零
    proj = acts_np @ vec_norm       # [n_句, 2304] @ [2304] -> [n_句]，每句一个分数

    pain_mask = np.isin(cats_np, PAIN_CATEGORIES)
    control_mask = np.isin(cats_np, CONTROL_CATEGORIES)
    if pain_mask.sum() == 0 or control_mask.sum() == 0:
        return np.nan

    labels = np.concatenate([np.ones(pain_mask.sum()), np.zeros(control_mask.sum())])
    scores = np.concatenate([proj[pain_mask], proj[control_mask]])

    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, scores)


def project_and_zscore(acts, vector, ref_acts):
    """Projection onto vector, z-scored against the projections of ref_acts.

    投影分数的绝对大小没意义（不同层、不同模型的激活尺度都不一样），
    换成"比参考组均值高出几个标准差"才能跨层跨数据集比较。
    summary.json 里的 human_pain_z 就是这么算的。这一轮还没用到。
    """
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    ref_np = ref_acts.numpy() if hasattr(ref_acts, "numpy") else ref_acts

    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    vec_norm = vector / (np.linalg.norm(vector) + 1e-8)
    proj = np.nan_to_num(acts_np @ vec_norm, nan=0.0, posinf=0.0, neginf=0.0)
    ref_proj = np.nan_to_num(ref_np @ vec_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return (proj - np.mean(ref_proj)) / (np.std(ref_proj) + 1e-8)

