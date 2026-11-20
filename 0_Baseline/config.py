"""需要改参数的话，在这个文件里改!"""
import os

# TransformerLens 2.18 只要 device="mps" 就无条件警告"可能静默算错"。
# 在这里我把这个警告关掉了。
os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
from pathlib import Path
import torch

# 路径相关
REPO_ROOT = Path(__file__).resolve().parent.parent
PAIN_AXIS = REPO_ROOT / "Pain-axis"
DATASETS = PAIN_AXIS / "datasets"
REFERENCE = PAIN_AXIS / "results"      # 论文作者跑出来的参考结果。只读，不要往里写任何内容！

assert DATASETS.is_dir(), f"找不到论文代码库 {PAIN_AXIS}。"

RUN_NAME = "gemma2-2b_dryrun_20261119"
RUN_DIR = REPO_ROOT / "runs" / RUN_NAME

MODEL = "google/gemma-2-2b-it" # MODEL_NAME 和 Pain-axis/results/ 下的目录名要完全一致，比对参考值时拼路径有用。
MODEL_NAME = "Gemma_2_2B_instruct"
MODEL_REVISION = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8" # 本机下载到的那份权重的 commit hash。先记一下……

DEVICE = "mps"
DTYPE = torch.bfloat16


RANDOM_SEED = 42           # 固定随机种子42，便于复现。
DENOISE_VARIANCE = 0.5     # 去噪强度（这里之后再来解释……？？）

# Pain-axis/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:45-57
PAIN_CATEGORIES = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E"]
NEUTRAL_CATEGORY = "D"

CATEGORY_LABELS = {
    "A1": "Physical Pain", "A2": "Psychological", "A3": "Social Pain",
    "A4": "Moral Injury", "A5": "Cognitive Pain",
    "B": "Fear", "C1": "Neg Emotion", "C2": "Neg World",
    "D": "Neutral", "E": "Body Sensation",
}

# 验证代码
if __name__ == "__main__":
    print("REPO_ROOT:", REPO_ROOT)
    print("DATASETS :", DATASETS, DATASETS.is_dir())
    print("REFERENCE:", REFERENCE, REFERENCE.is_dir())
    print("RUN_DIR  :", RUN_DIR, "(存在)" if RUN_DIR.is_dir() else "(待创建)")
    print("MODEL    :", MODEL, DTYPE, DEVICE)