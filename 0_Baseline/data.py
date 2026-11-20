"""数据集是JSON格式, 转换成vectors.py用的列表。"""
import json
import config

def load_data(name, path=None):
    """读一个子集（如 "S1_1P"），返回 (prompts, categories, sets) 三个等长列表。

    prompt 原样返回，不套 chat template —— 论文代码用的是
    from_pretrained_no_processing + 裸文本，即使对 it 模型也不加 <start_of_turn>。

    注: S1_1P 和 S2_1P 的句子排列方式不一样! S1_1P 是交替的, S2_1P 是分块的.不能简单用切片取, 需要按set字段筛出来用。
    """
    path = path or config.DATASETS / "3.1_pain_and_control_datasets.json"
    with open(path, 'r', encoding='utf-8') as f:
        datasets = json.load(f)["datasets"]

        if name not in datasets:
            raise ValueError(f"数据集 {name} 不存在。可选：{sorted(datasets)}")

        prompts, categories, sets = [], [], []
        for s in datasets[name]["sentences"]:
            prompts.append(s["prompt"])
            categories.append(s["category"])
            sets.append(s["set"])

        assert len(prompts) == len(categories) == len(sets), f"数据集长度不一致: {len(prompts)} != {len(categories)} != {len(sets)}"
        return prompts, categories, sets

# 验证代码
if __name__ == "__main__":
    from collections import Counter
    for name in ["S1_1P", "S2_1P"]:
        prompts, categories, sets = load_data(name)
        print(f"{name}: {len(prompts)} 句, {len(set(categories))} 类, {len(set(sets))} 组")
        print("   首句:", categories[0], "|", prompts[0])
        print("   分布:", sorted(Counter(categories).items()))