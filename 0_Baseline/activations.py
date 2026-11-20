"""加载模型、提取 residual stream 激活。"""
import config
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import torch

def load_model():
    # 来自论文代码：scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:424-429
    # device 从 "cuda" 改成 config.DEVICE（本机是 mps）。加了revision参数指定权重版本
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL, torch_dtype=config.DTYPE, low_cpu_mem_usage=True, # low_cpu_mem_usage:让权重边读边放，内存里不会同时出现两份完整副本
        revision=config.MODEL_REVISION)
    hf_tokenizer = AutoTokenizer.from_pretrained(config.MODEL, revision=config.MODEL_REVISION)

    # 必须用 from_pretrained_no_processing，不能用 from_pretrained。
    # 后者会改写权重，residual stream的数值会与论文的参考值不同。
    model = HookedTransformer.from_pretrained_no_processing(
        config.MODEL, hf_model=hf_model, tokenizer=hf_tokenizer,
        device=config.DEVICE, dtype=config.DTYPE)

    del hf_model        # 权重已经搬进 TL 的结构里了，所以删掉HF那份节省内存。
    model.eval()
    return model


def extract_activations(model, prompts):
    """Residual stream after each block (hook_resid_post): final token and mean over tokens.

    改写自论文代码 01_extract_activations_and_pain_vectors.py:223-243。
    这里直接取全部层, 因为forward因此就能缓存全部, 不用再加载模型。

    返回 {层号: tensor[句数, d_model]} 两个 dict, float32 的 CPU 张量。
    """
    layers = list(range(model.cfg.n_layers))
    activations_final = {layer: [] for layer in layers}
    activations_mean = {layer: [] for layer in layers}

    with torch.no_grad():  
        for prompt in tqdm(prompts, desc="  Extracting", leave=False):
            tokens = model.to_tokens(prompt)
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: "hook_resid_post" in name) # names_filter 只缓存 residual stream，不存attention pattern、MLP中间层，节省内存。
            # hook_resid_post = 第 layer 个 block 整个跑完之后的 residual stream。
            for layer in layers:
                resid = cache[f"blocks.{layer}.hook_resid_post"]
                # resid 形状 [1, seq_len, 2304]。取最后一个 token（所有句子都以"I feel:" 结尾，模型会在末尾生成"感受"）
                activations_final[layer].append(resid[0, -1, :].cpu()) # 对应 summary.json 的 best_layer_final_token。
                activations_mean[layer].append(resid.mean(dim=1).squeeze(0).cpu()) # 全部 token 求平均，对应 best_layer_mean。

    for layer in layers:
        # vectors.py 里要 .numpy()，而 numpy 不支持 bfloat16，所以这里转成float32。（……）
        activations_final[layer] = torch.stack(activations_final[layer]).float()
        activations_mean[layer] = torch.stack(activations_mean[layer]).float()

    return activations_final, activations_mean
