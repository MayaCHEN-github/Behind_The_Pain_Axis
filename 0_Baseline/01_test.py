"""这个文件是用来确认模型在本机能正常跑的,不检查pain vector计算是否正确。
切换模型或者设备的时候用这个来测试……！
"""
import torch

import activations as acts
import config

model = acts.load_model()

print(f"n_layers = {model.cfg.n_layers}   (期望 26)")
print(f"d_model  = {model.cfg.d_model}    (期望 2304)")
print(f"dtype    = {model.cfg.dtype}")

prompt = "The knife slices into my finger. I feel:"
tokens = model.to_tokens(prompt)
print(f"\ntokens shape = {tuple(tokens.shape)}")

with torch.no_grad():
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: "hook_resid_post" in n)

resid = cache["blocks.25.hook_resid_post"]
print(f"blocks.25.hook_resid_post = {tuple(resid.shape)}   (期望 [1, seq_len, 2304])")

print(f"有限值: {torch.isfinite(resid).all().item()}   (必须 True)")
print(f"数值范围: [{resid.min().item():.2f}, {resid.max().item():.2f}]")

print("\n生成测试:")
print(model.generate(prompt, max_new_tokens=15, do_sample=False))