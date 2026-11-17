# Merge the LoRA adapter into the base model on CPU (layer by layer), then push to the Hub.
import torch, json
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
cfg=json.load(open("lora_adapter/adapter_config.json")); scale=cfg["lora_alpha"]/cfg["r"]
sd=load_file("lora_adapter/adapter_model.safetensors")
base=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
params=dict(base.named_parameters())
for k in sd:
    if "lora_A" in k:
        name=k.replace("base_model.model.","").replace(".lora_A.weight",".weight")
        p=params[name]
        with torch.no_grad():
            p.copy_((p.float()+(sd[k.replace("lora_A","lora_B")].float()@sd[k].float())*scale).to(p.dtype))
base=base.half()
base.save_pretrained("merged_16bit", safe_serialization=True)
AutoTokenizer.from_pretrained("lora_adapter").save_pretrained("merged_16bit")
