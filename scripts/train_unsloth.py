# Run on a GPU (Colab/Kaggle T4). pip install unsloth
from unsloth import FastLanguageModel
import torch, json
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
model, tok = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-3B-Instruct", max_seq_length=4096, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)
def P(t):
    return ("Summarize the following US congressional bill in a clear, factual legislative summary "
            "of at most 200 words. Output only the summary.\n\nBILL:\n"+t)
data=[]
for r in (json.loads(l) for l in open("train.jsonl")):
    text=tok.apply_chat_template([{"role":"user","content":P(r["text"])},{"role":"assistant","content":r["summary"]}],tokenize=False)
    if len(tok(text).input_ids) <= 3072: data.append({"text":text})
ds=Dataset.from_list(data[:1500])
trainer=SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=SFTConfig(
    dataset_text_field="text", max_seq_length=3072, per_device_train_batch_size=2, gradient_accumulation_steps=8,
    num_train_epochs=1, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=5, logging_steps=5,
    optim="adamw_8bit", fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
    output_dir="out", report_to="none", seed=42))
trainer.train()
model.save_pretrained("lora_adapter"); tok.save_pretrained("lora_adapter")
