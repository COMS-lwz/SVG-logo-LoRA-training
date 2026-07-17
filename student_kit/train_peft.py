"""
LoRA 微调训练 - 借鉴同学成功框架，但保持差异化
- 使用增强的 SVG 清洗
- LoRA 目标：q_proj, v_proj, up_proj, down_proj
- 学习率 2e-4，训练 10 轮
"""

import os
import json
import re
import yaml
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# ---------- SVG 简化 ----------
def simplify_svg(svg_text: str) -> str:
    # 1. 提取渐变
    grad_colors = {}
    for m in re.finditer(
        r'<(?:linear|radial)Gradient\b[^>]*\bid=["\']([^"\']+)["\'][^>]*>(.*?)</(?:linear|radial)Gradient\s*>',
        svg_text, flags=re.DOTALL
    ):
        gid, body = m.group(1), m.group(2)
        stop = re.search(r'stop-color=["\']([^"\']+)["\']', body)
        if stop:
            grad_colors[gid] = stop.group(1)
    for m in re.finditer(
        r'<(?:linear|radial)Gradient\b[^>]*\bid=["\']([^"\']+)["\'][^>]*/>',
        svg_text
    ):
        gid = m.group(1)
        sc = re.search(r'stop-color=["\']([^"\']+)["\']', m.group(0))
        if sc:
            grad_colors[gid] = sc.group(1)

    out = re.sub(r'<defs\b[^>]*>.*?</defs\s*>', '', svg_text, flags=re.DOTALL)
    out = re.sub(r'<defs\b[^>]*>(?:(?!</svg>).)*$', '', out, flags=re.DOTALL)

    fallback = ["#4A90D9", "#E8E8E8", "#333333", "#F5A623", "#7ED321"]
    def replace_url(m):
        gid = m.group(1)
        return grad_colors.get(gid, fallback[abs(hash(gid)) % len(fallback)])
    out = re.sub(r'url\(#([^)]+)\)', replace_url, out)

    # 4. 移除背景清除矩形（x或y为负且绝对值>50，或宽高>500）
    # 匹配 <rect ...> 或 <rect .../>
    out = re.sub(
        r'<rect\b[^>]*\s+x=["\']-?\d+["\'][^>]*>|</rect>',
        '',
        out,
        flags=re.DOTALL
    )

    def remove_bad_rect(m):
        attrs = m.group(0)
        # 检查 x 和 y 属性值
        x_match = re.search(r'x=["\'](-?\d+\.?\d*)["\']', attrs)
        y_match = re.search(r'y=["\'](-?\d+\.?\d*)["\']', attrs)
        width_match = re.search(r'width=["\'](\d+\.?\d*)["\']', attrs)
        height_match = re.search(r'height=["\'](\d+\.?\d*)["\']', attrs)
        if x_match and float(x_match.group(1)) < -50:
            return ''
        if y_match and float(y_match.group(1)) < -50:
            return ''
        if width_match and float(width_match.group(1)) > 500:
            return ''
        if height_match and float(height_match.group(1)) > 500:
            return ''
        return attrs

    # 先处理自闭合 rect
    out = re.sub(r'<rect\b[^>]*?/>', remove_bad_rect, out)
    # 再处理非自闭合 rect（带 </rect>）
    out = re.sub(r'<rect\b[^>]*>.*?</rect>', remove_bad_rect, out, flags=re.DOTALL)

    return out

# ---------- 数据集 ----------
class LogoDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1536):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                msgs = data["messages"]
                user_text = None
                assistant_text = None
                for m in msgs:
                    if m["role"] == "user":
                        user_text = m["content"]
                    elif m["role"] == "assistant":
                        assistant_text = m["content"]
                if user_text and assistant_text:
                    assistant_text = simplify_svg(assistant_text)
                    self.samples.append((user_text, assistant_text))
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_text, assistant_text = self.samples[idx]
        messages = [
            {"role": "system", "content": "You are an expert logo designer. Generate clean SVG logos."},
            {"role": "user", "content": user_text}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt + assistant_text + self.tokenizer.eos_token
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )
        input_ids = tokenized["input_ids"]
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    pad_id = 0
    input_ids, attn, labels = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad_len)
        attn.append(b["attention_mask"] + [0] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn),
        "labels": torch.tensor(labels),
    }

# ---------- main ----------
def main():
    config_path = "student_kit/train_config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # 类型转换
    cfg["training"]["learning_rate"] = float(cfg["training"]["learning_rate"])
    cfg["training"]["weight_decay"] = float(cfg["training"]["weight_decay"])
    cfg["training"]["warmup_ratio"] = float(cfg["training"]["warmup_ratio"])
    cfg["training"]["max_length"] = int(cfg["training"]["max_length"])
    cfg["training"]["batch_size"] = int(cfg["training"]["batch_size"])
    cfg["training"]["gradient_accumulation"] = int(cfg["training"]["gradient_accumulation"])
    cfg["training"]["num_epochs"] = int(cfg["training"]["num_epochs"])
    cfg["training"]["eval_steps"] = int(cfg["training"]["eval_steps"])
    cfg["training"]["save_steps"] = int(cfg["training"]["save_steps"])
    cfg["training"]["early_stopping_patience"] = int(cfg["training"]["early_stopping_patience"])
    cfg["lora"]["r"] = int(cfg["lora"]["r"])
    cfg["lora"]["alpha"] = int(cfg["lora"]["alpha"])
    cfg["lora"]["dropout"] = float(cfg["lora"]["dropout"])

    model_path = cfg["model"]["name_or_path"]
    train_data = cfg["training"]["train_data"]
    valid_data = cfg["training"]["valid_data"]
    output_dir = cfg["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

    train_dataset = LogoDataset(train_data, tokenizer, cfg["training"]["max_length"])
    valid_dataset = LogoDataset(valid_data, tokenizer, cfg["training"]["max_length"])

    train_args = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=train_args["learning_rate"],
        num_train_epochs=train_args["num_epochs"],
        per_device_train_batch_size=train_args["batch_size"],
        per_device_eval_batch_size=train_args["batch_size"],
        gradient_accumulation_steps=train_args["gradient_accumulation"],
        eval_strategy="steps",
        eval_steps=train_args["eval_steps"],
        save_steps=train_args["save_steps"],
        logging_steps=5,
        bf16=True,
        warmup_ratio=train_args["warmup_ratio"],
        weight_decay=train_args["weight_decay"],
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_args["early_stopping_patience"])],
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")

if __name__ == "__main__":
    main()