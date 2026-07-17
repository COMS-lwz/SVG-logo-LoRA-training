"""
评估脚本：对比基座模型和微调模型在验证集上的表现
"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from reward import compute_reward

def load_model(model_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer

def generate_svg(model, tokenizer, prompt, max_tokens=768, temp=0.7, top_p=0.8):
    messages = [
        {"role": "system", "content": "You are an expert logo designer. Generate clean SVG logos."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # 提取第一个 <svg>...</svg>（如果存在）
    match = re.search(r'<svg.*?</svg>', generated, re.DOTALL)
    return match.group(0) if match else generated

def main():
    # 配置
    model_path = "./gemma-3-270m-it"
    adapter_path = "./adapter"
    data_path = "./logo-detailed-prompt/valid.jsonl"
    results_file = "results.json"
    
    # 加载数据
    with open(data_path, 'r') as f:
        samples = [json.loads(line) for line in f if line.strip()][:20]
    
    # 基座模型
    print("Loading base model...")
    base_model, tokenizer = load_model(model_path)
    base_scores = []
    for i, sample in enumerate(samples):
        prompt = sample["messages"][0]["content"]
        svg = generate_svg(base_model, tokenizer, prompt)
        print(f"Base [{i+1}] 输出前200字符: {svg[:200]}")
        score = compute_reward(svg, prompt)["total"]
        base_scores.append(score)
        print(f"Base [{i+1}/{len(samples)}] score: {score:.3f}")
    del base_model
    torch.cuda.empty_cache()
    
    # 微调模型
    print("\nLoading fine-tuned model...")
    ft_model, tokenizer = load_model(model_path, adapter_path)
    ft_scores = []
    for i, sample in enumerate(samples):
        prompt = sample["messages"][0]["content"]
        svg = generate_svg(ft_model, tokenizer, prompt)
        print(f"FT [{i+1}] 输出前200字符: {svg[:200]}")
        score = compute_reward(svg, prompt)["total"]
        ft_scores.append(score)
        print(f"FT   [{i+1}/{len(samples)}] score: {score:.3f}")
    
    # 汇总
    base_avg = sum(base_scores) / len(base_scores)
    ft_avg = sum(ft_scores) / len(ft_scores)
    
    results = {
        "num_samples": len(samples),
        "base_model_avg": base_avg,
        "fine_tuned_avg": ft_avg,
        "improvement": ft_avg - base_avg,
        "base_scores": base_scores,
        "fine_tuned_scores": ft_scores,
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*40)
    print(f"Base avg      : {base_avg:.4f}")
    print(f"Fine-tuned avg: {ft_avg:.4f}")
    print(f"Improvement   : {ft_avg - base_avg:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()