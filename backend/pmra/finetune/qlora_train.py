"""QLoRA fine-tune of a small open model for cited fund-metric extraction.

Run on a CUDA GPU (a free Colab T4 is enough for 0.5B-1.5B models):
    pip install -r requirements-finetune.txt
    python -m pmra.finetune.build_dataset
    python -m pmra.finetune.qlora_train --base Qwen/Qwen2.5-0.5B-Instruct --epochs 2
Then evaluate with:  python -m pmra.finetune.eval_ft --base ... --adapter artifacts/adapters/<run>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import config
from ..registry import Registry, file_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=str(config.ARTIFACTS / "sft"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    run = f"qlora-{args.base.split('/')[-1]}-r{args.r}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = config.ARTIFACTS / "adapters" / run

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    def encode(ex):
        """Mask the prompt so loss is computed on the assistant JSON only."""
        prompt = tok.apply_chat_template(ex["messages"][:-1], add_generation_prompt=True, tokenize=False)
        full = prompt + ex["messages"][-1]["content"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = tok(full, add_special_tokens=False, truncation=True, max_length=args.max_len)["input_ids"]
        return {"input_ids": ids, "attention_mask": [1] * len(ids),
                "labels": [-100] * len(p_ids) + ids[len(p_ids):]}

    ds = load_dataset("json", data_files={"train": f"{args.data}/train.jsonl", "val": f"{args.data}/val.jsonl"})
    ds = ds.map(encode, remove_columns=ds["train"].column_names)

    targs = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, optim="paged_adamw_8bit", report_to="none")
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True))
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    metrics = trainer.evaluate()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    reg = Registry()
    params = {**vars(args), "train_seconds": round(train_s, 1),
              "train_examples": len(ds["train"]), "train_data_hash": file_hash(f"{args.data}/train.jsonl"),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    reg.register_model(f"hf:{args.base}@{out_dir}", "qlora", base=args.base, params=params, artifact_uri=str(out_dir))
    reg.log_run(f"hf:{args.base}@{out_dir}", "training", {"eval_loss": metrics.get("eval_loss"),
                "train_seconds": train_s}, {"run": run}, params["train_data_hash"])
    (out_dir / "train_summary.json").write_text(json.dumps({"params": params, "metrics": metrics}, indent=2))
    print("Saved adapter to", out_dir)


if __name__ == "__main__":
    main()
