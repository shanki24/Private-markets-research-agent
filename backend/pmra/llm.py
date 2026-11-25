"""Provider-agnostic LLM client with token, cost and latency accounting."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests

from . import config


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    calls: int = 0

    def add(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.latency_s += other.latency_s
        self.calls += other.calls
        return self


def price(model: str, tin: int, tout: int) -> float:
    pin, pout = config.PRICING.get(model, (0.0, 0.0))
    return (tin * pin + tout * pout) / 1_000_000


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class LLM:
    provider: str = field(default_factory=lambda: config.PROVIDER)
    model: str = field(default_factory=lambda: config.MODEL)
    temperature: float = 0.0
    _hf: object = None

    @property
    def offline(self) -> bool:
        return self.provider == "offline"

    @property
    def version_id(self) -> str:
        return "offline-extractive-v1" if self.offline else f"{self.provider}/{self.model}"

    def complete(self, system: str, user: str, max_tokens: int = 800) -> tuple[str, Usage]:
        if self.offline:
            raise RuntimeError("Offline provider has no generative model; callers use deterministic fallbacks.")
        t0 = time.perf_counter()
        if self.provider == "anthropic":
            text, tin, tout = self._anthropic(system, user, max_tokens)
        elif self.provider == "openai":
            text, tin, tout = self._openai(system, user, max_tokens)
        elif self.provider.startswith("hf:"):
            text, tin, tout = self._hf_local(system, user, max_tokens)
        else:
            raise ValueError(f"Unknown provider {self.provider}")
        dt = time.perf_counter() - t0
        return text, Usage(tin, tout, price(self.model, tin, tout), dt, 1)

    def _anthropic(self, system, user, max_tokens):
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": self.model, "max_tokens": max_tokens, "temperature": self.temperature,
                  "system": system, "messages": [{"role": "user", "content": user}]},
            timeout=120,
        )
        r.raise_for_status()
        d = r.json()
        text = "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")
        return text, d["usage"]["input_tokens"], d["usage"]["output_tokens"]

    def _openai(self, system, user, max_tokens):
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": self.model, "max_tokens": max_tokens, "temperature": self.temperature,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=120,
        )
        r.raise_for_status()
        d = r.json()
        return d["choices"][0]["message"]["content"], d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]

    def _hf_local(self, system, user, max_tokens):
        """hf:<base>[@<adapter>] runs a local (optionally QLoRA-adapted) model."""
        if self._hf is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            spec = self.provider[3:]
            base, _, adapter = spec.partition("@")
            tok = AutoTokenizer.from_pretrained(base)
            model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
            if adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter)
            model.eval()
            self._hf = (tok, model, torch)
        tok, model, torch = self._hf
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
        gen = out[0][ids.shape[1]:]
        return tok.decode(gen, skip_special_tokens=True), int(ids.shape[1]), int(gen.shape[0])
