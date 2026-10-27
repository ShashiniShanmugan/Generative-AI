"""Story generator: Flask backend serving a locally loaded GPT-2 (Hugging Face transformers).

The model weights are downloaded once from the Hugging Face Hub (or loaded from a local
directory via GPT2_MODEL_PATH) and all generation runs on this machine. No hosted
text-generation API is used.
"""
import os
import time

import torch
from flask import Flask, jsonify, render_template, request
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, set_seed

MODEL_NAME = os.environ.get("GPT2_MODEL_PATH", "gpt2")  # "gpt2" = 124M params; try "gpt2-medium"
MAX_NEW_TOKENS_LIMIT = 400
DEFAULT_SEED = 42

# GPT-2 is not instruction-tuned, so "style" is steered by a short priming preamble that
# the model continues from. The preamble is stripped from the displayed story.
STYLES = {
    "fantasy": "From an epic fantasy novel of ancient kingdoms, dragons and magic.\n\n",
    "sci-fi": "From a science fiction novel set among the stars, with strange technology.\n\n",
    "horror": "From a chilling horror story full of dread and shadows.\n\n",
    "mystery": "From a classic detective mystery full of clues and suspects.\n\n",
    "romance": "From a tender romance novel about love and longing.\n\n",
    "fairy tale": "Once upon a time, in a fairy tale told to children at bedtime.\n\n",
    "noir": "From a hard-boiled noir crime story, rain-slicked streets and cynical narration.\n\n",
    "none": "",
}

app = Flask(__name__)

_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Lazy-load tokenizer and model once (cached on disk by Hugging Face)."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
        _tokenizer.pad_token = _tokenizer.eos_token  # GPT-2 has no pad token
        _model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
    return _tokenizer, _model


def model_info():
    tok, model = load_model()
    cfg = model.config
    return {
        "name": MODEL_NAME,
        "parameters": sum(p.numel() for p in model.parameters()),
        "layers": cfg.n_layer,
        "heads": cfg.n_head,
        "hidden_size": cfg.n_embd,
        "context_window": cfg.n_positions,
        "vocab_size": cfg.vocab_size,
        "device": _device,
    }


def generate_story(opening, style="none", temperature=0.8, max_new_tokens=150,
                   top_k=50, top_p=0.95, seed=DEFAULT_SEED):
    """Generate a continuation of `opening` with autoregressive sampling.

    Returns a dict with the story text and token statistics.
    temperature <= 0 falls back to greedy decoding (deterministic).
    """
    opening = (opening or "").strip()
    if not opening:
        raise ValueError("Opening sentence must not be empty.")
    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from {sorted(STYLES)}.")
    max_new_tokens = int(max(1, min(max_new_tokens, MAX_NEW_TOKENS_LIMIT)))

    tok, model = load_model()
    preamble = STYLES[style]
    enc = tok(preamble + opening, return_tensors="pt")
    context = model.config.n_positions
    # Keep room for generation inside GPT-2's 1024-token window.
    if enc["input_ids"].shape[1] + max_new_tokens > context:
        keep = context - max_new_tokens
        enc = {k: v[:, -keep:] for k, v in enc.items()}
    enc = {k: v.to(_device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]

    set_seed(seed)  # same seed -> the only difference between runs is temperature
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature),
                      top_k=int(top_k), top_p=float(top_p))
    else:
        kwargs.update(do_sample=False)

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    elapsed = time.perf_counter() - start

    new_ids = out[0, prompt_len:]
    continuation = tok.decode(new_ids, skip_special_tokens=True)
    story = opening + continuation
    return {
        "story": story,
        "temperature": temperature,
        "style": style,
        "prompt_tokens": int(prompt_len),
        "generated_tokens": int(new_ids.shape[0]),
        "total_tokens": int(out.shape[1]),
        "story_tokens": len(tok.encode(story)),
        "seconds": round(elapsed, 2),
        "unique_word_ratio": _unique_ratio(story),
    }


def _unique_ratio(text):
    words = text.lower().split()
    return round(len(set(words)) / len(words), 3) if words else 0.0


def tokenize_preview(text):
    """Show how GPT-2's byte-level BPE splits text (useful for the tokenization demo)."""
    tok, _ = load_model()
    ids = tok.encode(text)
    return {"ids": ids, "tokens": tok.convert_ids_to_tokens(ids), "count": len(ids)}


@app.route("/")
def index():
    return render_template("index.html", styles=list(STYLES))


@app.route("/api/info")
def api_info():
    return jsonify(model_info())


@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    data = request.get_json(force=True)
    return jsonify(tokenize_preview(data.get("text", "")))


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True)
    try:
        temperature = float(data.get("temperature", 0.8))
        compare_temperature = float(data.get("compare_temperature", 1.5))
        common = dict(
            opening=data.get("opening", ""),
            style=data.get("style", "none"),
            max_new_tokens=int(data.get("max_new_tokens", 150)),
            top_k=int(data.get("top_k", 50)),
            top_p=float(data.get("top_p", 0.95)),
            seed=int(data.get("seed", DEFAULT_SEED)),
        )
        primary = generate_story(temperature=temperature, **common)
        comparison = generate_story(temperature=compare_temperature, **common)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"primary": primary, "comparison": comparison})


if __name__ == "__main__":
    load_model()  # fail fast / download weights on startup
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
