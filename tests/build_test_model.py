"""Build a tiny local GPT-2 model + BPE tokenizer for offline testing.

No HuggingFace Hub download required: the model has random weights but a real
BPE tokenizer with offset mappings. Saved to ~/.cache/huggingface/tiny-gpt2-test
so the LLMFeatureExtractor code path (which checks "gpt2" in the model name)
loads it via GPT2LMHeadModel.from_pretrained.
"""
import json
from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelPostProcessor

OUT_DIR = Path.home() / ".cache" / "huggingface" / "tiny-gpt2-test"

# Representative English text so the BPE vocab covers test sentences.
# Repeated many times so the BPE trainer learns multi-character merges
# including the GPT-2-style space prefix (Ġ).
_BASE = [
    "The cat sat on the mat.",
    "The dog ran in the park.",
    "She said hello to the boy.",
    "A bird flew over the house.",
    "The sun is bright today.",
    "He reads a book by the lake.",
    "The car stopped at the red light.",
    "They walked through the forest.",
    "The water is cold and deep.",
    "A small child played with a toy.",
    "The old man told a story.",
    "Rain fell on the green grass.",
    "She opened the door and smiled.",
    "The train arrived at the station.",
    "Birds sing in the morning light.",
    "The dog and the cat are friends.",
    "He said the book is on the table.",
    "The sun sets behind the green hill.",
    "A small bird sang in the cold rain.",
    "The old car stopped at the station.",
]
CORPUS = _BASE * 50  # repeat to boost merge frequencies

def build_tokenizer():
    tok = Tokenizer(BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    trainer = BpeTrainer(
        vocab_size=1024,
        special_tokens=["<|endoftext|>", "<|unk|>"],
        initial_alphabet=ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer)
    # ByteLevel post-processor trims the leading space from offset mappings,
    # so word boundaries appear as gaps (what bpe_to_words_fast expects).
    tok.post_processor = ByteLevelPostProcessor(trim_offsets=True)
    fast = GPT2TokenizerFast(tokenizer_object=tok)
    fast.pad_token = "<|endoftext|>"
    return fast

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer()
    tok.save_pretrained(str(OUT_DIR))
    config = GPT2Config(
        vocab_size=tok.vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_positions=128,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(str(OUT_DIR))
    # Smoke test
    enc = tok("The cat sat on the mat.", return_tensors="pt")
    out = model(**enc)
    print(f"Saved tiny GPT-2 to {OUT_DIR}")
    print(f"  vocab_size={tok.vocab_size}  n_params={sum(p.numel() for p in model.parameters())}")
    print(f"  is_fast={tok.is_fast}")
    print(f"  logits shape={tuple(out.logits.shape)}")
    print(f"  offset_mapping supported: {'offset_mapping' in tok('test', return_offsets_mapping=True)}")

if __name__ == "__main__":
    main()
