import argparse
import os
import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_MODEL = "google/gemma-4-E4B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 in a local terminal session."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id to load. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--prompt",
        default="Explique en 5 phrases max pourquoi la VRAM est precieuse en inference locale.",
        help="User prompt to send to the model.",
    )
    parser.add_argument(
        "--system",
        default="You are a concise, technical assistant.",
        help="System prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Gemma 4 thinking mode.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.getcwd(), ".hf-cache"),
        help="Directory used for Hugging Face downloads.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only use files already present in the local Hugging Face cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", args.cache_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Gemma 4 test expects an NVIDIA GPU.")

    print(f"Loading model: {args.model}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__} | CUDA runtime: {torch.version.cuda}")

    processor = AutoProcessor.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=args.local_only,
    )
    print(
        "VRAM after load: "
        f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GiB allocated | "
        f"{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GiB reserved"
    )

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(raw_response)

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Parsed Response ===")
    if isinstance(parsed, dict):
        content = parsed.get("content")
        thought = parsed.get("thought")
        answer = parsed.get("answer")
        if thought:
            print("[thought]")
            print(textwrap.fill(str(thought), width=100))
            print()
        if answer:
            print("[answer]")
            print(answer)
        if content:
            print(content)
        if not thought and not answer and not content:
            print(parsed)
    else:
        print(parsed)


if __name__ == "__main__":
    main()
