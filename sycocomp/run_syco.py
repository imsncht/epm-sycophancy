import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_model_and_tokenizer(model_path_or_name):
    print(f"loading model and tokenizer : {model_path_or_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    return model, tokenizer


def generate_response(model, tokenizer, prompts, max_new_tokens=150):
    responses = []
    for prompt in tqdm(prompts, desc="generating open-weight responses"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generate_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        responses.append({
            "prompt": prompt,
            "response": generate_text,
        })
    return responses


def main():
    parser = argparse.ArgumentParser(description="run open-weight model inference for sycophancy evaluations")
    parser.add_argument("--model_name", type=str, required=True, help="hugging face id or model path")
    parser.add_argument("--input_file", type=str, required=True, help="json file containing list of prompt objects")
    parser.add_argument("--output_file", type=str, required=True, help="path to save output responses json")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    prompts = [item["prompts"] for item in data]
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    results = generate_response(model, tokenizer, prompts)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved responses to {args.output_file}")


if __name__ == "__main__":
    main()