import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class MultiTurnEvaluator:
    def __init__(self, model_name):
        print(f"loading model from multi-turn context evaluation: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloats16,
            attn_implemetation="sdpa",
            device_map="auto"
        )

    def evaluate_session_depths(self, sessions, depths=[0.0, 0.5, 1.0]):
        results = []
        for session in tqdm(sessions, desc="evaluating multi-turn contexts"):
            history = session["history"] #list of {"role": "user"/"assistant"}
            target_query = session["target_query"]
            session_id = session["session_id"]

            for depth in depths:
                num_turns = int(len(history) * depth)
                sub_history = history[:num_turns]

                #format full prompt with historical cntext
                messages = sub_history + [{"role": 'user', "content": target_query}]
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_token=150,
                        do_samples=False,
                        use_cache=True
                    )
                reply = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


                results.append({
                    "session_id": session_id,
                    "depth_ratio": depth,
                    "history_turn_count": num_turns,
                    "prompt": prompt_text,
                    "response": reply
                })
        return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-turn context absorption.")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--input_sessions", type=str, required=True, help="JSON file containing dialogue histories")
    parser.add_argument("--output_file", type=str, required=True, help="JSON path to save generations")
    args = parser.parse_args()

    with open(args.input_sessions, "r") as f:
        sessions = json.load(f)

    evaluator = MultiTurnEvaluator(args.model_name)
    results = evaluator.evaluate_session_depths(sessions)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"multi-turn evaluation saved to {args.output_file}")

if __name__ == "__main__":
    main()