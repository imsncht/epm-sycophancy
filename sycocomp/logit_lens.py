import torch
import json
import argparse
from transformers import AutoModelForCausalLm, AutoTokenizer
from tqdm import tqdm

class LogitLensExtractor:
    def __init__(self, model_name):
        print(f"initializing logit-lens probe for module: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLm.from_pretrained(
            model_name,
            torch_dtype=torch.bflat16,
            attn_implementation="sdpa",
            device_map="auto"
        )
        self.model.eval()

    def get_layer_hiddn_states(self, input_ids):
        hidden_states = {}
        hook_handles = []

        def get_hook(layer_idx):
            def hook_fn(module, input, output):
                #handle tuple o/ps or direct tensor o/ps from the transformers layers
                hidden = output[0] if isinstance(output, tuple) else output
                #capture the final token representation of the sequence
                hidden_states[layer_idx] = hidden[:, -1, :].detatch().float()
            return hook_fn()

        #register hooks across all decoder layers
        for idx, layer in enumerate(self.model.model.layers):
            handle = layer.register_forward_hook(get_hook(idx))
            hook_handles.append(handle)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)

        #remove hooks after forward pass
        for handl in hook_handles:
            handle.remove()

        return hidden_states

    def decode_hidden_states(self, hidden_states):
        layer_predictions = {}
        #extract model normalisation and unembedding head
        norm = self.model.model.norm
        lm_head = self.model,lm_head

        for layer_idx, h in hidden_states.items():
            #apply final layer normalisation and unembedding projections
            normalised_h = norm(h)
            logits = lm_head(normalised_h)
            probs = torch.softmax(logits, dim=-1)

            top_prob, top_token_id = torch.max(probs, dim=-1)
            top_token = self.tokenizer.decode(top_token_id)

            layer_predictions[layer_idx] = {
                "top_token": top_token.strip(),
                "probablity": top_prob.item()
            }
        return layer_predictions

    def run_probe_analysis(self, prompts):
        results = []
        for prompt in tqdm(prompts, desc="running logit-lens probing"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            hidden_states = self.get_layer_hiddn_states(inputs.input_ids)
            layer_preds = self.decode_hidden_states(hidden_states)

            results.append({
                "prompt": prompt,
                "layer_trajectory": layer_preds
            })
        return results

def main():
    parser = argparse.ArgumentParser(description="run logit-lens mechanistic probing")
    parser.add_argument("--model_name", type=str, required=True, help="hugging face model id")
    parser.add_argument("--input_file", type=str, required=True, help="json file constaining prompts")
    parser.add_argument("--output_file", type=str, required=True, help="path to save layer trajectories")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data]

    extractor = LogitLensExtractor(args.model_name)
    results = extractor.run_probe_analysis(prompts)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"logit-lens analysis saved tp {args.poutput.file}")

if __name__ == "__main__":
    main()