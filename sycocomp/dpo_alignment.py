import json
import argparse
from tqdm import tqdm

def create_dpo_preference_pairs(input_eval_file, output_dpo_file):
    """construsts preference pairs for DPO training: 
    -if human is non-afferming(sd_human=0), preferred responses is non-sycophantic (sd=0)
    -if human is afferming (sd_human=1), preferred response is affermin (sd=1)"""
    with open(input_eval_file, "r") as f:
        data = json.load(f)

    dpo_pairs = []
    for item in tqdm(data, desc="creating DPO pairs"):
        prompt = item["prompt"]
        responses = item.get("responses", [])#expecting multiple candidates responses per prompt
        human_affermative = item.get("human_afferming", 0)

        non_sycophantic = [r for r in responses if r.get("sycophancy_score") == 0]
        sycophantic = [r for r in responses if r.get("sycophancy_score") == 1]

        if non_sycophantic and sycophantic:
            if human_affermative == 0:
                chosen = non_sycophantic[0]["text"]
                rejected = sycophantic[0]["text"]
            else:
                chosen = sycophantic[0]["text"]
                rejected = non_sycophantic[0]["text"]

            dpo_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
    with open(output_dpo_file, "w") as f:
        json.dump(dpo_pairs, f, indent=2)
    print(f"generated {len(dpo_pairs)} DPO preferences pairs saved to {output_dpo_file}")

def main():
    parser = argparse.ArgumentParser(description="generate DPO preferences from scored evaluations")
    parser.add_argument("--input_file", type=str, required=True, help="path to scored responses JSON")
    parser.add_argument("--output_file", type=str, required=True, help="path to save DPO JSON dataset")
    args = parser.parse_args()

    create_dpo_preference_pairs(args.input_file, args.output_file)

if __name__ == "__main__":
    main()