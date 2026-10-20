import os
import json
import argparse
from openai import OpenAI
from tqdm import tqdm

def generate_api_responses(input_file, output_file, model_name="gpt-4o"):
    client = OpenAI(api_key=os.environ.get("openai_api_key"))

    with open(input_file, "r") as f:
        data = json.load(f)

    results = []
    for item in tqdm(data, desc=f"quering API ({model_name})"):
        prompt = item["prompt"]
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "you are a helpful and objective assistant"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.0
            )
            reply = response.choices[0].message.content
        except Exception as e:
            print(f"error processing prompt: {e}")
            reply = ""

        results.append({
            "prompt": prompt,
            "response": reply,
            "model": model_name
        })

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved API response to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="run API inference for frontier models")
    parser.add_argument("--input_file", type=str, required=True, help="JSON file containing prompt onjects")
    parser.add_argument("--output_file", type=str, required=True, help="path to save output response JSON")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="API model identifier")
    args = parser.parse_args()


    generate_api_responses(args.input_file, args.output_file, args.model_name)

if __name__ == "__main__":
    main()
