import re
import json
import argparse

# Regular Expression Patterns for Modal and Temporal Dynamics
MODAL_PAT = re.compile(r"\b(would|could|should|might|must|may|can)\b", re.IGNORECASE)
TIME_PAT = re.compile(r"\b(always|never|frequently|sometimes|often|rarely)\b", re.IGNORECASE)

class LexicalConvergenceClassifier:
    def __init__(self, keyword_dict_path=None):
        if keyword_dict_path:
            with open(keyword_dict_path, "r") as f:
                self.keywords = json.load(f)
        else:
            # Fallback default dictionary for elaboration indicators
            self.keywords = {
                "elaboration_markers": ["specifically", "furthermore", "in addition", "to elaborate", "moreover"],
                "affirmation_markers": ["you are right", "exactly", "absolutely", "i agree", "valid point"]
            }

    def analyze_text(self, text):
        matches = {
            "elaboration_count": 0,
            "affirmation_count": 0,
            "modal_matches": len(MODAL_PAT.findall(text)),
            "time_matches": len(TIME_PAT.findall(text))
        }

        for kw in self.keywords.get("elaboration_markers", []):
            matches["elaboration_count"] += len(re.findall(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE))

        for kw in self.keywords.get("affirmation_markers", []):
            matches["affirmation_count"] += len(re.findall(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE))

        # Decision threshold logic based on keyword presence
        is_convergent = (matches["affirmation_count"] > 0) and (matches["elaboration_count"] > 0 or matches["modal_matches"] >= 2)

        return {
            "is_convergent": is_convergent,
            "metrics": matches
        }

    def process_dataset(self, input_file, output_file):
        with open(input_file, "r") as f:
            data = json.load(f)

        results = []
        for item in data:
            resp_text = item.get("response", "")
            classification = self.analyze_text(resp_text)
            
            item_res = dict(item)
            item_res["convergence_analysis"] = classification
            results.append(item_res)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Convergence classification completed. Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run rule-based lexical convergence classifier.")
    parser.add_argument("--input_file", type=str, required=True, help="JSON file with prompt-response entries")
    parser.add_argument("--output_file", type=str, required=True, help="JSON file to save classification outputs")
    parser.add_argument("--dictionary_file", type=str, default=None, help="Optional external JSON dictionary file")
    args = parser.parse_args()

    classifier = LexicalConvergenceClassifier(keyword_dict_path=args.dictionary_file)
    classifier.process_dataset(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
