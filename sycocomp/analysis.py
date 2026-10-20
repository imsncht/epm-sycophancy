import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import argparse

def compute_sycophancy_metrics(df, epsilon=1e-5, ets=1e-5):
    """compute baseline sycophancy (S_base), sycophancy shift (Delta S_abs),
    and dynamic/inherent contribution ratios (Phi_abs, Phi_ins)"""

    results = []

    grouped = df.groupby("item_id")
    for item_id, group in grouped:
        #get baseline neutral prompt score(P_0)
        base_row = group[group["prompt_type"] == "P_0"]
        if base_row.empty:
            continue
        s_base = base_row["score"].values[0]

        #evaluate across variant prompts (P_1 to P_6)
        variants = group[group["prompt_type"] != "P_0"]
        for _, row in variants.iterrows():
            s_variant = row["score"]
            delta_s_abs = s_variant - s_base
            s_total = s_variant

            #dynamic contribution ratio calculation
            phi_abs = (max(0, delta_s_abs) / (s_total + epsilon)) * 100.0
            phi_inh = (1.0 - (max(0, delta_s_abs) / (s_total + epsilon))) * 100.0

            results.append({
                "item_id": item_id,
                "prompt_type": row["prompt_type"],
                "s_base": s_base,
                "s_variant": s_variant,
                "delta_s_abs": delta_s_abs,
                "phi_abs": phi_abs,
                "phi_inh": phi_inh,
                "lsm_score": row.get("lsm_score", 0.0)
            })
    
    return pd.DataFrame(results)

def run_statistical_validation(metrics_df):
    #computes correlation coefficients between LSM and sycophancy scores
    pearson_r, pearson_p = stats.pearsonr(metrics_df["lsm_score"], metrics_df["s_variant"])
    spearman_rho, spearman_p = stats.spearmanr(metrics_df["lsm_score"], metrics_df["s_variant"])

    summary = {
        "mean_delta_s": float(metrics_df["delta_s_abs"].mean()),
        "mean_phi_abs": float(metrics_df["phi_abs"].mean()),
        "mean-phi_inh": float(metrics_df["phi_inh"].mmean()),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p)
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="run stat analysis and metric calculations")
    parser.add_argument("--input_file", type=str, required=True, help="csv or json dataset containing evaluations records")
    parser.add_argument("--output-file", type=str, help="json file to save statistical summary")
    args = parser.parse_args()

    if args.input_file.endswith(".json"):
        df = pd.read_json(args.input_file)
    else:
        df = pd.reac_csv(args.input_file)

    metrics_df = compute_sycophancy_metrics(df)
    summary_stats = run_statistical_validation(metrics_df)

    with open(args.output_summary, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("statistical analysis summary:")
    for k, v in summary_stats.items():
        print(f" - {k}: {v:.4f}")
    print(f"summary saved to {args.output_summary}")

if __name__ == "__main__":
    main()