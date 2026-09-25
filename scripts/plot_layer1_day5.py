from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/tables/day5_layer1_scores.csv"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,default=DEFAULT)
    ap.add_argument("--out",type=Path,default=ROOT/"results/figures/day5_layer1_score_distributions.png")
    args=ap.parse_args()
    df=pd.read_csv(args.input)
    df=df[(df.evaluation_type=="human_verification") & (df.enrollment_mode=="clean")]
    if df.empty: raise SystemExit("No human-verification rows available.")
    groups=list(df.codec.dropna().unique())
    fig,ax=plt.subplots(figsize=(11,6))
    data=[df.loc[df.codec==g,"score"].to_numpy() for g in groups]
    ax.boxplot(data,tick_labels=groups,showfliers=False)
    ax.set_xlabel("Codec condition")
    ax.set_ylabel("ECAPA cosine similarity")
    ax.set_title("Day 5 — Layer 1 speaker similarity by codec")
    ax.tick_params(axis="x",rotation=35)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=160)
    plt.close(fig)
    print(f"Saved: {args.out}")
if __name__=="__main__": main()
