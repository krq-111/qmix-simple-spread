# -*- coding: utf-8 -*-
"""从 progress.csv 画训练曲线(reward 曲线 + loss 曲线)。

用法:
    python plot_results.py results/run1/progress.csv
    python plot_results.py results/run1/progress.csv --smooth 1000 --out my.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体(Windows 自带微软雅黑; 找不到时 matplotlib 会自动退到默认字体)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def moving_average(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("--smooth", type=int, default=500, help="滑动平均窗口(局)")
    p.add_argument("--out", type=str, default="", help="输出 png 路径")
    args = p.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    eps = np.array([int(r["episode"]) for r in rows])
    train_ret = np.array([float(r["train_return"]) for r in rows])
    loss = np.array([float(r["loss"]) if r["loss"] not in ("", "nan") else np.nan
                     for r in rows])
    ev_mask = np.array([r["eval_return"] not in ("",) and r["eval_return"] != ""
                        and "±" in r["eval_return"] for r in rows])
    ev_eps = eps[ev_mask]
    ev_ret = np.array([float(r["eval_return"].split("±")[0]) for r in rows
                       if "±" in r["eval_return"]])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.4]})

    # --- 上图: 回报曲线 ---
    ax1.plot(eps, train_ret, color="lightsteelblue", linewidth=0.5,
             alpha=0.7, label="训练回报(含探索噪声)")
    if len(train_ret) >= args.smooth:
        ax1.plot(eps[args.smooth - 1:], moving_average(train_ret, args.smooth),
                 color="royalblue", linewidth=2,
                 label=f"训练回报 滑动平均(窗口={args.smooth})")
    if len(ev_eps) > 0:
        ax1.plot(ev_eps, ev_ret, "o-", color="crimson", linewidth=1.5,
                 markersize=4, label="贪心评估回报")
    ax1.set_ylabel("每局团队回报 (return)")
    ax1.set_title("QMIX 在 PettingZoo simple_spread 上的训练曲线")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")

    # --- 下图: TD loss ---
    valid = ~np.isnan(loss)
    ax2.plot(eps[valid], loss[valid], color="darkorange", linewidth=0.8)
    if valid.sum() >= args.smooth:
        ax2.plot(eps[valid][args.smooth - 1:],
                 moving_average(loss[valid], args.smooth),
                 color="saddlebrown", linewidth=2, label="loss 滑动平均")
        ax2.legend(loc="upper right")
    ax2.set_ylabel("TD loss")
    ax2.set_xlabel("训练局数 (episode)")
    ax2.grid(alpha=0.3)

    out = args.out or args.csv_path.replace("progress.csv", "reward_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot  ] saved -> {out}")
    print(f"[stats ] train return: start≈{np.nanmean(train_ret[:50]):.1f}, "
          f"end≈{np.nanmean(train_ret[-50:]):.1f}; "
          f"best eval≈{np.max(ev_ret) if len(ev_ret) else float('nan'):.1f}")


if __name__ == "__main__":
    main()
