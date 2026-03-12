#!/usr/bin/env python3
"""Visualize memory profiling data for Socratic AI Tutor — Online vs Offline modes."""

import matplotlib.pyplot as plt
import numpy as np
import os

# ── Data ──────────────────────────────────────────────────────────────────────
categories = ["Java", "Native", "Graphics", "Code", "Others"]

# Values in MB
online = {
    "Java": 15.6, "Native": 72.4, "Graphics": 85.8,
    "Code": 53.7, "Others": 251.4, "Total": 479.1,
}
offline_before = {
    "Java": 7.9, "Native": 784.9, "Graphics": 55.6,
    "Code": 27.1, "Others": 183.8, "Total": 1059.3,  # ~1 GB
}
offline_after = {
    "Java": 14.3, "Native": 630.2, "Graphics": 87.8,
    "Code": 33.4, "Others": 201.0, "Total": 966.9,
}

modes = ["Online Mode\n(No Model)", "Offline\n(n_ctx=2048)", "Offline Optimized\n(n_ctx=512)"]
colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]

out_dir = os.path.dirname(os.path.abspath(__file__))

# ── Figure 1: Stacked bar — memory breakdown ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

data = np.array([
    [online[c] for c in categories],
    [offline_before[c] for c in categories],
    [offline_after[c] for c in categories],
])

x = np.arange(len(modes))
width = 0.5
bottom = np.zeros(len(modes))

for i, cat in enumerate(categories):
    bars = ax.bar(x, data[:, i], width, bottom=bottom, label=cat, color=colors[i])
    # Add value labels on larger segments
    for j, val in enumerate(data[:, i]):
        if val > 40:
            ax.text(x[j], bottom[j] + val / 2, f"{val:.0f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
    bottom += data[:, i]

# Total labels on top
for j, mode_data in enumerate([online, offline_before, offline_after]):
    ax.text(x[j], bottom[j] + 15, f"{mode_data['Total']:.0f} MB",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Memory Usage (MB)", fontsize=12)
ax.set_title("Memory Usage Breakdown — Online vs Offline Modes\n(Huawei 4 GB RAM, Qwen3-0.6B Q4_K_M)", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(modes, fontsize=11)
ax.legend(loc="upper left", fontsize=10)
ax.set_ylim(0, max(bottom) + 80)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
path1 = os.path.join(out_dir, "memory_breakdown.png")
fig.savefig(path1, dpi=150)
print(f"Saved: {path1}")

# ── Figure 2: Native memory comparison (key metric) ──────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

native_vals = [online["Native"], offline_before["Native"], offline_after["Native"]]
bar_colors = ["#27AE60", "#E74C3C", "#F39C12"]
bars = ax2.bar(modes, native_vals, color=bar_colors, width=0.5, edgecolor="white", linewidth=1.5)

for bar, val in zip(bars, native_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
             f"{val:.1f} MB", ha="center", va="bottom", fontsize=12, fontweight="bold")

# Annotations
ax2.annotate(
    f"Model overhead\n+712.5 MB",
    xy=(1, offline_before["Native"]),
    xytext=(1.6, 600),
    fontsize=10, ha="center",
    arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5),
    color="#E74C3C", fontweight="bold",
)
ax2.annotate(
    f"After optimization\n−154.7 MB (−20%)",
    xy=(2, offline_after["Native"]),
    xytext=(2.5, 450),
    fontsize=10, ha="center",
    arrowprops=dict(arrowstyle="->", color="#F39C12", lw=1.5),
    color="#2E7D32", fontweight="bold",
)

ax2.set_ylabel("Native Memory (MB)", fontsize=12)
ax2.set_title("Native Memory: LLM Model Impact\n(KV cache is the main memory consumer)", fontsize=13, fontweight="bold")
ax2.set_ylim(0, 900)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
path2 = os.path.join(out_dir, "native_memory_comparison.png")
fig2.savefig(path2, dpi=150)
print(f"Saved: {path2}")

# ── Figure 3: Optimization summary table ─────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.axis("off")

table_data = [
    ["Metric", "Online", "Before Opt.", "After Opt.", "Change"],
    ["Total Memory", "479.1 MB", "1,059 MB", "966.9 MB", "−92.4 MB (−8.7%)"],
    ["Native Memory", "72.4 MB", "784.9 MB", "630.2 MB", "−154.7 MB (−19.7%)"],
    ["KV Cache (n_ctx)", "—", "2048", "512", "4× smaller"],
    ["Think Tokens", "—", "Yes (~30-50%)", "/no_think", "Eliminated"],
    ["Max Tokens", "—", "Unlimited", "150", "Bounded"],
    ["Inference (release)", "N/A", "4-7s", "~4-7s (est.)", "Same"],
    ["Conversation Length", "Unlimited", "6+ turns", "6+ turns", "Maintained"],
]

table = ax3.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

# Style header
for j in range(5):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(table_data)):
    color = "#F7F9FC" if i % 2 == 0 else "white"
    for j in range(5):
        table[i, j].set_facecolor(color)

# Highlight the "Change" column
for i in range(1, len(table_data)):
    table[i, 4].set_text_props(fontweight="bold", color="#27AE60")

ax3.set_title("Optimization Summary — Socratic AI Tutor (Offline Mode)",
              fontsize=13, fontweight="bold", pad=20)

plt.tight_layout()
path3 = os.path.join(out_dir, "optimization_summary.png")
fig3.savefig(path3, dpi=150, bbox_inches="tight")
print(f"Saved: {path3}")

print("\nDone! 3 charts generated.")
