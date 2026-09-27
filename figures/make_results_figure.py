from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bound_summary.csv"
OUTPUT = Path(__file__).resolve().parent / "certificate_selection_results.pdf"
PREVIEW = Path(__file__).resolve().parent / "certificate_selection_results_preview.png"
MEASURED_BUDGETS = (8, 12, 16)
READINGS = ("full_field", "byte_domain")


def load_rows() -> list[dict[str, str]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_over_conditions(
    rows: list[dict[str, str]], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            grouped[int(row["budget"])].append(float(row["bound"]))
    budgets = np.asarray(sorted(grouped), dtype=float)
    values = np.asarray([grouped[int(b)] for b in budgets], dtype=float)
    return budgets, values.mean(axis=1), values.min(axis=1), values.max(axis=1)


def domain_values(
    rows: list[dict[str, str]], reading: str, method: str, field: str = "bound"
) -> np.ndarray:
    return np.asarray(
        [
            float(row[field])
            for row in rows
            if row["reading"] == reading
            and row["method"] == method
            and int(row["budget"]) == 8
        ],
        dtype=float,
    )


def observed_top1(budget: int) -> tuple[float, float, float]:
    """Recompute the paper's CI from four-condition trial means, not pixels."""
    grouped: dict[int, dict[tuple[int, str], float]] = defaultdict(dict)
    for participant in (1, 2):
        stem = f"repeated_shadow_summary_p{participant}_b{budget}.json"
        with (ROOT / "data" / stem).open(encoding="utf-8") as handle:
            summary = json.load(handle)
        expected = {
            "gallery_prior": "uniform",
            "gallery_split": "test",
            "gallery_size": 200,
            "participant": participant,
            "budget": budget,
            "trials": 1000,
            "authentication_source": "100075.jpg",
            "candidate_branch": 0,
        }
        for key, value in expected.items():
            if summary[key] != value:
                raise ValueError(f"Unexpected {key} in {stem}: {summary[key]}")
        strata: dict[str, list[float]] = defaultdict(list)
        trial_path = ROOT / "data" / f"repeated_shadow_trials_p{participant}_b{budget}.csv"
        with trial_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["method"] != "greedy":
                    continue
                reading, trial = row["reading"], int(row["trial"])
                condition = (participant, reading)
                if reading not in READINGS or condition in grouped[trial]:
                    raise ValueError(f"Unexpected or duplicate condition in {trial_path}")
                value = float(row["top1"])
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"Accuracy outside [0,1] in {trial_path}")
                grouped[trial][condition] = value
                strata[reading].append(value)
        for reading in READINGS:
            if len(strata[reading]) != 1000 or not np.isclose(
                np.mean(strata[reading]), summary["readings"][reading]["top1"]["greedy"]["mean"],
                rtol=0.0, atol=1e-12,
            ):
                raise ValueError(f"Trial/summary mismatch in {stem}, {reading}")
    expected_conditions = {(participant, reading) for participant in (1, 2) for reading in READINGS}
    if set(grouped) != set(range(1000)) or any(
        set(conditions) != expected_conditions for conditions in grouped.values()
    ):
        raise ValueError("Each of the 1,000 trials must contain all four conditions")
    trial_means = np.asarray([np.mean(list(grouped[t].values())) for t in range(1000)])
    mean = float(trial_means.mean())
    half_width = float(1.96 * trial_means.std(ddof=1) / np.sqrt(len(trial_means)))
    return mean, mean - half_width, mean + half_width


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot existing ICASSP linkage evidence.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT.parent)
    parser.add_argument("--panel-a-only", action="store_true",
                        help="Export the single-column linkage-success panel; preserve the two-panel source.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Keep a fixed page box: tight-bbox export can silently scale font sizes
    # down when LaTeX subsequently fits the figure to the paper's text width.
    if args.panel_a_only:
        fig, axis = plt.subplots(figsize=(3.38, 1.75))
        axes = np.asarray([axis])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.35))
    ax = axes[0]
    budgets = np.asarray(MEASURED_BUDGETS, dtype=float)
    observed = np.asarray([observed_top1(budget) for budget in MEASURED_BUDGETS])
    all_budgets, all_bounds, _, _ = aggregate_over_conditions(rows, "greedy")
    bound_by_budget = dict(zip(all_budgets.astype(int), all_bounds))
    bounds = np.asarray([bound_by_budget[budget] for budget in MEASURED_BUDGETS])
    for participant in (1, 2):
        for budget in MEASURED_BUDGETS:
            summary_path = ROOT / "data" / f"repeated_shadow_summary_p{participant}_b{budget}.json"
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            for reading in READINGS:
                source_bounds = [
                    float(row["bound"]) for row in rows
                    if int(row["participant"]) == participant and row["reading"] == reading
                    and int(row["budget"]) == budget and row["method"] == "greedy"
                ]
                if len(source_bounds) != 1 or not np.isclose(
                    source_bounds[0], summary["readings"][reading]["bounds"]["greedy"],
                    rtol=0.0, atol=1e-12,
                ):
                    raise ValueError(f"Certificate mismatch in {summary_path}, {reading}")

    highlight = MEASURED_BUDGETS.index(12)
    observed_label = (
        f"Observed ({100.0 * observed[highlight, 0]:.2f}%)"
        if args.panel_a_only else "Observed top-1"
    )
    bound_label = (
        f"Mean lower bound ({100.0 * bounds[highlight]:.2f}%)"
        if args.panel_a_only else "Analytical lower bound"
    )
    observed_handle = ax.errorbar(
        budgets, 100.0 * observed[:, 0],
        yerr=100.0 * np.vstack((observed[:, 0] - observed[:, 1], observed[:, 2] - observed[:, 0])),
        label=observed_label, color="#1F4E79", linestyle="-", marker="o",
        markersize=4.0, linewidth=1.35, capsize=2.5, zorder=4,
    )
    bound_handle, = ax.plot(
        budgets, 100.0 * bounds, label=bound_label,
        color="#B26A00" if args.panel_a_only else "#1F4E79",
        linestyle="--", marker="s", markerfacecolor="white", markersize=4.0,
        linewidth=1.25, zorder=3,
    )
    prior_handle = ax.axhline(100.0 / 200, color="#777777", linestyle=":", linewidth=1.2,
                             label="Chance level (0.5%)" if args.panel_a_only else "Prior-only (0.5%)")
    if not args.panel_a_only:
        ax.text(
            0.97, 0.87,
            "$B=12$:\n"
            f"{100.0 * observed[highlight, 0]:.2f}% observed\n"
            f"{100.0 * bounds[highlight]:.2f}% lower bound",
            transform=ax.transAxes, ha="right", va="top", color="#1F4E79",
        )
        ax.set_title("(a) One shadow, 200 candidates")
    ax.set_xlabel("Number of pixels ($B$)" if args.panel_a_only else "Selected positions $B$")
    ax.set_ylabel("Attack success rate (%)" if args.panel_a_only else "Linkage success (%)")
    ax.set_xticks(MEASURED_BUDGETS)
    ax.set_xlim(7.5, 17.0)
    ax.set_ylim(-2, 103)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.legend(handles=[observed_handle, bound_handle, prior_handle], frameon=False,
              title="Values at 12 pixels" if args.panel_a_only else None,
              title_fontsize=9.5,
              loc="lower right", bbox_to_anchor=(1.0, 0.12))

    if not args.panel_a_only:
        ax = axes[1]
        reading_keys = ["full_field", "byte_domain"]
        labels = ["Full field", "Byte domain"]
        x = np.arange(len(reading_keys), dtype=float)
        random_p05 = np.asarray(
            [domain_values(rows, key, "random_auth0", "p05").min() for key in reading_keys]
        )
        random_p50 = np.asarray(
            [domain_values(rows, key, "random_auth0", "p50").mean() for key in reading_keys]
        )
        random_p95 = np.asarray(
            [domain_values(rows, key, "random_auth0", "p95").max() for key in reading_keys]
        )
        random_max = np.asarray(
            [
                domain_values(rows, key, "random_auth0", "maximum").max()
                for key in reading_keys
            ]
        )
        ax.vlines(
            x,
            100.0 * random_p05,
            100.0 * random_p95,
            color="#777777",
            linewidth=5.0,
            alpha=0.38,
            label="Random envelope",
        )
        ax.scatter(x, 100.0 * random_p50, color="#555555", marker="_", s=45,
                   label="Random median", zorder=3)
        ax.scatter(x, 100.0 * random_max, color="#555555", marker="x", s=24,
                   label="Random maximum", zorder=3)
        comparison = [
            ("static", "Static overlap", "#C44E52", "s", -0.12),
            ("mi_rank", "MI rank", "#B26A00", "D", 0.0),
            ("greedy", "Greedy", "#1F4E79", "o", 0.12),
        ]
        for method, label, color, marker, offset in comparison:
            values = [domain_values(rows, key, method) for key in reading_keys]
            mean = np.asarray([item.mean() for item in values])
            low = np.asarray([item.min() for item in values])
            high = np.asarray([item.max() for item in values])
            ax.errorbar(
                x + offset,
                100.0 * mean,
                yerr=np.vstack((100.0 * (mean - low), 100.0 * (high - mean))),
                color=color,
                marker=marker,
                markersize=4.0,
                linewidth=1.0,
                capsize=2.0,
                linestyle="none",
                label=label,
                zorder=4,
            )
        ax.set_title("(b) Position selection, $B=8$")
        ax.set_ylabel("Analytical certificate (%)")
        ax.set_xticks(x, labels=labels)
        ax.set_xlim(-0.45, 1.45)
        # Reserve a clear strip above the data (maximum below 32%) for the legend.
        ax.set_ylim(22.5, 37.5)
        ax.set_yticks([24, 28, 32, 36])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.6,
                  handlelength=1.25, handletextpad=0.35, borderpad=0.15,
                  labelspacing=0.3)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    if args.panel_a_only:
        axes[0].set_yticks([0, 50, 100])
        fig.subplots_adjust(left=0.165, right=0.985, bottom=0.265, top=0.975)
        output_name = "single_shadow_linkage_results.pdf"
        preview_name = "single_shadow_linkage_results_preview.png"
    else:
        fig.tight_layout(w_pad=1.2, pad=0.35)
        output_name, preview_name = OUTPUT.name, PREVIEW.name
    fig.savefig(args.output_dir / output_name)
    fig.savefig(args.output_dir / preview_name, dpi=220)
    print(json.dumps({
        "budgets": list(MEASURED_BUDGETS),
        "observed_top1_percent": (100.0 * observed[:, 0]).tolist(),
        "ci95_percent": (100.0 * observed[:, 1:]).tolist(),
        "analytical_bound_percent": (100.0 * bounds).tolist(),
        "prior_only_percent": 100.0 / 200,
    }, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
