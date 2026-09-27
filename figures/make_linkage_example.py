"""Figure 1: a recorded TCSVT21 single-shadow linkage example.

Run from the repository root after generating the full-shadow examples:
    python figures/make_linkage_example.py --dataset-root /path/to/BSDS500/data

The example is fixed without inspecting outcomes: participant 1, full-field
reading, first saved trial, first lexicographic BSDS500 test image. The existing
12-position greedy schedule is loaded without reselection. The shadow is taken
from the independent source-rule generator's saved full-image trial, never
sampled from the likelihood tables. It is losslessly PNG-encoded and decoded
before the 12 values are scored against all 200 saved gallery images.

The figure shows the observed 12-value strip, audited likelihood scoring, and
selection of an existing gallery candidate. The six displayed thumbnails are all
candidates with positive likelihood after the first six positions, in gallery
order; they are a display subset, not the decoder's gallery. Only the matched
candidate's identifier is annotated. The prefix candidate counts printed by the
loader are diagnostics and are not plotted as a performance curve.

Input provenance is retained in validation/tcsvt21_generator/protocol.json
and outputs/tcsvt21_generator/{full_shadow_example_full_field.npz,
linkage_summary.json}.
This script checks the complete saved generator trace and replays its first
trial using the recorded seed. It changes no experimental inputs or aggregates.
The output contains the original 64x64 grayscale arrays and vector annotations;
no contrast enhancement, denoising, or synthesized image content is used.

The white paper layout distinguishes observed values, the source-rule model,
and likelihood scoring. W_b and the 12-position product follow the manuscript;
the model is not estimated from the observed shadow. Ranking uses a uniform prior.
Use preview_only=True to render only the preview PNG. The default layout
omits the left outer frame; show_shadow_frame=True adds that frame.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath
import numpy as np
from PIL import Image

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "scripts"))
sys.path.insert(0, str(PAPER / "validation" / "tcsvt21_generator"))
from run_repeated_linkage import build_channel, load_gallery
from generator import generate, source_accepts, source_shares

READING = "full_field"
PARTICIPANT = 1
TARGET = 0
TRIAL = 0
BUDGET = 12
BLUE = "#1F4E79"
AMBER = "#B26A00"
INK = "#222222"
GRAY = "#777777"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "mathtext.fontset": "stix",
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def load_example(generator_output_dir=None, dataset_root=None):
    folder = PAPER / "validation" / "tcsvt21_generator"
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    source_path = (generator_output_dir or PAPER / "outputs" / "tcsvt21_generator") / f"full_shadow_example_{READING}.npz"
    with np.load(source_path, allow_pickle=False) as stored:
        record = {name: stored[name].copy() for name in stored.files}
    gallery = record["secrets"]
    mask = record["public_mask"]
    names = record["filenames"]
    assert gallery.shape == (200, 64, 64)
    assert str(names[TARGET]) == "100007.jpg"
    assert names.tolist() == sorted(names.tolist())
    stop = protocol["coefficient_stops"][READING]
    assert stop == 257 and protocol["budget"] == BUDGET

    # All saved source executions, not just the plotted target, must be valid.
    x1, x2 = source_shares(gallery, record["coefficient"])
    assert np.array_equal(x1, record["shadow1"])
    assert np.array_equal(x2, record["shadow2"])
    assert source_accepts(gallery, record["coefficient"],
                          record["pattern1"], record["pattern2"]).all()
    assert np.array_equal((2 * x1.astype(np.int64) - x2) % 257, gallery)
    domain_index = list(protocol["coefficient_stops"]).index(READING)
    seed_parts = [protocol["base_seed"], 3, domain_index, TRIAL]
    replay = generate(gallery, mask, stop,
                      np.random.default_rng(np.random.SeedSequence(seed_parts)))
    for key, values in replay.items():
        assert np.array_equal(values, record[key]), f"Saved trace mismatch: {key}"

    # Cross-check saved gallery/mask against the same raw data, when available.
    dataset = Path(dataset_root) if dataset_root is not None else Path(protocol["dataset_root"])
    if dataset.is_dir():
        source_gallery, source_mask, auth_name = load_gallery(dataset, "test")
        assert np.array_equal(source_gallery, gallery.reshape(200, -1))
        assert np.array_equal(source_mask, mask.ravel())
        assert auth_name == protocol["authentication_source"]

    metadata = json.loads((PAPER / "data" /
        "repeated_shadow_summary_p1_b12.json").read_text(encoding="utf-8"))
    positions = np.asarray(metadata["readings"][READING]["greedy_positions"])
    assert metadata["gallery_size"] == len(gallery)
    assert metadata["authentication_source"] == protocol["authentication_source"]
    assert len(positions) == len(np.unique(positions)) == BUDGET
    assert np.all(mask.ravel()[positions] == 0)

    # Observe just participant 1; participant 2 above is used for validation only.
    shadow = record[f"shadow{PARTICIPANT}"][TARGET]
    buffer = io.BytesIO()
    Image.fromarray(shadow).save(buffer, format="PNG")
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        observed = np.asarray(decoded, dtype=np.uint8).copy()
    assert np.array_equal(observed, shadow)
    observations = observed.ravel()[positions]

    channels = build_channel(stop, PARTICIPANT)
    per_pixel = np.stack([
        channels[int(mask.ravel()[j])][gallery.reshape(200, -1)[:, j], y]
        for j, y in zip(positions, observations)
    ], axis=1)
    logs = np.full_like(per_pixel, -np.inf)
    np.log(per_pixel, out=logs, where=per_pixel > 0)
    cumulative = np.cumsum(logs, axis=1)
    counts = np.r_[len(gallery), np.isfinite(cumulative).sum(axis=0)]
    # Uniform prior, so MAP equals maximum likelihood; np.argmax uses first tie.
    prediction = int(np.argmax(cumulative[:, -1]))
    displayed = np.flatnonzero(np.isfinite(cumulative[:, 5]))
    assert prediction == TARGET and counts[-1] == 1
    assert len(displayed) == 6
    assert np.isfinite(cumulative[TARGET]).all()
    assert np.array_equal(np.isfinite(cumulative), np.cumprod(per_pixel, axis=1) > 0)
    print(json.dumps({
        "example_selection": "first trial, first gallery image, participant 1, full field",
        "target": str(names[TARGET]), "prediction": str(names[prediction]),
        "seed_sequence": seed_parts,
        "positions_row_col_zero_based": [divmod(int(j), 64) for j in positions],
        "observed_values": observations.tolist(),
        "nonzero_likelihood_candidate_counts": counts.tolist(),
        "displayed_after_six_pixels": names[displayed].tolist(),
        "source_trace_replay": "exact match",
        "png_round_trip": "exact match",
        "full_gallery_size": len(gallery),
        "target_log_likelihood": float(cumulative[TARGET, -1]),
    }, indent=2))
    return gallery, names, observed, positions, counts, displayed, prediction


def draw_figure(gallery, names, shadow, positions, counts, displayed, prediction,
                *, preview_only=False, show_shadow_frame=False):
    values = shadow.ravel()[positions]

    W, H, Y0 = 570, 200, 0
    paper_width_in = 178 / 25.4
    fig = plt.figure(figsize=(paper_width_in, paper_width_in * H / W),
                     facecolor="white")
    # Grouping lives behind the data images and all labels/connectors.
    background = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=0)
    background.set_xlim(0, W)
    background.set_ylim(Y0, Y0 + H)
    background.set_axis_off()
    regions = [(162, 188), (384, 182)]
    if show_shadow_frame:
        regions.insert(0, (4, 134))
    for x, width in regions:
        background.add_patch(FancyBboxPatch(
            (x, 8), width, 184, boxstyle="round,pad=0,rounding_size=3",
            facecolor="white", edgecolor="#CAD0D5", linewidth=0.65,
        ))
    layer = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=4)
    layer.set_xlim(0, W)
    layer.set_ylim(Y0, Y0 + H)
    layer.set_axis_off()


    def text(x, y, value, **kwargs):
        options = dict(fontsize=9, color=INK, va="center", ha="left")
        options.update(kwargs)
        return layer.text(x, y, value, **options)


    def connector(points, color=BLUE, curved=False, width=0.9):
        codes = ([PlotPath.MOVETO] + [PlotPath.CURVE4] * 3 if curved else
                 [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
        line = FancyArrowPatch(path=PlotPath(points, codes), arrowstyle="-|>",
                               mutation_scale=7, linewidth=width, color=color,
                               joinstyle="round", capstyle="round")
        layer.add_patch(line)


    def image(x, y, size, array, *, match=False, observed=False):
        # Original 64x64 grayscale samples; no contrast or content alteration.
        ax = fig.add_axes([x / W, (y - Y0) / H, size / W, size / H], zorder=1)
        ax.imshow(array, cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest", origin="upper")
        ax.set_xlim(-0.5, 63.5)
        ax.set_ylim(63.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(BLUE if match else "#B4BBC0")
            spine.set_linewidth(1.8 if match else 0.55)
        if match:
            # A second, thin frame is outside the data image, not a painted overlay.
            layer.add_patch(Rectangle((x - 2, y - 2), size + 4, size + 4,
                                      fill=False, edgecolor=BLUE, linewidth=0.45))
        if observed:
            rows, cols = np.divmod(positions, 64)
            for row, col in zip(rows, cols):
                mark = Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                                 edgecolor=AMBER, linewidth=0.9, zorder=3)
                mark.set_path_effects([
                    path_effects.Stroke(linewidth=1.8, foreground="white"),
                    path_effects.Normal(),
                ])
                ax.add_patch(mark)
        return ax


    # Equal-height semantic regions, with no stage numbers or colored banners.
    text(14, 179, "Observed shadow", fontsize=9.6, weight="bold")
    text(14, 166, "12 selected pixels", fontsize=9)
    image(14, 42, 114, shadow, observed=True)

    # Amber encodes exactly the same selected observations in the image and vector.
    text(256, 179, "Pixel values and scoring", fontsize=9.6,
         weight="bold", ha="center")
    strip_x, strip_y, cell, gap, cell_h = 175, 136, 12.3, 1.3, 20
    for number, value in enumerate(values):
        x = strip_x + number * (cell + gap)
        shade = float(value) / 255
        layer.add_patch(Rectangle((x, strip_y), cell, cell_h,
                                  facecolor=(shade, shade, shade),
                                  edgecolor="#B4BBC0", linewidth=0.35))
    text(256, 125, "12 observed byte values", ha="center", fontsize=9)
    # Image, observation vector, and gallery thumbnails share a common top edge.
    connector([(132, 146), (169, 146)], color=AMBER, width=0.85)
    text(150, 159, "Read", ha="center", fontsize=9, color=AMBER)
    # Observations enter scoring directly. They do not generate the audited model.
    connector([(338, 146), (344, 146), (344, 61), (339, 61)],
              color=AMBER, width=0.8)

    # Light semantic boundaries distinguish a model input from a computation.
    # Draw them behind labels and arrows; values and ranking remain unboxed.
    background.add_patch(FancyBboxPatch(
        (173, 82), 166, 31, boxstyle="round,pad=0,rounding_size=2",
        facecolor="#F3F7FA", edgecolor="#A9BFCE", linewidth=0.6,
    ))
    background.add_patch(FancyBboxPatch(
        (173, 29), 166, 43, boxstyle="round,pad=0,rounding_size=2",
        facecolor="white", edgecolor=BLUE, linewidth=0.65,
    ))
    # The source-rule audit independently supplies the observation distribution.
    text(256, 105, "Audited pixel distributions", fontsize=9,
         weight="bold", color=BLUE, ha="center")
    text(256, 91, r"$W_b(y\mid s)$", fontsize=10.4, color=BLUE, ha="center")
    connector([(256, 81), (256, 73)], color=BLUE, width=0.8)
    # A single computation formula connects audit to image identity.
    text(256, 64, "Likelihood scoring", fontsize=9.2,
         weight="bold", color=BLUE, ha="center")
    formula = text(
        256, 44,
        r"$\mathrm{score}(m)=\prod_{j\in J}W_{b_j}(y_j\mid s_{m,j})$",
        fontsize=9.5, ha="center",
    )
    connector([(256, 28), (256, 22)], color=BLUE, width=0.8)
    text(256, 15, "Rank 200 candidates", fontsize=9,
         weight="bold", color=BLUE, ha="center")

    text(398, 179, "Known candidate gallery", fontsize=9.6, weight="bold")
    text(398, 166, "200 images; 6 shown", fontsize=9)
    size, spacing = 45, 8
    match_anchor = None
    for order, index in enumerate(displayed):
        row, col = divmod(order, 3)
        x, y = 399 + col * (size + spacing), 111 - row * 64
        matched = int(index) == prediction
        image(x, y, size, gallery[index], match=matched)
        if matched:
            text(x + size / 2, y - 11, str(Path(str(names[index])).stem),
                 fontsize=9, color=BLUE, weight="bold", ha="center")
            match_anchor = (x - 4, y + size / 2)
    assert match_anchor is not None
    # Smooth, unambiguous selection edge; not a candidate-count or accuracy curve.
    connector([(308, 15), (392, 15), (360, match_anchor[1]), match_anchor],
              curved=True, width=1.1)

    # The established caption carries public-input timing and matched-preprocessing
    # assumptions. Keep only the outcome key inside this paper-width preview.
    text(475, 24, "Outline: matched image", fontsize=9, ha="center")

    # Guard the exact scoring notation at the designed paper-figure dimensions.
    fig.canvas.draw()
    formula_bounds = formula.get_window_extent(fig.canvas.get_renderer())
    formula_bounds = formula_bounds.transformed(layer.transData.inverted())
    assert 175 <= formula_bounds.x0 < formula_bounds.x1 <= 337, formula_bounds
    assert 31 <= formula_bounds.y0 < formula_bounds.y1 <= 57, formula_bounds

    output = PAPER / "figures" / "single_shadow_linkage_example.pdf"
    staged_output = PAPER / "tmp" / "build" / "single_shadow_linkage_example.pdf"
    preview = PAPER / "tmp" / "build" / "linkage_layout_sketch.png"
    staged_preview = preview.with_name("linkage_layout_sketch_render.png")
    staged_output.parent.mkdir(parents=True, exist_ok=True)
    if not preview_only:
        fig.savefig(staged_output, metadata={
            "Title": "A recorded TCSVT21 single-shadow linkage example",
            "Subject": "Audited observation model, likelihood scoring, and gallery matching",
            "Creator": "figures/make_linkage_example.py",
        })
    fig.savefig(staged_preview, dpi=220, facecolor="white")
    plt.close(fig)
    if not preview_only:
        staged_output.replace(output)
        print(f"Saved {output}")
    staged_preview.replace(preview)
    print(f"Preview {preview}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render the first recorded full-shadow linkage example.")
    parser.add_argument("--generator-output-dir", type=Path, default=PAPER / "outputs" / "tcsvt21_generator")
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    draw_figure(*load_example(args.generator_output_dir, args.dataset_root))
