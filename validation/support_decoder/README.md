# MAP versus support-only decoding

    python validation/support_decoder/run_support_ablation.py --dataset-root /path/to/BSDS500/data

The experiment replays the retained greedy zero-bit schedules at budgets
8 and 12 with 1,000 trials for each participant/domain condition. It uses
the original seed, random coordinate union, and random draw order.
All 8,000 MAP trial values must match the retained records exactly.

MAP accumulates float64 log likelihoods. Support-only retains all candidates
with positive likelihood at every observed position. Both choose the lowest
index on computed-score ties. At 12 zero-bit positions, support-only reaches
97.99%, close to MAP's 98.07%.

The true candidate must survive support filtering. Integer correct counts,
unique-survivor counts, total survivor counts, observations, and runtime
metadata are written under outputs/support_decoder or --output-dir.
Original CSV/JSON records remain under results/.
