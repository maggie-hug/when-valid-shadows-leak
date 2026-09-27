# One-bit position-pool experiment

    python validation/auth1_pool/run_auth1_pool.py --dataset-root /path/to/BSDS500/data

The eligible pool contains the 1,982 positions with public mask bit one.
Greedy schedules are selected before observing targets. Budgets 8 and 12
share observations for their common positions. There are 1,000 trials for
each participant/domain condition, with base seed 20260830 and a separate
auth1_pool seed namespace.

At 12 positions, MAP reaches 6.76%, support-only reaches 0.54%, and the
analytical clipped certificate is zero. The experiment distinguishes
likelihood-driven leakage from support exclusion.

Fresh outputs are saved under outputs/auth1_pool or --output-dir. An existing
completed output is protected against overwriting. The original published
records are under results/. The script retains integer counts, observations,
input/data hashes, candidate ordering, and per-condition certificates.
