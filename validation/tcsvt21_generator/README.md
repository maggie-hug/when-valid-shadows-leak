# Independent source-rule generator

generator.py implements the TCSVT21 (2,2) primitive, hidden pattern draw,
polynomial evaluation, and coefficient retry loop. It receives secrets,
public bits, a coefficient domain, and an RNG. Accepted-set tables and
observation distributions are used only by the separate validator/attacker.

The frozen protocol is in protocol.json. Its numerical settings match the
retained execution; its original dataset path is replaced by a portable
placeholder. Pass the real path with --dataset-root for linkage/all.

    python validation/tcsvt21_generator/validate_generator.py --stage exact
    python validation/tcsvt21_generator/validate_generator.py --stage marginal
    python validation/tcsvt21_generator/validate_generator.py --stage linkage --dataset-root /path/to/BSDS500/data

Stages:

- exact exhausts 256 secrets, both domains, four hidden patterns, and hidden
  coin/permutation primitives; it compares against separately implemented channels.
- marginal draws 16,384 samples per secret/public-bit/domain combination,
  performs Pearson tests with Holm correction, and retains the deliberately
  incorrect pattern-resampling control.
- linkage generates 100 full-gallery trials per domain. Both complete
  participant shadows undergo lossless PNG write/read before the saved
  12-pixel schedules are applied. Every pixel is checked for reconstruction.

The four-condition source-minus-channel difference is -0.01775 percentage
points with a 95% interval of [-0.11542, 0.07992] percentage points, within
the predeclared +/-0.5-point margin. Detailed strata and integer correct counts
are in results/linkage_summary.json and results/full_shadow_trials.csv.

The original random negative-control tests returned zero corrected
rejections. The separate post-hoc deterministic trace detects incorrect
hidden-pattern resampling. Retained results/marginal_summary.json and
results/posthoc_diagnostics.json preserve both findings.

    python validation/tcsvt21_generator/diagnose_negative_control.py

Fresh outputs go to outputs/tcsvt21_generator or --output-dir. Large raw
histograms, complete generated examples, and PNG files are produced locally;
the published original evidence consists of CSV/JSON records.
