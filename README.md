# When Valid Shadows Leak: Exact Privacy Audits of Secret Image Sharing

Code and numerical records for the manuscript by Meijuan Li, Zhe Cui,
Yuxuan Liu, Yidong Wang, Ziwen Wei, and Wei Deng.

The code establishes disjoint single-share released-output sets for
source-permitted (2,2) instances of TCSVT21, MBE22, and SBC24. The TCSVT21
image experiment uses an audited observation channel, a Bhattacharyya-based
success certificate, and target-independent greedy position selection.

In the 200-image BSDS500 closed-world gallery, 12 selected pixels from one
shadow achieve 98.07% mean top-1 identification across four equally weighted
participant/domain conditions. The certificate is at least 95.88% in each
condition; its four-condition mean is 95.96%.

## Quick start

Use Python 3.12 or 3.13. The exact witness checks need only the standard library:

    python scripts/check_cross_scheme_witnesses.py --summary
    python scripts/check_tcsvt21_trace.py
    python scripts/check_reported_results.py

The first command checks all eight scheme/interpretation/participant witnesses,
including accepted sets, released outputs, reconstruction, and the JPEG
quantization/stability arithmetic. The third recomputes headline statistics
from retained trial records.

For image experiments, create an isolated environment:

    python -m venv .venv
    # Linux/macOS: source .venv/bin/activate
    # Windows PowerShell: .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

Install requirements-core.txt instead if only the linkage and decoder
experiments are needed. SciPy is used for the independent-generator statistical
checks; Matplotlib is used for figures. The pinned NumPy/Pillow/SciPy versions
match the retained generator-validation run. Matplotlib matches figure generation.

## Data

Obtain BSDS500 from the [official Berkeley dataset page](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).
After extraction, supply the BSDS500/data directory containing:

    images/train/*.jpg
    images/test/*.jpg
    images/val/*.jpg

The test gallery contains all 200 test images; the held-out gallery contains
all 100 validation images. Preprocessing is Pillow grayscale conversion,
largest-square center crop, and 64-by-64 LANCZOS resize. Gallery filenames are
lexicographically sorted. The public mask is computed from training image
100075.jpg: strictly above its median is one, otherwise zero.

Dataset images and full-shadow examples containing gallery pixels are obtained
or regenerated locally. This repository contains scripts and numerical records.

## Reproduce the main linkage experiment

Run from the repository root. Replace /path/to/BSDS500/data with the data path:

    python scripts/run_repeated_linkage.py --dataset-root /path/to/BSDS500/data --participant 1 --trials 1000 --budget 12
    python scripts/run_repeated_linkage.py --dataset-root /path/to/BSDS500/data --participant 2 --trials 1000 --budget 12

Each command evaluates both coefficient domains. Repeat with --budget 8 and
--budget 16 for the other columns of Table 5. For its held-out column:

    python scripts/run_repeated_linkage.py --dataset-root /path/to/BSDS500/data --gallery-split val --participant 1 --trials 1000 --budget 8
    python scripts/run_repeated_linkage.py --dataset-root /path/to/BSDS500/data --gallery-split val --participant 2 --trials 1000 --budget 8

The default base seed is 20260830; there are 1,000 random schedules. Outputs
are saved under outputs/linkage, or in the directory given by --output-dir.
The retained paper records remain in data/.

For a short execution check, use --trials 2 --random-schedules 4. This checks
the pipeline; it is not the paper's 1,000-trial result.

The main attack samples fresh observations from the enumerated pixel channel.
The independent generator below instead performs complete image generation
with explicit coefficient rejection and lossless PNG write/read.

## Certificates and sensitivity

    python scripts/generate_bound_summary.py --dataset-root /path/to/BSDS500/data
    python scripts/generate_robustness.py --dataset-root /path/to/BSDS500/data

The first writes analytical certificate tables into data/ by default, with
--output, --branch-output, and --target-output available for separate reruns.
The second writes gallery-size and five-mask checks under outputs/robustness.
Greedy's (1-1/e) approximation applies to the submodular overlap objective F;
it does not directly apply to success probability or the clipped certificate.

## Decoder and independent-generator checks

    python validation/support_decoder/run_support_ablation.py --dataset-root /path/to/BSDS500/data
    python validation/auth1_pool/run_auth1_pool.py --dataset-root /path/to/BSDS500/data
    python validation/tcsvt21_generator/validate_generator.py --stage exact
    python validation/tcsvt21_generator/validate_generator.py --stage all --dataset-root /path/to/BSDS500/data
    python validation/tcsvt21_generator/diagnose_negative_control.py

Detailed protocols: [support decoder](validation/support_decoder/README.md),
[one-bit pool](validation/auth1_pool/README.md), and
[independent generator](validation/tcsvt21_generator/README.md).
Fresh outputs are saved under outputs/. Original numerical records are
preserved under validation/*/results/.

The JPEG witnesses check coefficient arithmetic and release-path rules.
They do not generate or decode JPEG share files. The full-image file
round-trip check uses TCSVT21 and lossless PNG.

## Figures

    python figures/make_results_figure.py --panel-a-only --output-dir outputs/figures

This recreates the results panel using retained numerical records.
After running the full-shadow generator, recreate the recorded linkage example:

    python figures/make_linkage_example.py --dataset-root /path/to/BSDS500/data

It replays the first stored full-shadow trial and verifies its source trace
before drawing the example. It writes the example PDF under figures/ and a
preview under tmp/build/.

## Statistical definitions and source semantics

The four conditions are two participants times two coefficient domains, with
equal weight. Main intervals first average the four conditions within each
trial and then use mean +/- 1.96 times the standard error across 1,000 trials.
Method differences are paired on the same target observations. Intervals
condition on the fixed gallery, public mask, preprocessing, and sampling law.

The linkage setting is closed-world: the known gallery contains the target,
preprocessing matches, and positions are chosen before the target identity
and shadow are observed. The main pixel channel uses independent uniform
coefficient proposals and fixes the sampled authentication pattern during
coefficient retries. Disjoint-output privacy counterexamples do not need a
probability model for accepted coefficients.

Source implementations here follow the audited paper rules. The independent
generator and the enumerated channel have separate acceptance implementations.
The predeclared stochastic negative control produced zero corrected
rejections; the post-hoc deterministic retry trace detects the incorrect
pattern-resampling control. Both results are retained.

Audited sources:

| Label | Source |
|---|---|
| TCSVT21 | X. Yan et al., A Common Method of Share Authentication in Image Secret Sharing, [DOI](https://doi.org/10.1109/TCSVT.2020.3025527) |
| MBE22 | [DOI](https://doi.org/10.3934/mbe.2022538), Eq. (2.7), Algorithm 1, Steps 4-11 |
| SBC24 | [DOI](https://doi.org/10.1109/TMM.2024.3407694), Algorithm 1, Eqs. (10), (13)-(20) |

## Published records and license

data/ and validation/*/results/ contain the retained CSV/JSON measurements.
Machine-specific paths in metadata were redacted; numerical settings,
measurements, seeds, timestamps, and original execution hashes remain intact.
Public entry-point changes concern imports, data-path arguments, and output
locations. release_manifest.json records original and published code hashes.

The code is released under the MIT license. Third-party datasets remain under
their respective terms. See CITATION.cff for software citation metadata.
