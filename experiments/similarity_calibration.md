# Similarity-threshold calibration (2026-11-12)

**Status: completed; learned thresholds were not adopted.**

We tested whether independently labeled Figshare images could calibrate duplicate detection or infer patient groups for images without known patient IDs. This was a data-quality experiment supporting the inference-performance study, not an additional model-training objective.

## Records and reproduction

- [W&B run and full artifact](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/ndapbsmy): `brain-mri-similarity-calibration`, containing code, source archive identities/checksums, image/patient partitions, controlled-copy scores, threshold grid, frozen selection, held-out results, verification, and a comparison figure.
- Cluster working outputs: `~/brain-mri-vlm/data/threshold-calibration-v1/` (ignored by Git).
- This note is the tracked summary. Generated CSVs, plots, source images and archives remain outside Git.

The archived `calibrate.py` uses the original Figshare archives and the full-recovery candidate CSV to identify known benchmark-test patients. It requires the data-preparation dependencies plus `h5py==3.16.0`; recorded numerical package versions are in `results.json`. `plot_results.py` renders those saved results using `matplotlib==3.11.2`. A replay needs the archived inputs/cache layout and a fresh output directory; it does not require a GPU. Run cluster work inside a Slurm compute allocation.

## Design

Used 3,064 original images and 233 original patient IDs, rather than treating our already accepted image matches as independently labeled positives.

- Blocked all 125 patients already matched to benchmark test images from threshold fitting. Unrecognized benchmark patient overlap cannot be ruled out.
- Deterministically split the remaining patients by source tumor class, seed **20261112**. Calibration: **75 patients / 638 images**; held-out evaluation: **158 patients / 2,426 images**.
- Features: 64-bit pHash distance and normalized grayscale correlation at 128 x 128.
- Duplicate positives: known JPEG compression, resizing, intensity adjustment and 2% crop variants of each original. Different original image IDs served as controls, excluding identical pixel hashes. Those IDs are not exhaustive manual near-duplicate annotations.
- Patient positives: different original images from the same patient. Other patients were negative controls, with same-tumor-class controls reported separately.
- High-precision selection targeted at most **1% of calibration images receiving any negative match**. A separate relaxed rule maximized balanced pair accuracy to examine the tradeoff.
- Threshold selections were saved before held-out evaluation. A two-degree rotation was excluded from threshold fitting and used as a stress test.

## Held-out results

| Duplicate rule | pHash / correlation | Known-copy recall | Images with a distinct-image false match |
| --- | --- | ---: | ---: |
| Current reference | <= 4 / >= 0.98 | 74.99% | 0 of 2,426 observed |
| Calibrated candidate | <= 6 / >= 0.95 | 75.19% | 26 of 2,426 (1.07%) |

Recall is an **equally weighted four-transformation control average**, not estimated real-world duplicate recall. Both rules caught virtually all tested JPEG, resize and brightness/contrast variants. Crop recall was 0% vs 0.74%; rotation-stress recall was 9.44% vs 38.46%. The candidate's held-out false-match rate slightly exceeded the intended 1% bound. Its distinct-image links did not merge different known patients in this evaluation, but did join different source image IDs. Zero observed errors for the current rule is not a population guarantee.

| Patient rule | pHash / correlation | Same-patient pair recall | Images with any wrong-patient match | Grouping outcome |
| --- | --- | ---: | ---: | --- |
| Cautious calibrated | <= 4 / >= 0.80 | 2.12% | 8 of 2,426 (0.33%) | 4 mixed-patient components; 155 of 158 patients fragmented |
| Relaxed diagnostic | <= 24 / >= 0.50 | 43.82% | 97.32% | Different patients collapse into shared components |

Pair recall alone is not a clustering metric: the cautious rule gave 27.85% of eligible images a correct same-patient neighbor, but the component checks still showed poor patient reconstruction. The relaxed rule's same-class balanced accuracy was 60.35%.

## Decision and limitations

Keep the current duplicate thresholds and the patient keys recovered through strong original-image correspondences. Do not fill unknown IDs by relaxing appearance thresholds. These results do not invalidate source-copy matches; recognizing the same image and recognizing a patient across different slices are different tasks.

The modest controlled duplicate-recall gain did not justify an automatic policy change. Cropping/rotation robustness remains limited. This experiment covers one source and three tumor classes, with no no-tumor patients or external acquisition sources. Negative controls were original images, not a comprehensive transformed-negative suite. Query-level error depends on gallery size, and the sample is not a certification of generalization to other datasets.

Current preprocessing config, recovered mappings, source labels and manifests were unchanged; existing manifest checksums were verified. Outstanding manual-review items from recovery are separately archived in the [full-recovery run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/fniput3c): 11 patient-ID candidates withheld after higher-resolution checking, and 13 source-label disagreements. Those are not the calibration labels.
