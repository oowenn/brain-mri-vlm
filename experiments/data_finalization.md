# Conservative data checkpoint — 2026-11-12

Decision: use `data/prepared-v3-conservative-patients/` for downstream model work. The prior `data/prepared-v2-phash4-ncc98/` image audit and pair-review notebook remain unchanged. No source image or official test label was altered.

The policy in `configs/data_policy.json` pins the original inventory and Figshare mapping by SHA-256. `scripts/finalize_data.py` unions existing image groups with 1,778 confirmed source correspondences covering 212 recovered patients. It leaves 11 below-threshold identities and 5,411 unmatched identities blank. It excludes 12 original training images with source-label disagreements; the one test disagreement stays flagged with its official label. Training components connected to test are excluded, and remaining eligible components are split as whole groups with seed 42 and a 15% validation target per class.

| Class | Train | Validation | Test | Excluded |
| --- | ---: | ---: | ---: | ---: |
| glioma | 1,023 | 181 | 400 | 196 |
| meningioma | 890 | 157 | 400 | 353 |
| notumor | 684 | 121 | 400 | 595 |
| pituitary | 682 | 120 | 400 | 598 |
| Total | 3,279 | 579 | 1,600 | 1,742 |

All 7,200 source images are accounted for. Exclusion reasons are mutually exclusive: 1,730 connected to test and 12 source-label disagreements. Compared with the earlier patient-group projection, one clean meningioma training example remains eligible after explicitly excluding its conflicting annotation instead of quarantining the entire remaining component.

Validation on Insomnia CPU allocation 13877804: all 10 unit tests passed. An independent repeat into a temporary directory produced an identical report and byte-identical CSVs. Official test identities/labels were preserved, all 12 conflicting training examples were excluded, and all 11 withheld IDs remained blank. No recovered patient or accepted image constraint crosses retained splits. Unknown patient identities can still overlap, so this checkpoint must not be described as fully patient-disjoint.

Final inventory SHA-256: `ac8bc157c04710da31110c741fe8afa9984e78d9befaa18726d3d1727d3c824f`.

Published [W&B run](https://forge.coreweave.com/wandb/coms6998-team32-fall2026/brain-mri-vlm/runs/lnqg1zfw): `coms6998-team32-fall2026/brain-mri-vlm/brain-mri-manifests:v2`. The artifact includes final CSVs, summary, policy and finalization script, with the pinned recovery artifact recorded as an input. The user explicitly approved uploading the image-path, class-label and hashed patient-linkage metadata; no images or original patient IDs were uploaded.

Generated manifests, mapping and review CSVs stay under ignored `data/`; only code, policy, tests and this note are tracked. `label_disagreements.csv` and `withheld_identities.csv` in the final directory expose the pending cases without requiring clinical decisions from the team. The cluster notebook and its saved outputs were preserved.

Next checkpoint: request a GPU allocation, verify the model/runtime and freeze the model revision, then run the proposal's zero-shot baseline on these fixed manifests. Do not change similarity thresholds or split policy based on test classification results.
