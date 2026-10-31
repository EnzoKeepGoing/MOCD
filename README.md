# MOCD

This repository contains the **MOCD** (Multi-level semantic-guided Online Cognitive Diagnosis) model, its five-fold configurations, mapped response data and training scripts for Grade3, Math50k and MidMath. No comparison models are included.

## Repository layout

```text
configs/                 Per-dataset, per-fold MOCD configurations
data/                    Mapped responses, Q-matrices and fixed ID mappings
MODE/                    MOCD model, semantic evidence readers and training
train.py                 Train and evaluate one fold
run_five_folds.py        Run folds 0–4 sequentially on one device
requirements.txt         Python dependencies
```

## Data and semantic assets

| Dataset | Directory | Students | Exercises | KCs | Train/validation responses | Test responses |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Grade3 | `Grade3` | 534 | 8,027 | 844 | 19,823 | 5,043 |
| Math50k | `Math50k` | 962 | 20,460 | 1,483 | 40,674 | 10,262 |
| MidMath | `MidMath` | 701 | 17,944 | 931 | 24,291 | 6,127 |

Each dataset has `train_valid.csv`, `test.csv`, `q_matrix.csv` and `id_mapping.json`. Response columns are `user_id,question_id,correct,fold`. For fold *f*, rows with `fold == f` are validation and the other rows in `train_valid.csv` are training. The same `test.csv` is used across folds. IDs in the responses, Q-matrix and semantic assets share the supplied mapping.

The precomputed semantic evidence is supplied separately as `mocd_semantic_assets_encoded_20261029.tar.gz`. This release provides mapped IDs and encoded vectors, without exercise text, reasoning text or KC names. Install it with:

```bash
python install_assets.py /path/to/mocd_semantic_assets_encoded_20261029.tar.gz
```

This creates the following directories:

```text
assets/Grade3/item_bank/
assets/Grade3/kc_bank/
assets/Math50k/item_bank/
assets/Math50k/kc_bank/
assets/MidMath/item_bank/
assets/MidMath/kc_bank/
```

The installer labels the encoded banks with the corresponding dataset names
(`Grade3`, `Math50k`, `MidMath`) and updates their metadata fingerprints.

The item banks contain the frozen exercise-text, correct-reasoning and error-reasoning embeddings and availability/reliability masks. The KC banks contain KC-name and item–KC evidence embeddings. Both bank readers check the data and array fingerprints before training. The assets are excluded from Git by `.gitignore` because of their size; training uses the released precomputed vectors and makes no API calls.

## Environment

Use Python 3.10 or newer. Install a PyTorch build compatible with the available CUDA runtime, then:

```bash
pip install -r requirements.txt
```

## Train one fold

```bash
python train.py \
  --config configs/MidMath_fold0.json \
  --data-dir data/MidMath \
  --item-bank assets/MidMath/item_bank \
  --kc-bank assets/MidMath/kc_bank \
  --output runs/MidMath/fold0 \
  --device cuda
```

Before training, run the same command with `--dry-run` to validate the real model inputs and one forward pass. Training selects the best validation-AUC epoch (maximum 30 epochs, patience 5), restores it, and evaluates the fixed test data. Each output directory receives `best_model.pt`, `mastery.npy`, `results.json` and `run_manifest.json`.

## Run all five folds

```bash
python run_five_folds.py --dataset Grade3 --device cuda --assets assets
python run_five_folds.py --dataset Math50k --device cuda --assets assets
python run_five_folds.py --dataset MidMath --device cuda --assets assets
```

Each command runs folds **sequentially** on one device. The launcher refuses to overwrite an existing fold directory. To smoke-test all fold configurations without training, append `--dry-run`.

## Five-fold reference configurations

All folds use a 30-dimensional student/KC embedding, batch size 128, up to 30 epochs and the full MOCD architecture. The complete settings, including router width, gate initialization, semantic scale and projection initialization, are in `configs/`.

| Dataset | Fold | Hidden layers | Dropout | Learning rate | Seed | β | Semantic temperature |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Grade3 | 0 | 256, 128 | 0.2, 0.5 | 0.0005 | 3407 | 13 | 2.0 |
| Grade3 | 1 | 512, 64 | 0.2, 0.5 | 0.0005 | 42 | 12 | 1.5 |
| Grade3 | 2 | 512, 64 | 0.2, 0.5 | 0.0007 | 42 | 12 | 2.0 |
| Grade3 | 3 | 512, 128 | 0.3, 0.3 | 0.0009 | 3407 | 13 | 2.0 |
| Grade3 | 4 | 512, 128 | 0.2, 0.3 | 0.0005 | 42 | 13 | 1.5 |
| Math50k | 0 | 512, 128 | 0.3, 0.5 | 0.0007 | 42 | 13 | 1.5 |
| Math50k | 1 | 512, 128 | 0.2, 0.5 | 0.0007 | 42 | 12 | 2.0 |
| Math50k | 2 | 256, 64 | 0.3, 0.5 | 0.0005 | 3407 | 13 | 2.0 |
| Math50k | 3 | 512, 128 | 0.3, 0.3 | 0.0009 | 42 | 12 | 1.5 |
| Math50k | 4 | 512, 128 | 0.3, 0.3 | 0.0009 | 42 | 13 | 2.0 |
| MidMath | 0 | 512, 64 | 0.3, 0.3 | 0.0005 | 3407 | 13 | 2.0 |
| MidMath | 1 | 256, 64 | 0.2, 0.2 | 0.0007 | 42 | 13 | 2.0 |
| MidMath | 2 | 512, 128 | 0.3, 0.3 | 0.0009 | 42 | 13 | 1.5 |
| MidMath | 3 | 512, 64 | 0.3, 0.2 | 0.0009 | 3407 | 13 | 2.0 |
| MidMath | 4 | 512, 64 | 0.3, 0.5 | 0.0005 | 3407 | 12 | 1.5 |

The fold-specific configuration JSON files are the executable parameter record.

## Five-fold reference results

Values are mean ± sample standard deviation across the five folds; all metrics are in percent.

| Dataset | AUC | ACC | KC-MND |
| --- | ---: | ---: | ---: |
| Grade3 | 74.37 ± 0.33 | 73.56 ± 0.20 | 6.43 ± 2.69 |
| Math50k | 73.29 ± 0.21 | 70.94 ± 0.15 | 7.56 ± 1.43 |
| MidMath | 75.48 ± 0.31 | 71.94 ± 0.29 | 5.08 ± 2.56 |

## Model and optimization

MOCD combines student–KC mastery and exercise–KC difficulty with three exercise semantic views (exercise text, correct reasoning and error analysis) and KC-level semantic evidence. The response objective is uncertainty-weighted BCE. The mastery-profile objective is `−V + βV²`, where KC weights come from frozen response-conditioned semantic evidence. Training applies the recorded module-wise gradient alignment to the two objectives, then updates the model with Adam. The model's semantic banks are fixed inputs, not online LLM requests.
