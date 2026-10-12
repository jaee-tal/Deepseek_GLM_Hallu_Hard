# Portable DeepSeek V4.1 Flash and GLM-5.3-Flash experiment

This folder is a self-contained copy of the hallucination-probe transfer experiment for:

- `deepseek_v41_flash` → `deepseek-ai/DeepSeek-V4.1-Flash`, pinned to revision
  `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`.
- `glm_53_flash` → `zai-org/GLM-5.3-Flash`, pinned to revision
  `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`.

It contains the code, fixed questions, fixed question-level splits, HalluHard source data,
dependency pins, checkpointing, W&B tracking, GCP monitoring configuration, and empty output
directories. Model weights are not copied into the folder; the pinned checkpoints download from
Hugging Face on first use.

## Experiment retained in this bundle

Each model is evaluated on PopQA, TriviaQA, and the permanent 700-question HalluHard split. The
HalluHard pool contains 250 medical-guideline, 250 research, 100 legal, and 100 coding questions,
split 420/140/140 at the question level.

The model first answers every question organically. GPT-mini through OpenRouter labels organic
truth and organic hallucination. Only questions that were always correct organically are eligible
for induced generation. The three error registers are `lie`, `misremember`, and `unknown`.

The pooled probes are difference of means, logistic regression, and LDA. Each register is fitted in
four arms:

1. `plain_inorganic`
2. `same_instruction`
3. `plain_inorganic_plus_organic`
4. `same_instruction_plus_organic`

The mixed arms use 20% of eligible organic training questions with 50% total organic and 50% total
inorganic training weight. Source arms select layers and hyperparameters on source validation;
mixed arms inherit the corresponding source selection. Thresholds maximize balanced accuracy on
source validation. The untouched organic test labels are used only for final evaluation.

## Hardware

These are large FP8 checkpoints. Use a machine on which the complete pinned model passes the
included model-load and activation-replay preflight. The intended cloud configuration is one
multi-GPU worker per model, such as an 8-GPU A4/H100-class instance. `device_map: auto` distributes
the checkpoint across visible GPUs. Run the two models on separate workers, or sequentially on one
worker. Do not launch both simultaneously on the same GPUs.

Allow enough local disk for both checkpoint downloads, generated responses, and replay shards.
Hourly experiment checkpoints exclude reproducible materialized activation arrays, but the local
activation-replay stage still needs substantial temporary disk space.

## Copy and set up

Copy this entire directory to the destination machine, then run:

```bash
cd deepseek_glm_experiment
cp .env.example .env
# Edit .env with the OpenRouter, Hugging Face, W&B, and GCP values.
bash scripts/setup.sh
```

`setup.sh` creates `.venv`, installs the pinned environment, creates all output directories, and
verifies the included fixed inputs. No credential is stored in this bundle.

Required online-run settings in `.env`:

```dotenv
OPENROUTER_API_KEY=...
HF_TOKEN=...
CROSS_PROBE_ARTIFACT_URI=gs://YOUR_BUCKET/cross-probe
CROSS_PROBE_CONTAINER_DIGEST=sha256:YOUR_IMMUTABLE_IMAGE_DIGEST
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_MODE=online
GOOGLE_CLOUD_PROJECT=...
CROSS_PROBE_ALERT_EMAIL=you@example.com
```

For a local run without remote checkpoint or W&B upload, use
`CROSS_PROBE_ARTIFACT_URI=file://cloud_artifacts` and `WANDB_MODE=offline`. OpenRouter remains
required for judging.

## Validate and run

Run the full model-load, generation, residual-stream replay, checkpoint, W&B, and alert preflight
before committing to the long run:

```bash
bash scripts/preflight_model.sh deepseek_v41_flash ds-glm-v1
bash scripts/preflight_model.sh glm_53_flash ds-glm-v1
```

Run one model on a worker:

```bash
bash scripts/run_model.sh deepseek_v41_flash ds-glm-v1
# On the other worker:
bash scripts/run_model.sh glm_53_flash ds-glm-v1
```

To run both sequentially on the same worker:

```bash
bash scripts/run_both_sequential.sh ds-glm-v1
```

The runner is idempotent. Generation resumes by deterministic conversation ID, judging resumes
from saved verdicts, and activation replay resumes from immutable row shards.

## Resume from GCS after interruption

On a replacement machine, copy this folder, configure the same `.env`, install the environment,
and restore using the same experiment ID and model:

```bash
bash scripts/setup.sh
bash scripts/resume_model.sh deepseek_v41_flash ds-glm-v1
```

The restore command reads the newest verified manifest from
`CROSS_PROBE_ARTIFACT_URI`. Checkpoints are written every hour and at every stage boundary. Objects
are content-addressed by SHA-256, and `LATEST.json` is updated only after the complete checkpoint
manifest exists.

## Status and outputs

```bash
bash scripts/status.sh deepseek_v41_flash ds-glm-v1
bash scripts/status.sh glm_53_flash ds-glm-v1
```

The bundle writes to these pre-created directories:

```text
data/hf_hub_cache/                         downloaded pinned model files
data/hf_datasets_cache/                    Hugging Face dataset cache
runs/<model>/<dataset>/<experiment-id>/    generations, judgments, replay, datasets
runs/deepseek_glm_cloud_v1/...             stage state, heartbeats, provenance
results/deepseek_glm_cloud_v1/...          final metrics, CSVs, and PNG figures
cloud_artifacts/                            local checkpoint fallback
wandb/                                      W&B local state
logs/                                       optional launcher logs
```

Remote checkpoints use a model-separated namespace below the configured bucket prefix, so both
models may safely use the same experiment ID.

## Optional container

```bash
docker build -t cross-probe-deepseek-glm:latest .
docker run --gpus all --env-file .env \
  -v "$PWD/data/hf_hub_cache:/workspace/cross_probe/data/hf_hub_cache" \
  -v "$PWD/runs:/workspace/cross_probe/runs" \
  -v "$PWD/results:/workspace/cross_probe/results" \
  cross-probe-deepseek-glm:latest \
  python3 scripts/30_cloud_experiment.py preflight \
    --config configs/cloud_experiment.yaml --experiment-id ds-glm-v1 \
    --model deepseek_v41_flash
```

Mount a Google service-account credential or use the VM service account rather than placing a key
inside the image.

## Monitoring and alerts

W&B receives one stable run per model and stage, plus progress, sample yields, selected layers,
metrics, and final figures. The runner writes a local heartbeat every minute and emits structured
alerts for failures, GPU OOMs, NVIDIA Xid errors, stale checkpoints, and stalled stages.

To create the GCP email-alert policies once:

```bash
cd infra/gcp
terraform init
terraform apply -var project_id="$GOOGLE_CLOUD_PROJECT" \
  -var alert_email="$CROSS_PROBE_ALERT_EMAIL"
```

The VM or container service account needs Storage object access for the bucket and Monitoring
metric-writer access for the project.

## Fixed-input regeneration

The fixed inputs are already included and should normally remain untouched. To reproduce them from
the bundled HalluHard sources and pinned Hugging Face dataset definitions:

```bash
bash scripts/prepare_inputs.sh
```

Preparation is additive: existing question split assignments are retained. It refuses to drop an
existing question unless explicitly forced.
