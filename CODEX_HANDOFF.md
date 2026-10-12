# DeepSeek / GLM hallucination-transfer experiment: Codex handoff

This document explains the experiment and the portable implementation to a Codex session with no previous conversation. It describes the files in `deepseek_glm_experiment/`, inspected on 2026-10-12. All paths below are relative to that folder, so the folder can move to another machine.

**Copy the entire `deepseek_glm_experiment/` directory to run the experiment.** This Markdown file provides context; it does not contain the Python implementation, datasets, credentials, or model weights.

## 1. Research question and history

Can a linear probe trained to recognize deliberately induced factual errors detect factual errors the model makes naturally?

The primary transfer direction is:

```text
Training:   induced truth vs induced hallucination
Evaluation: held-out organic truth vs organic hallucination
```

Earlier versions used Qwen3-8B and GPT-OSS-20B. Organic-trained probes transferred to induced errors more readily than induced-trained probes transferred to organic errors; low organic hallucination recall was a recurring problem. The purpose of this bundle is to compare the baseline with instruction controls and limited organic supervision on two new subject models.

The subject transformer stays frozen. Only small linear classifiers are trained on its saved internal activations. This is not language-model fine-tuning.

This bundle is the **whole-response, mean-pooled activation experiment**. The separate GPT-OSS `localized_probe/` study concerns token/span probing and is not part of this bundle. Do not import its token labels, span-max loss, or span-selection rules into this experiment.

## 2. What is actually present and verified

The bundle contains code, configuration, fixed question pools and splits, HalluHard source files, cloud/checkpoint utilities, and output directories. At handoff creation:

- `scripts/verify_bundle.py` passed: dataset counts, split coverage/disjointness, HalluHard domains, and output-directory write access were checked.
- `sha256sum --check --quiet BUNDLE_MANIFEST.sha256` passed for the existing snapshot.
- `runs/` and `results/` contained no experiment artifacts beyond directory placeholders. There are no completed DeepSeek/GLM findings to quote from this portable copy.
- No subject-model load, GPU benchmark, remote model-availability check, paid judging run, or external alert delivery was performed for this handoff.

The configured model IDs and revisions below are values read from the bundle, not independent confirmation that the destination machine can download or execute them. Run the supplied preflight on that machine before a full run. Do not silently replace a checkpoint to get a run working.

## 3. Authoritative files

Read these before editing the scientific setup:

| File | Role |
|---|---|
| [configs/cloud_experiment.yaml](configs/cloud_experiment.yaml) | Active models, four arms, three probes, mixing settings, seeds, checkpoint and monitoring settings |
| [configs/experiment.yaml](configs/experiment.yaml) | Inherited datasets, judging, generation, representation, splits, hyperparameter grids |
| [cross_probe/prompts.py](cross_probe/prompts.py) | Exact instruction wording, frames, truthful variants, template assignment |
| [scripts/02b_select_known.py](scripts/02b_select_known.py) | Organic correctness requirement for induction eligibility |
| [cross_probe/transfer_experiments.py](cross_probe/transfer_experiments.py) | Basic/crossed source views, weights, preprocessing, linear probe fitting |
| [cross_probe/pooled_transfer.py](cross_probe/pooled_transfer.py) | Dataset pooling, duplicate-question checks, organic sampling, macro validation metrics |
| [cross_probe/cloud/probes.py](cross_probe/cloud/probes.py) | Current four-arm experiment and source-first layer selection |
| [cross_probe/cloud/report.py](cross_probe/cloud/report.py) | Report aggregation and figures |
| [cross_probe/cloud/pipeline.py](cross_probe/cloud/pipeline.py) | Stage order, resumability, progress checks |
| [cross_probe/cloud/checkpoint.py](cross_probe/cloud/checkpoint.py) | Checkpoint contents, upload and restore |
| [cross_probe/cloud/monitoring.py](cross_probe/cloud/monitoring.py) | W&B runs, heartbeats, alerts |

Cloud configuration overrides inherited configuration. For example, the base config mentions Ridge, but `cloud.probes` activates only difference of means, logistic regression, and LDA. Generic helper constants may list older mix grids; the active cloud run uses the single mixture specified below.

Some copied modules/docstrings retain Qwen, GPT-OSS, or `cloud_v3` names. Use the active configuration and cloud entry points to determine behavior, not those historical names.

## 4. Subject models and judge

| Setting | DeepSeek | GLM |
|---|---|---|
| Runner model key | `deepseek_v41_flash` | `glm_53_flash` |
| Configured checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash` | `zai-org/GLM-5.3-Flash` |
| Pinned revision | `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` |
| Loader | `causal_lm` | `multimodal_lm`, with processor |
| Quantization | Native FP8 | Native FP8 |
| Configured dtype / placement | `bfloat16` / `device_map: auto` | `bfloat16` / `device_map: auto` |
| Expected text blocks / hidden width | 40 / 5120 | 45 / 4096 |
| Minimum Transformers in model config | 5.6.0 | 5.16.0 |
| Chat-template options | `thinking_mode: thinking`, `reasoning_effort: 1` | `reasoning_effort: low`, `clear_thinking: true` |

`requirements.txt` pins Transformers to 5.16.0, PyTorch to 2.8.0+cu128, and torchvision to 0.23.0+cu128. Other dependencies use version ranges, so record the actual installed environment as well as the container digest. Setup requires Python 3.11 or newer.

Both subject models generate answers and expose hidden states **on the GPU worker**. OpenRouter is used for judging, not as a replacement for subject-model activation extraction.

The configured judge is `~openai/gpt-mini-latest` through OpenRouter, temperature 0, minimal reasoning effort, structured output, and eight workers. Preserve the returned model identity and raw judge responses; a `latest` alias is not an immutable judge version.

Short-answer judging first uses the existing deterministic alias rules, then the LLM where needed. A fixed 10% audit of rule matches measures agreement without changing those verdicts. HalluHard uses its existing domain-aware reference/knowledge/coding judging paths. Preserve these conventions rather than sending every row through a newly written judge.

## 5. Data and fixed splits

The active canonical datasets are PopQA, TriviaQA, and HalluHard. Probes fit these datasets **together**, separately for each model, induction register, and arm.

Splits are 60% train / 20% validation / 20% test, seed 42, at the **question-ID level**. Every generated response associated with one question inherits its split.

| Dataset key | Question pool | Train | Validation | Test |
|---|---:|---:|---:|---:|
| `pop_qa` | 9,000 | 5,400 | 1,800 | 1,800 |
| `trivia_qa` | 5,000 | 3,000 | 1,000 | 1,000 |
| `hallu_hard` | 700 | 420 | 140 | 140 |
| **Total** | **14,700** | **8,820** | **2,940** | **2,940** |

PopQA covers factual entity/relation knowledge and was sampled across popularity bins. TriviaQA supplies general factual/trivia questions. The permanent HalluHard pool is:

| HalluHard domain | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Medical guidelines | 250 | 150 | 50 | 50 |
| Research questions | 250 | 150 | 50 | 50 |
| Legal cases | 100 | 60 | 20 | 20 |
| Coding | 100 | 60 | 20 | 20 |

The exact copied inputs are `data/<dataset>/questions.jsonl` and `data/<dataset>/split_seed42.json`. HalluHard source JSONL files are under `data/hallu_hard/source/`. The original upstream dataset's split name is not the experiment split.

Keep these files unchanged for comparable runs. The preparation script exists for rebuilding inputs; an ordinary run does not require resampling or redownloading the fixed question pools.

Identical normalized question text appearing in different datasets across different splits is excluded as complete question groups during pooled loading. Response duplicates, invalid alignment, truncation/exclusions, and unusable judge verdicts can further reduce usable rows. The table gives the fixed question pool, not final classifier sample counts.

## 6. Organic generation, eligibility, and labels

For each subject model independently:

1. Generate organic answers for all questions, using the saved organic prompt convention.
2. Judge those answers and retain raw verdicts.
3. Select induction-eligible questions from organic judgments only.
4. Generate truthful-instruction and induced-error answers only for those eligible questions.
5. Judge the new answers; derive labels from actual correctness, not the instruction.

Labels are always:

```text
0 = judged correct / truth
1 = judged incorrect / hallucination
```

An incorrect answer under a truthful instruction is still label 1. A correct answer under a lie/misremember/unknown instruction is still label 0. `condition == "incorrect"` identifies an error-inducing prompt, not proof that the response was incorrect. `abstain` and unresolved `unknown` verdicts are not binary training examples.

The active eligibility criterion is `correct_never_wrong`. Precisely, a question must have at least one organic `correct` verdict and no organic `incorrect` verdict. All planned organic answers must have resolved judgments, and invalid/truncated organic generations disqualify the question. With two organic answers, **correct + abstain is eligible under the current predicate**; two abstentions are not. The older README's shorthand "always correct" is stricter than the actual predicate.

There are two organic generations per question: nominally 29,400 organic responses per model before exclusions, including 5,880 responses attached to test questions. Actual truth/hallucination counts depend on the new model's outputs and cannot be borrowed from Qwen or GPT-OSS.

Organic evaluation retains usable organic test answers across the full test pool, including questions that produced natural errors. It is not restricted to induction-eligible questions. The organic training sample used for mixing likewise includes naturally correct and incorrect answers, not just the induction-eligible population.

## 7. Exact prompts and generation conventions

The three error registers are `lie`, `misremember`, and `unknown`. Preserve the strings in `cross_probe/prompts.py`.

- `lie`: deliberately supply a plausible incorrect answer without announcing the deception.
- `misremember`: answer from a supposedly faulty recollection, confidently and incorrectly.
- `unknown`: act as though the fact was never learned, but still confidently supply a plausible incorrect answer. This is not an abstention condition.

The current prompts explicitly induce incorrectness. Do not describe this snapshot as having already switched to subtle, intent-free confusion prompts.

Organic and lie prompts use the standard frame. Misremember and unknown use the strong task frame. The truthful control is matched to the frame: `TRUTHFUL_FOR = {lie: v1, misremember: strict, unknown: strict}`. Short-answer and long-form datasets have corresponding prompt styles.

Each instruction family has five paraphrases. A question gets a deterministic seen paraphrase from IDs 0–3; test questions additionally receive held-out paraphrase 4. Preserve that assignment. The default source-test evaluation includes the available seen and held-out instructed rows.

Current generation defaults are temperature 0.7, top-p 0.8, top-k 20, sampling enabled, and base seed 12345. The cloud models override batch size to 1, with a 512-token generation budget for short answers and 768 for HalluHard; do not mistake these for the smaller inherited dataset defaults.

Truthful instructions receive two samples per question per applicable template/variant. The cloud generation stage computes induced-error samples adaptively per dataset/register:

```text
n_error = clamp(round(target_rows / number_of_eligible_questions), 2, 10)
target_rows = 8000 for PopQA/TriviaQA; 2000 for HalluHard
```

Test-template duplication and actual instruction success affect final counts. The inherited `inorganic_samples_per_question: 4` is not the cloud stage's final adaptive count.

## 8. Four experimental arms: resolve the naming ambiguity

The user's intended comparison is baseline transfer, instruction control, limited organic supervision, and the combination. The following is the **current portable implementation**, defined by `cloud/probes.py:ARMS` and `transfer_experiments.py:source_view`.

| Arm key | Induced source rows | Organic training data |
|---|---|---|
| `plain_inorganic` | Truthful instruction + correct; chosen error instruction + incorrect | None |
| `same_instruction` | Both judged outcomes under both the matched truthful instruction and the chosen error instruction | None |
| `plain_inorganic_plus_organic` | Same source view as `plain_inorganic` | 20% organic training-question sample; 50% total training weight |
| `same_instruction_plus_organic` | Same source view as `same_instruction` | Same organic sampling convention and weight |

**Important:** in this bundle, `same_instruction` maps to source variant `crossed`, which contains these four cells:

| Instruction | Judged truth (0) | Judged hallucination (1) |
|---|---|---|
| Matched truthful instruction | Successful factual answer | Failed factual instruction |
| Chosen lie/misremember/unknown instruction | Failed error induction | Successful error induction |

Thus both labels can occur within an instruction family. It does **not** mean every source row receives one identical instruction, or that training uses only successes/failures under the inducing prompt. Older experiments used similar names differently; inspect the source rows before equating their results.

The baseline takes only the upper-left and lower-right cells. The crossed view keeps all four and balances their weight within each dataset. Empty required cells cause fitting to fail; do not silently substitute the baseline or merge registers to make it pass.

Train lie, misremember, and unknown probes separately. Only the datasets are pooled; the three registers are not pooled into one source classifier. Here, "crossed" names the four-cell source design; the combined arm adds organic data to that design.

The two source-only arms measure transfer without organic training labels. The mixed arms measure transfer with limited target-domain supervision and must be described that way.

## 9. What "20% organic, 50–50 weight" means

Active cloud settings are:

```yaml
organic_fraction: 0.20
organic_mass: 0.50
seeds: [42, 43, 44, 45, 46]
mixed_layer_policy: inherit_source
mixed_hyperparameter_policy: inherit_source
```

`organic_question_sample` starts from usable **organic train-split rows**. Within each dataset, questions are stratified by whether any retained organic answer is incorrect (`max(label)` per question). It samples about 20% of questions in each stratum, with rounding and a minimum of one where available, then includes all retained organic responses for those sampled questions.

All eligible induced training rows for the selected source design remain in training. Consequently:

- 20% is a fraction of the organic training-question pool, not the organic share of the concatenated row count.
- It is not 20% of all organic data, and never includes organic test questions.
- It is not a schedule of 80% inorganic epochs followed by 20% organic epochs. Each mixed linear probe is fitted on concatenated induced and sampled organic activations.
- The induced rows jointly receive 50% of total training weight and organic rows jointly receive 50%, regardless of row counts.

Within a domain, datasets receive equal weight. Induced classes/cells and organic classes are balanced, with equal question weight within each cell/class and that question's weight shared across its responses. Logistic/LDA preprocessing is fitted on the actual weighted training matrix, including the sampled organic rows for mixed arms. Validation/test rows do not fit preprocessing.

Five seeds repeat the organic question draw. The report averages those trials. There is one configured mixture, not a search over fractions/weights, and the report does not choose the best organic-test seed. The same seed produces the same organic sample across registers/probes/arms when the retained organic pool is unchanged.

For exact counts, read `training_sample_counts.csv` and `sample_counts_by_dataset_and_label.csv`. Distinguish unique questions from generated-answer rows, and sampled-data fraction from loss weight.

## 10. Activation representation, fitting, and selection

Replay the exact saved prompt and generated tokens through the same pinned subject model. The current representation is hidden states at generated **answer-token positions**, mean-pooled across that answer, at every available layer. Honor the existing answer-boundary extraction; do not pool prompt/template tokens or replace the answer with the reasoning trace.

This produces one vector per retained response per layer. Replay uses compact, resumable chunks of 32 response rows. Saved pooled activations are float32. The cloud representation includes embedding output at index 0 and text-block outputs at subsequent indices: expected counts are 41 for DeepSeek and 46 for GLM. Report this indexing convention with selected layers.

The active probes are:

| Probe key | Fitting convention |
|---|---|
| `dim` | Weighted hallucination mean minus weighted truth mean; no standardization by default |
| `logistic` | Weighted L2 logistic regression, weighted train-only centering/scaling, up to 5000 iterations |
| `lda` | Weighted LDA with train-only centering/scaling and numeric shrinkage search |

Logistic C grid: `[0.0001, 0.001, 0.01, 0.1, 1, 10]`. LDA shrinkage grid: `[0.1, 0.3, 0.5, 0.7, 0.9]`. Saved probe scores are linear decision scores; do not call all of them calibrated probabilities.

For each model × probe × register:

1. Fit `plain_inorganic` candidates across available layers and hyperparameters, using its source training rows.
2. Select using equal-dataset macro AUROC on that source validation set. Layer ties choose the shallower layer.
3. Independently do the same full search for `same_instruction` using its own source view.
4. Fit each mixed arm at the **layer and hyperparameter inherited from its corresponding source-only arm**. Coefficients and training preprocessing are newly fitted for each organic sampling seed.
5. For every fitted model, select a decision threshold maximizing equal-dataset balanced accuracy on its corresponding **source validation** rows.
6. Freeze fitting/selection choices and evaluate on source test and organic test.

This implements the user's later request to follow the faster Qwen/GPT-OSS source-layer-first workflow. Do not restore a complete independent mixed-arm layer search as a routine refactor.

Organic validation AUROC is calculated for mixed arms as a diagnostic. With the current inherited policy it does not select layer or hyperparameter. A mixed result's `selection_auroc` field contains that organic-validation diagnostic; inspect `selection_distribution`, `layer_selection_mode`, and `organic_validation_auroc` before describing it as source-selection performance.

Never use organic test labels to choose preprocessing, hyperparameters, layers, thresholds, mixture, or a winning seed. Thresholds use balanced accuracy, not a fixed 5% FPR operating point.

## 11. Evaluation and report interpretation

Positive class is organic hallucination. Record AUROC, AUPRC, hallucination precision/recall, F1, FPR, FNR, balanced accuracy, confusion counts, and truth-class precision/recall. In particular, `FNR = 1 - hallucination recall` and `balanced accuracy = (truth recall + hallucination recall) / 2`.

The primary metric table uses **equal-dataset macro metrics**: calculate each dataset's metric, then average the three values. This prevents PopQA's larger pool dominating the headline. It is different from computing one metric on all concatenated test rows.

Plots are organized by probe and compare registers/arms; the user does not want a separate primary panel for every dataset. Retain per-dataset numbers in machine-readable results for auditing and sample-count questions.

The report saves AUROC confidence intervals using 1000 question-cluster bootstrap resamples within datasets, plus paired AUROC differences against `plain_inorganic`. Mixed summary rows average the five seeds, including the reported interval endpoints; those averaged endpoints are not a new bootstrap over seed variability.

Two plotting details matter when comparing charts with tables:

- Confusion figures sum counts across datasets and average counts across mixed seeds, then show row-normalized percentages. Their implied pooled precision/recall need not equal the macro table. Decimal confusion counts arise from seed averaging.
- Score histograms combine datasets, use normalized sample density, and use mean per-response scores across mixed seeds. They show the mean seed threshold. Metrics in the table instead average per-seed evaluations; a histogram is not a separately evaluated ensemble.

Histogram colors are inorganic truth **blue**, inorganic hallucination **red**, organic truth **green**, and organic hallucination **orange**. The x-axis is probe score. Current plots label the y-axis "Normalized sample density". Mixed layer figures mark the inherited layer; they are not full mixed-layer searches.

## 12. Pipeline and saved artifacts

The full runner executes:

```text
prepare -> organic_generation -> organic_judging -> eligibility
        -> inorganic_generation -> inorganic_judging
        -> activation_replay -> dataset_build -> probe_fit -> report
```

Generation/judging reuse completed conversation records. Replay resumes from completed shards. Probe fitting caches completed layers/trials. Preserve these artifacts when resuming; do not regenerate already saved answers or replace existing judgments merely to rerun plotting.

Paths use `experiment_namespace: deepseek_glm_cloud_v1`:

```text
runs/<model>/<dataset>/<experiment_id>/
  generations.jsonl
  known_questions.json
  judge/verdicts.jsonl, raw.jsonl, stats.json
  replay/chunks/, index.jsonl, activations.npy
  dataset/rows.parquet, acts_layer_major.npy

runs/deepseek_glm_cloud_v1/<experiment_id>/<model>/
  stages/<stage>.json
  monitoring/<stage>/status.json

runs/summary/deepseek_glm_cloud_v1/<experiment_id>/<model>/
  results.json
  models/<probe>/<register>/<arm>/<fixed_or_seed_N>/
    grid.json, selection.json, result.json, scores.npz
    selected_probe.npz, layer_*.npz

results/deepseek_glm_cloud_v1/<experiment_id>/<model>/
  all_trials.csv
  equal_dataset_summary.csv
  training_sample_counts.csv
  sample_counts_by_dataset_and_label.csv
  paired_organic_auroc_deltas.csv
  report_manifest.json
  <probe>/
    metrics.csv
    organic_transfer_metric_matrices.png
    organic_test_confusion_matrices.png
    layer_selection_curves.png
    score_histograms/<register>_activation_score_distributions.png
```

The metric figure contains AUROC, AUPRC, precision, recall, FPR, FNR, and balanced-accuracy matrices. F1 and classwise metrics remain available in tables/JSON. This bundle's report does not currently generate a PDF; do not promise the separate localized experiment's PDF as an output here.

## 13. Setup and launch on another GPU machine

Copy the whole folder, including hidden template files. Keep existing results in their current namespaces; choose a fresh experiment ID for a changed scientific configuration. Keep the same ID for an exact resume.

Hardware must load the full pinned checkpoint and pass generation/replay preflight. Native FP8 support, CUDA compatibility, system RAM, and local activation/cache disk capacity matter. The bundle targets a large multi-GPU worker, not a verified single-small-GPU deployment. `device_map: auto` does not guarantee that a checkpoint fits. Run one subject model per worker, or run them sequentially on shared GPUs.

From the copied directory:

```bash
cd deepseek_glm_experiment
bash scripts/setup.sh
# Setup creates .env from .env.example only when .env is absent.
# Fill .env with the destination credentials and storage settings.
.venv/bin/python scripts/verify_bundle.py
sha256sum --check --quiet BUNDLE_MANIFEST.sha256
```

Important environment settings:

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | GPT-mini judging |
| `HF_TOKEN` | Hugging Face checkpoint access where required |
| `CROSS_PROBE_ARTIFACT_URI` | Set to `gs://YOUR_BUCKET/PREFIX` for remote checkpoints |
| `CROSS_PROBE_CONTAINER_DIGEST` | Record the actual immutable container digest for provenance |
| `WANDB_API_KEY`, `WANDB_ENTITY` | W&B tracking account/project ownership |
| `WANDB_MODE` | `online` for remote tracking; `offline` is a local diagnostic fallback |
| `GOOGLE_CLOUD_PROJECT` | Cloud Monitoring project |
| `CROSS_PROBE_ALERT_EMAIL` | Alert recipient configuration |
| `GOOGLE_APPLICATION_CREDENTIALS` | Optional service-account file path if not using attached credentials/ADC |

The `.env.example` defaults `file://cloud_artifacts` and `local-unpinned` are placeholders, not proof of remote recovery or immutable provenance. Configure the bucket and real environment identity for the requested cloud experiment. Credentials are not included in the bundle.

For local structural inspection without model loading or judging:

```bash
.venv/bin/python scripts/30_cloud_experiment.py validate \
  --config configs/cloud_experiment.yaml \
  --experiment-id ds-glm-v1 --model deepseek_v41_flash \
  --artifact-uri file://cloud_artifacts --offline
```

This validation writes a small local preflight/storage check. It does not establish GPU/model compatibility. Do not pass `--offline` to scientific run/stage commands.

Before a full model run:

```bash
bash scripts/preflight_model.sh deepseek_v41_flash ds-glm-v1
bash scripts/preflight_model.sh glm_53_flash ds-glm-v1
```

Online preflight checks credentials/storage, loads the selected model, exercises generation and activation replay, and tests monitoring/alert channels. It is not a cheap config-only check. Its credit check requires a minimum balance, not enough credit for the entire experiment. No runtime or total cost has been measured for this portable copy.

Run both sequentially on one suitable worker:

```bash
bash scripts/run_both_sequential.sh ds-glm-v1
```

Or run the appropriate command on each dedicated worker:

```bash
bash scripts/run_model.sh deepseek_v41_flash ds-glm-v1
bash scripts/run_model.sh glm_53_flash ds-glm-v1
```

These foreground runners should live inside the destination's persistent job runner or terminal session. Status commands are:

```bash
bash scripts/status.sh deepseek_v41_flash ds-glm-v1
bash scripts/status.sh glm_53_flash ds-glm-v1
```

If only figures need rebuilding after completed fits:

```bash
.venv/bin/python scripts/32_cloud_probes.py \
  --config configs/cloud_experiment.yaml \
  --experiment-id ds-glm-v1 --model deepseek_v41_flash \
  --phase report --no-gate
```

Use `glm_53_flash` for the other model. This rewrites that run's derived report files from saved fits; it does not rerun generation or judging. Use a separate output copy if preserving a particular report snapshot is required.

## 14. Hourly checkpoints and recovery

The pipeline starts checkpoint uploads every 3600 seconds while a stage runs, and checkpoints at stage completion and on handled failures/signals. Uploads use content-addressed objects, SHA-256 manifests, and a `LATEST.json` pointer published after the checkpoint manifest.

Remote keys are under the configured artifact prefix, with run namespace:

```text
cross-probe/deepseek_glm_cloud_v1/<experiment_id>/<model>/<stage>/
```

Checkpoint roots include the run control state, raw dataset runs, probe summaries/models, and reports. Immutable replay shards are retained. Large reproducible materializations such as `replay/activations.npy`, `dataset/acts_layer_major.npy`, and the source activation cache are excluded from upload and rebuilt from shards when needed after restore.

Model-weight caches and the portable source/input bundle are not uploaded as part of those run roots. Keep the bundle and pinned environment available separately. An hourly checkpoint captures completed persisted work, not an in-flight GPU forward pass or optimizer iteration. On abrupt failure, work since the last successful remote checkpoint may need repeating; uploads also take time.

For a local restart with artifacts still intact, rerun `run_model.sh` with the same ID. On a replacement worker, configure the same artifact URI, model, experiment ID, and compatible original configuration/environment, then:

```bash
bash scripts/resume_model.sh deepseek_v41_flash ds-glm-v1
# Or:
bash scripts/resume_model.sh glm_53_flash ds-glm-v1
```

The resume wrapper restores the latest available stage checkpoint and invokes the normal runner. Restored files are checksum-verified. Check model/config/container provenance before continuing; do not treat a changed configuration as an exact resume.

## 15. W&B and alerts

W&B project: `cross-probe-deepseek-glm`. Runs are grouped by experiment ID and model, with a resumable run ID per stage. Tracking covers stage progress, system/checkpoint information, and final reports.

Configured timings are:

- Heartbeat every 60 seconds.
- In-process stall alert after 900 seconds without detected experiment-file progress.
- Stale-checkpoint alert at 5400 seconds since a verified checkpoint.
- Alert on stage exceptions and repeated checkpoint failures; GCP policies also cover high disk utilization.

Progress-based stall detection is a heuristic: an unusually long operation that has not written a progress artifact can look stalled. Inspect logs and GPU activity before interpreting the alert as a failed scientific run.

An in-process monitor cannot notify anyone after its whole VM disappears. The external missing-heartbeat policy is therefore part of the required resilience setup. Merely setting an email address does not create GCP notification channels or policies.

The included [infra/gcp/monitoring.tf](infra/gcp/monitoring.tf) defines metrics, an email notification channel, and policies for structured process alerts, missing heartbeats, stale checkpoints, stalled stages, and disk usage. Deploy it to the chosen project and configure stdout collection into Cloud Logging as described in [infra/gcp/README.md](infra/gcp/README.md):

```bash
terraform -chdir=infra/gcp init
terraform -chdir=infra/gcp apply \
  -var='project_id=YOUR_PROJECT' \
  -var='alert_email=YOUR_EMAIL'
```

The worker needs Cloud Storage object access and Monitoring metric-write permissions. Alert infrastructure also needs its deployment permissions and a logging agent. Verify actual alert delivery with the monitoring preflight. W&B/GCP logging errors can be reported without terminating scientific work, so a running process alone does not prove remote tracking is healthy.

## 16. Guidance for the receiving Codex session

Use this as context for the user's next request. Inspect the current destination's stage markers and saved artifacts before reporting progress; this handoff's empty-run status is only a dated snapshot.

Preserve existing answers, labels, question splits, exact prompts, pinned checkpoints, and prior outputs. Use new namespaces for changed experiments. Make scientific changes explicit rather than silently changing an arm's source cells, organic budget, weighting, layer policy, or threshold convention.

When discussing results, identify the subject model, probe, register, arm, source of selection, and whether numbers are per seed or averaged. Distinguish pure induced-to-organic transfer from transfer after organic training supervision. Prefer headline comparisons across arms over dataset-specific narratives, while keeping enough counts and per-dataset data to audit the result.

The key questions are whether instruction control helps, whether limited organic supervision helps, and whether their combination improves AUROC and hallucination recall without an unacceptable false-positive rate. Report the measured tradeoff; do not describe zero positive predictions as successful detection merely because overall accuracy looks high.
