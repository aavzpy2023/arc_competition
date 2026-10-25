## Context

- Business context: [ARC Prize 2026 - ARC-AGI-2, overview](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview)
- Data context: [ARC Prize 2026 - ARC-AGI-2, data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data)

## Summary

A 7B coder model fine-tuned to write Python programs for ARC-AGI-2 scored **0.0** on the leaderboard and **0.0%** on the 120-task public evaluation set: only 1 of 7,360 sampled programs reproduced all training pairs. This writeup measures why. The fine-tune buys format, not reasoning; two training failures looked healthy on paper; and five interventions (more samples, temperature, repair, rotated and recoloured views, extra synthetic data) moved proxy metrics by a few points and solves not at all. Everything ran on Kaggle's L4x4 machines (four NVIDIA L4 GPUs, 24 GB each) within the weekly 30 GPU-hour quota. The full report (PDF) is attached.

## Overview of the approach

**Data.** About 2,000 verified generator programs. Phase 1: 1,035 synthetic tasks written from a hand-authored specification of 57 rule categories (see ARC_AGI2_Problem_Types.pdf) and more than 30 combinations (see ARC_AGI2_Problem_Combinations.pdf). Phase 2: 962 public ARC training tasks solved by an LLM-in-the-loop solver (a frontier model writes the program, a validator runs it and returns tracebacks; 992 of ~995 solved). Every program follows one contract: `transform(grid)` with an analytical docstring, `generate_input(rng)` with bounded loops, a `METADATA` line and a `_self_verify()` (see an example in TASK_ID 3B7C2F9A.py). Each is re-run on the original pairs and under 2 random colour permutations before entering the corpus, and each generator is tested over 30 seeds.

**Prompt format: a dual-channel grid DSL** (domain-specific language). Each grid is shown as (1) a symbolic channel: background, 8-connected objects with shape, box, centroid and size, alignment/containment relations, and the input-to-output D4 relation; and (2) a pixel channel with runs of 3+ written as `C*N`. On 2,100 grids it is 82.2% shorter than JSON and 47.4% shorter than compact digits, never longer. It is not self-delimiting (a greedy decoder mis-parses 48% of rows), but given the stated width every one of 24,728 rows has exactly one parse.

```text
[MATRIX IN]            row "7*114*37*11" = 7x11, 4x3, 7x11 (width 25)
7*25
7*114*37*11
7*114747*11
```

**Training.** Qwen2.5-Coder-7B-Instruct, QLoRA (4-bit NF4, LoRA r = 32, α = 64), DDP on 4 L4s, effective batch 32, max length 4,096, ~60 s/step. The submitted model (v3): 160 steps, NLL on the answer only, at most 6 variants per program, validation on 28 programs never trained on.

```python
args = SFTConfig(loss_type="nll", completion_only_loss=True, max_length=4096,
                 gradient_accumulation_steps=8, learning_rate=1.5e-4,
                 lr_scheduler_type="cosine", optim="adamw_8bit", eval_steps=60)
```

**Inference (offline, ~60 min for 240 tasks).** Merged bf16 weights on vLLM, one process per GPU, 64 programs per task (16 at T = 0.3, 32 at 0.7, 16 at 0.9). Each distinct program runs in a sandbox: a 3 s alarm per call raised as a `BaseException` (generated `try/except` cannot swallow it), a 6 GB memory cap and a 40 s hard kill. Train-passing programs vote with their test outputs for the two attempts; empty slots fall back to the best cell accuracy, then the test input. A valid `submission.json` is written before any model loads.

## Results

Metrics on each task's training pairs: shape OK (all outputs have the right shape), cell accuracy (0 when the shape is wrong), train-passing (all exact). Eval-30 = the 30 lowest-complexity public evaluation tasks; eval-120 = all.

| Experiment (eval-30 unless noted) | Shape OK | Cell acc. | Train-passing | Score |
| --- | --- | --- | --- | --- |
| Base model, no LoRA (48 samples) | 2.6% | 0.020 | 0 (50% truncated) | 0% |
| v3 LoRA, 64 samples, mixed T | 60.5% | 0.471 | 2 of 1,920 (1 task) | 3.3% once, 0% in reruns |
| v3, 64 samples at T = 0.3 | 66.1% | 0.514 | 0 | 0% |
| v3, 8 D4 views × 8 | 65.5% | 0.513 | 2 (wrong on test) | 0% |
| v3, 8 D4 + colour views × 8 | 66.2% | 0.516 | 0 | 0% |
| v3 repair loop, 1,200 repairs | n/a | 27 better, 189 worse | 0 | 0% |
| **v3, eval-120** | 60.3% | 0.476 | 1 of 7,360 | 0% |
| **v4 (+synthetic rows), eval-120** | 61.2% | 0.489 | 0 of 7,360 | 0% |
| **Kaggle leaderboard (v3)** | | | | **0.0** |

Oracle pass@k grows linearly with k (0.07% at k = 1 to 4.44% at k = 64), all from one task. Lower temperature improves every proxy, yet the only solve came from the hot tail.

**Sanity check on seen tasks.** The test file visible during the notebook run is a placeholder: all 240 tasks are identical to public training tasks, which the corpus covers. Even there, only 24 to 27 of 231 tasks got a train-passing program (three runs, v3 and v4). Recall of trained rules is weak; on new tasks it is 1 in 7,360.

## Two training failures the metrics hid

1. **Prose instead of code.** The first adapter was trained with the full reasoning docstring in the answer. It solved 0 of 120 evaluation tasks, and 583 attempts contained no code at all, only object descriptions or raw matrices until the token limit. Fix: the answer is the verified code alone.
2. **Mode collapse at 94% token accuracy.** 180 attempts produced 40 distinct programs; one BFS template filled 44 attempts across 8 unrelated tasks, with identical lengths even at T = 0.85. Causes: a dynamic fine-tuning (DFT) loss active although NLL was intended (entropy fell to 0.0085), loss over the whole sequence (83% prompt), and ~18 variants per program with a random row-level validation split. Fix: NLL on the answer only, ≤6 variants, validation by distinct program. Entropy recovered to ~0.40.

## Why nothing moved the score

The bottleneck is **idea coverage**: on unseen tasks the model almost never proposes the right rule, and every intervention multiplies that near-zero rate instead of raising it.

- **Sampling multiplies zero.** With per-sample success p, a hit in k samples has probability 1 − (1 − p)^k; for 29 of 30 easy tasks p ≈ 0.
- **Views re-express, they add no ideas.** A model that cannot infer a rule in one orientation cannot in eight; per-view shape accuracy stayed at 62–70%.
- **Repair needs training to repair.** The model never saw (failed program, feedback, fix) triples; 45% of repairs returned the code unchanged. SOAR's key step is exactly that training.
- **More data of the same kind is memorised.** With under a thousand distinct real programs, held-out loss bottoms out at step 60–120 (v3: 0.2354; v4: 0.2435) and rises after.
- **The teacher solves what the student cannot.** The frontier model solved 992 of ~995 training tasks with execution feedback and up to 18 attempts. Distilling its finished programs transferred format and idioms, not the loop of hypothesis, test and revision.

The ablation separates two skills: writing a terminating function that returns a well-formed grid is learned in under 200 steps (shape OK 2.6% → ~60%); inferring a new rule from a few examples is not learned at this data scale. Every top Kaggle entry closes that gap with per-task adaptation at test time.

## Engineering lessons

Most lost GPU time came from engineering. Two runs used the wrong loss, and an evaluator without time limits hung for 9+ hours. Of 21 evaluation notebook versions before the final pipeline, 13 failed and 7 timed out, all on infrastructure. The first real submission lost 85% of its tasks because hidden-test prompts reach 40,765 tokens. Checks that would have saved most of it:

- Print the active loss type and completion mask before step one; track distinct programs per task, not only loss.
- Split validation by distinct program, never by row.
- Time-limit every call into generated code with a `BaseException`, plus a hard kill and a global budget.
- Run the base model on your prompts in week one; measure on the full public set.
- Write a fallback `submission.json` first; let no single task kill a process.

## Limitations and next steps

One model size, about 2,000 distinct programs, single runs, no per-task identity baseline, no diagnostics on the hidden test set beyond its score. Three extensions reuse the pipeline as is: test-time training on each task's demonstrations and its safe D4/colour views; training on the repair loop's successful traces, as SOAR does; and thousands of new distinct programs from the existing generators.

## Sources

- [ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems](https://arxiv.org/pdf/2505.11831)
- [ARC Prize 2025 Results and Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis) (NVARC, the ARChitects, MindsAI, TRM, SOAR, CompressARC)
- [ARC Prize 2024 results](https://arcprize.org/2024-results) (induction and transduction, test-time training, latent program search)
- Code (Apache 2.0): [submission notebook](https://www.kaggle.com/code/andreyvinajerazamora/lora-for-arc-prize-2026-arc-agi-2), [data-pipeline notebook](https://www.kaggle.com/code/andreyvinajerazamora/arc2-data-pipeline), [data-pipeline dataset](https://www.kaggle.com/datasets/andreyvinajerazamora/arc2-datasets-pipeline)
