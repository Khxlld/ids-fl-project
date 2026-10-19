# Project instructions for Codex

## Purpose and current scope

This repository develops a Federated Learning-Based Intrusion Detection System for IoT networks. UAV networks are a primary simulated use case. The IDS classifies traffic and raises alerts; it does not automatically mitigate attacks.

The current research task is Contribution 1: federated CLOSR for known-class classification and unknown-class rejection under heterogeneous Non-IID data. Start with a trustworthy centralized CLOSR baseline and a FedAvg CLOSR baseline before modifying the method. Knowledge distillation and unknown-to-known adaptation are later contributions. Exclude the previously suggested server-side generator.

## Read and maintain project context

- On starting work on this research task, read `research/PROJECT_CONTEXT.md` and `research/STATUS.md`. Inspect the relevant local code and outputs before changing them.
- Treat those files as the handover from the research conversation. They are not proof of the current local checkout or installed environment; verify local facts.
- Record material decisions, completed changes, exact checks and outcomes, unresolved issues, and the next step in `research/STATUS.md` after each completed task.
- User instructions in the active session take precedence. If the mentor's direction changes, update the context rather than retaining contradictory plans.

## Scientific requirements

- Distinguish reported paper findings, observed code/data behavior, assumptions, design decisions, and untested hypotheses.
- Support methodological claims with the exact paper section/equation/table or source file and revision. Never invent results, gaps, dataset fields, or novelty.
- CLOSR uses class-specific CLAD objectives. Do not describe CLAD as a detachable prerequisite binary detector or replace the method with generic contrastive learning without documenting the change.
- Freeze the unknown attack categories before an experiment. Exclude their samples from all clients' training, preprocessing fitting, class-center estimation, validation-based model selection, and threshold calibration. Final unknown test samples are for evaluation only.
- Use fresh model initialization for new class-disjoint experiments. Existing binary checkpoints are reference artifacts, not pretrained zero-day baselines.
- Fit preprocessing using the new experiment's training data only. Preserve group/source separation and check duplicate-feature overlap. Never silently substitute a random row split.
- Use known-class validation data for checkpoint selection and operational rejection calibration. Report how the threshold was chosen and what error it controls.
- Evaluate unknown recall, known-class Macro-F1, benign total alert FPR, known-versus-unknown AUROC, and per-client/worst-client results when valid held-out client data exists. Label in-sample diagnostics explicitly.
- A missing class on one client is not globally unknown if another training client has it. Explicitly separate local absence from global novelty.
- Synthetic checks establish implementation behavior only. They do not establish UAV performance or a novel research contribution.

## Implementation agreements

- Preserve the working binary IDS, Docker/security pipeline, dashboard, datasets, and existing locked evaluation artifacts. Put new research work in a separate experiment directory/branch after inspecting local Git status.
- Use CPU-compatible Python/PyTorch. Python 3.11 is preferred if compatible with the verified environment; CUDA is optional. Keep configuration, seeds, label mappings, and feature lists explicit.
- Specify how absent-class heads, valid positive/negative pairs, aggregation weights, and class centers are handled. Compute centers in the representation of the model that actually performs inference.
- Keep client-local computations separate from server aggregation. Disclose any centrally pooled preprocessing, raw features, embeddings, or validation data. A single-process simulation is not physical data isolation or a privacy guarantee.
- Match the paper and official implementation deliberately. Document deviations in architecture, loss normalization, sampling, or scoring, and retain required upstream attribution/licenses.
- Run meaningful targeted checks, including leakage exclusion, score direction, absent classes, and center/model consistency. Record actual outcomes and unavailable checks.
- Work on the requested phase and routine prerequisites without repeated confirmation. Do not automatically begin a later research contribution.
- Explain the result in simple English: what changed, why, evidence, limitations, and the next step.
