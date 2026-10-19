# Federated CLOSR research handover

Handover date: 2026-10-16. This records the research conversation; inspect the local repository and PDFs before relying on implementation details.

## 1. Project and mentor direction

Project: Federated Learning-Based Intrusion Detection for IoT Networks. The existing demonstration has five simulated UAV clients, a central aggregation server, a binary Normal-versus-Attack model, sample-weighted FedAvg, Docker, post-quantum protection, and a dashboard. It is not a UAV-only research project. Traffic is public simulated research data, not operational military traffic. The IDS raises alerts only.

The user supplied a written summary of the mentor's direction. The original latest meeting record was not supplied in this session, so attribute the following to that summary:

1. **Current priority: Federated unknown-attack detection.** Adapt CLOSR/open-set recognition to classify normal traffic and known attack types and reject globally unseen attack classes across Non-IID clients. Investigate whether local heterogeneity harms known classification or unknown rejection.
2. **Later: Lightweight rejection-preserving CLOSR.** Investigate knowledge distillation that preserves known classification, class relationships, rejection behavior, and useful decision boundaries. Do not assume ordinary compression preserves rejection.
3. **Later: Unknown-to-known federated adaptation.** After investigation yields trustworthy labels, learn a new class locally and propagate it while preserving previous classes and rejection of other unknowns. Address forgetting and class expansion.

The mentor advised dropping the proposed server-side generator. The ordinary FL coordinator remains.

## 2. Sources

- Primary: Jack Wilkie et al., **A Novel Contrastive Loss for Zero-Day Network Intrusion Detection**, IEEE Transactions on Network and Service Management, volume 23, 2026, pp. 2064 onward. DOI: https://doi.org/10.1109/TNSM.2026.3652529
- Official implementation: https://github.com/jackwilkie/CLOSR
- Official implementation inspected in the prior workspace at commit `79da4d1f2a6f4e5ecf92c4588195aaf163df40f8`. Verify the local revision; it may differ.
- Supporting paper: Yuto Kodama et al., **Open-set Recognition with Supervised Contrastive Learning**, MVA 2021. It studies computer vision and EVT scoring. It is not the primary CLOSR NIDS paper; do not import its rejection mechanism into CLOSR.
- User's repository branch: https://github.com/Khxlld/ids-fl-project/tree/Kaust-Final-submission
- User repository inspected at `8e48e981afc001b2d406e659f1409c38aa77e1b4`. Verify the current local revision.

Place the two PDFs locally in `research/references/` using descriptive filenames. Do not automatically publish or commit publisher PDFs; the handover needs their local availability, not redistribution.

## 3. Established methodological understanding

### Architecture and training

- CLAD learns a benign representation using benign anchors, benign positive pairs, and known malicious negatives. Its binary inference compares a sample with the benign center. See Sections III-B-C, Equations 4, 7-10.
- CLOSR extends the objective to each known class through class-specific normalized projection heads over one shared MLP encoder. Every sample is processed by every head. See Figure 1 and Sections III-D-E.
- For normalized vectors, the rescaled cosine distance is `d(u,v) = (1 - dot(u,v))/2`.
- Each class-specific CLAD loss pulls target-class examples together and pushes other known classes away. Equation 11 gives the anchor/pair averages. Equation 12 sums these class-specific losses.
- There is no requirement to run a separate binary CLAD detector before CLOSR. However, the CLAD objective is part of CLOSR, so 'remove CLAD to reduce computation' is not an established valid modification.
- `losses/closr_loss.py` in the official implementation averages across heads rather than summing and uses active-pair normalization details that differ from the literal paper averages. Inspect and document the intended reference before reproducing results. Some differences are scaling; do not assume all are.

### Class centers and inference

For each known class c, the center is the normalized sum of its training embeddings in head c (Equation 13). All centers must correspond to the inference model's current representation.

For an input x:

1. Compute normalized head output `u_c(x)`.
2. Compute cosine similarity `a_c = dot(u_c(x), mu_c)`.
3. Compute weights `p_c = softmax(a)_c`; the candidate known class maximizes `a_c` (Equation 14).
4. Compute unknown score `s(x) = -sum_c p_c * a_c^2` (Equation 16).
5. Predict UNKNOWN when `s(x) > tau`; otherwise use the candidate known class (Equation 17).

Scores closer to zero indicate the assumed unknown geometry; more negative scores indicate strong directional alignment or opposition. This is not simply thresholding the maximum class probability.

The prose surrounding Equations 10 and 17 contains contradictory threshold wording. For CLOSR, Equation 17 and the sign used by `eval_closr.py` agree with the rule above. The code forms a positive weighted-square score and negates it for unknown-positive AUROC evaluation. Verify signs explicitly.

The proposed unknown geometry assumes unseen embeddings tend to be more diffuse and approximately orthogonal to class centers. This is not a guarantee for every unseen attack. Shared input/encoder means separate heads do not by themselves prove statistical independence. These are methodological considerations to investigate, not measured federated failures.

UNKNOWN means unfamiliar traffic, not confirmed malicious intent. Unfamiliar benign behavior can also trigger rejection. Alerts require investigation.

### What the primary experiments establish

- The original study uses centralized Lycos2017 training and withholds SQL Injection and Heartbleed as unknown classes (Section IV-A).
- It reports improved unknown discrimination with some reduction in known-class classification relative to leading closed-set baselines (Section IV-C, Tables III-IV).
- It does not calibrate an operational deployment threshold; it discusses calibration on held-out known data (Section III-D). Calibration alone is therefore not an established novel contribution.
- Table VII reports approximately 3.05 million parameters and 6.07 MFLOPs per CLOSR inference, with timing measured on workstation hardware. This does not establish constrained UAV deployment performance or federated training cost.
- No federated Non-IID training result was established in this session.

## 4. Existing repository information to verify locally

The remote Git root inspected contains a nested `ids-fl-project/` directory. Local checkout layouts may differ. Relevant paths below are relative to that inner code directory:

- `experiments/uavids_federated/prepare_phase2_partitions.py`
- `experiments/uavids_federated/config/phase2_partition_config.json`
- `experiments/uavids_federated/results_phase2/partition_manifest.json`
- `experiments/uavids_federated/results_phase2/PHASE2_PARTITION_SUMMARY.md`
- `experiments/uavids_federated/src/uavids_fl/modeling.py`
- `experiments/uavids_federated/run_phase3_development.py`
- `experiments/uavids_federated/config/phase3_locked_config.json`

The previous model uses `BinaryMLP`, `BCEWithLogitsLoss`, and sample-weighted FedAvg. It is not already a CLOSR implementation.

The saved Phase 2 manifest describes five training-source logical clients and separate validation/test sources. `SrcAddr` is a logical grouping key, not proof of physical UAV identity. Source-disjoint testing does not establish scenario-, session-, temporal-, or physical-UAV-disjoint testing.

The model-facing CSV schema retains 15 numeric features, `original_label`, and `binary_label`. Original labels are Normal Traffic, Blackhole Attack, Flooding Attack, Sybil Attack, and Wormhole Attack. Use the manifest to verify names and schema rather than hard-coding assumptions.

Saved client distributions:

| Client | Normal | Blackhole | Flooding | Sybil | Wormhole |
|---|---:|---:|---:|---:|---:|
| 1 | 597 | 255 | 17 | 23 | 338 |
| 2 | 264 | 367 | 42 | 25 | 348 |
| 3 | 145 | 161 | 3 | 162 | 120 |
| 4 | 201 | 329 | 1429 | 0 | 68 |
| 5 | 619 | 252 | 15 | 0 | 368 |

These are saved report counts, not a new recount of local CSVs. Clients 4 and 5 locally lack Sybil, but Sybil is not globally unknown when other training clients include it.

The partition CSVs were absent from the remote repository copy. Their presence on the user's machine has not been verified. Resolve their actual local paths and validate the manifest hashes before training. Do not ask for another upload if they are already accessible locally.

## 5. Current research questions

1. **Missing local classes:** A class with no anchors contributes no class-specific pair loss in the reference code on that client. Other heads still update the shared encoder. Does this damage the global head or rejection after FedAvg? Untested.
2. **Class-center consistency:** FedAvg averages model tensors, not derived class centers. How will centers be computed with the fixed aggregated model without centrally collecting raw traffic? This needs an explicit baseline design.
3. **Representation changes:** Local centers computed before model aggregation can be stale for the aggregated model. Whether the resulting error is material needs measurement.
4. **Local score distributions:** A single rejection threshold may behave differently across clients. This is a hypothesis, not an approved locally calibrated solution.

Routine fixes for missing data, invalid pairs, or center bookkeeping are not automatically research novelty. First establish a faithful baseline; propose a methodological change only when evidence supports it and relevant prior work has been examined.

## 6. Agreed next work

The user authorized starting Contribution 1, then paused implementation to move to Codex in VS Code. First inspect the local environment and files; then build an isolated centralized/FedAvg CLOSR experiment.

Baseline comparisons should ultimately include the old binary IDS as a contextual reference, centralized CLOSR, and FedAvg CLOSR. A modified federated method is a later comparison only after a demonstrated problem justifies it. Do not compare binary Macro-F1 directly as if it were the same task as known attack-type Macro-F1.

Before any training, define the complete experiment: unknown class categories; fresh initialization; client assignments; preprocessing fit scope; training/validation/final-test separation; class-center computation; batching; absent-class behavior; checkpoint selection; and rejection calibration.

No particular unknown class, threshold target, architecture width, number of rounds, local epoch count, or local validation policy has been approved for this new experiment. Earlier illustrative examples are not experimental decisions. State defensible initial choices explicitly and make them configurable.

Withhold whole attack categories from every client's training and from all fitting/tuning, including preprocessing. Preserve the existing source/group controls. A prior binary experiment exposed all attack types; do not reuse its trained weights, fitted preprocessing, or results as evidence of a fresh unknown-class experiment. Disclose the dataset's previous research use.

The earlier source-disjoint test set contains other sources, not held-out samples from each of the five training clients. Define a defensible held-out local evaluation design before claiming per-client or worst-client generalization. In-sample client checks are diagnostics only.

Do not adopt official script defaults blindly. At the inspected upstream revision, training defaults include batch size 4096 and `drop_last=True`, while these saved clients have fewer than 4096 rows. Direct reuse would yield no full training batches. This is an implementation adaptation requirement, not novelty.

Use the current local environment if it is viable. Python 3.11 is preferred, CPU execution must work, CUDA is optional. Inspect packages and GPU availability rather than assuming them. Keep deployment, crypto, and GUI changes outside this initial research task.
