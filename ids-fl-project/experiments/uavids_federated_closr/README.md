# Federated CLOSR Contribution 1 baseline

Peer reviewers should begin with `PEER_REVIEW_README.md`, which describes the review scope, external-data setup, selected evidence, findings, limitations, and requested review questions.

This isolated experiment provides centralized CLOSR, IID sample-weighted FedAvg CLOSR, and natural five-client Non-IID sample-weighted FedAvg CLOSR for the existing UAVIDS Phase 2 partitions. All three use CLOSR only, with the same architecture, objective, initialization, inference, training pool, and evaluation partitions. Read `EXPERIMENT_SPEC.md` before changing the unknown class, split, loss, centers, IID control, or calibration policy.

The implementation is CPU-compatible and does not load the final test partition during training or smoke execution. Successful tests or smoke execution establish implementation behavior only.

## Local Windows setup

Python 3.11 was preferred but was not installed on this machine. The verified available interpreter is Python 3.12.1, which is compatible with the pinned packages.

From PowerShell:

```powershell
Set-Location 'C:\Users\feras\Downloads\KAUST 4th stage\glasgow\Project\useCases\smartCity\ids-fl-project-federated-closr\ids-fl-project\experiments\uavids_federated_closr'

& 'C:\Users\feras\Downloads\KAUST 4th stage\glasgow\Project\useCases\military theme datasets\experiments\uavids_federated\.venv\Scripts\python.exe' -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pip install -e . --no-deps --no-build-isolation
```

The CPU-only PyTorch index is pinned in `requirements.txt`. CUDA is not available locally and is not required.

## Verification and bounded smoke run

```powershell
& '.\.venv\Scripts\python.exe' -m pytest

& '.\.venv\Scripts\python.exe' .\run_experiment.py `
  --smoke `
  --output-dir .\runs\smoke-20261017
```

The runner refuses to overwrite an existing output directory. Smoke mode uses capped, known-only rows, one centralized epoch, and one federated round. It hash-checks the final test file but never loads or evaluates it.

Generated run directories and the virtual environment are ignored by the experiment-local `.gitignore`.

## Future full baseline command

After reviewing the frozen specification and smoke handover, the next bounded baseline can be started with:

```powershell
& '.\.venv\Scripts\python.exe' .\run_experiment.py `
  --output-dir .\runs\contribution1-three-way-seed-240517
```

This trains and calibrates the three CLOSR models but still does not evaluate the final test partition. Final evaluation should be a separate, one-time stage after the training design and reporting protocol are accepted.

## Important paths

- Configuration: `config/contribution1_baseline.json`
- Package: `src/uavids_closr/`
- Tests: `tests/`
- Method specification: `EXPERIMENT_SPEC.md`
- Upstream code attribution: `UPSTREAM_ATTRIBUTION.md`
- External populated data root: `C:\Users\feras\Downloads\KAUST 4th stage\glasgow\Project\useCases\military theme datasets\experiments\uavids_federated`
