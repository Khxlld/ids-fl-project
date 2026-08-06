"""Execute all guide notebooks from clean kernels with the experiment root as cwd."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    ROOT / "notebooks" / "05_phase3_binary_modeling_guide.ipynb",
    ROOT / "notebooks" / "06_phase4_docker_federated_demo_guide.ipynb",
    ROOT / "notebooks" / "07_phase5_post_quantum_security_guide.ipynb",
]


for path in NOTEBOOKS:
    value = nbformat.read(path, as_version=4)
    executor = ExecutePreprocessor(timeout=180, kernel_name="python3", allow_errors=False)
    executor.preprocess(value, {"metadata": {"path": str(ROOT)}})
    nbformat.write(value, path)
    print(f"Executed {path.name}: {len(value.cells)} cells")
