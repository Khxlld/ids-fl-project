"""Execute the Phase 5 guide from a clean kernel with the experiment root as cwd."""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = Path(__file__).resolve().parent
PATH = ROOT / "notebooks" / "07_phase5_post_quantum_security_guide.ipynb"
value = nbformat.read(PATH, as_version=4)
ExecutePreprocessor(timeout=180, kernel_name="python3", allow_errors=False).preprocess(
    value, {"metadata": {"path": str(ROOT)}}
)
nbformat.write(value, PATH)
print(f"Executed {PATH.name}: {len(value.cells)} cells")
