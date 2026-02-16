import os
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    # Work around duplicate OpenMP runtimes in mixed conda/pip stacks on Windows.
    # os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    src_dir = Path(__file__).resolve().parent / "src"
    src_script = src_dir / "train_jepa_initial.py"
    sys.path.insert(0, str(src_dir))
    sys.argv = [str(src_script), *sys.argv[1:]]
    runpy.run_path(str(src_script), run_name="__main__")
