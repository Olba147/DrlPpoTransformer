import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    src_dir = Path(__file__).resolve().parent / "src"
    src_script = src_dir / "train_PPO_initial.py"
    sys.path.insert(0, str(src_dir))
    sys.argv = [str(src_script), *sys.argv[1:]]
    runpy.run_path(str(src_script), run_name="__main__")
