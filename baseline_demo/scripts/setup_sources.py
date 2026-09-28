"""Initialize pinned submodules and apply the saved PyCOLMAP compatibility patch."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "baseline_demo"


def git(*args, cwd=ROOT, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check)


def main():
    manifest = json.loads((BASE / "experiment_manifest.json").read_text(encoding="utf-8"))
    git("submodule", "update", "--init", "--recursive")
    for name in ("vggt", "gsplat"):
        folder = BASE / "vendor" / name
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=folder, text=True
        ).strip()
        if actual != manifest[f"{name}_commit"]:
            raise RuntimeError(f"Unexpected {name} commit: {actual}")
    folder = BASE / "vendor" / "vggt"
    patch = str(BASE / "vggt_pycolmap_compat.patch")
    if git("apply", "--reverse", "--check", patch, cwd=folder, check=False).returncode == 0:
        print("PyCOLMAP compatibility patch already applied.")
    else:
        git("apply", "--check", patch, cwd=folder)
        git("apply", patch, cwd=folder)
        print("Applied PyCOLMAP compatibility patch.")


if __name__ == "__main__":
    main()
