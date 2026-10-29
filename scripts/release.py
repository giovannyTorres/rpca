"""Validate and build distributions; upload only when --publish is explicit."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run(*args: str, cwd: Path) -> None:
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Upload validated files")
    parser.add_argument("--repository", choices=("pypi", "testpypi"), default="pypi")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run("-m", "pytest", "-W", "error", "-m", "not gpu", cwd=root)
    run("-m", "ruff", "check", "src", "tests", "examples", "scripts", cwd=root)
    with tempfile.TemporaryDirectory(prefix="rpca-kit-release-") as temporary:
        build_dir = Path(temporary)
        run("-m", "build", "--outdir", str(build_dir), cwd=root)
        artifacts = sorted(build_dir.glob("*.whl")) + sorted(build_dir.glob("*.tar.gz"))
        if len(artifacts) != 2:
            raise RuntimeError("Expected exactly one wheel and one source distribution")
        run("-m", "twine", "check", *(str(path) for path in artifacts), cwd=root)
        destination = root / "dist"
        destination.mkdir(exist_ok=True)
        for artifact in artifacts:
            shutil.copy2(artifact, destination / artifact.name)
        if args.publish:
            run(
                "-m", "twine", "upload", "--non-interactive", "--repository",
                args.repository, *(str(path) for path in artifacts), cwd=root,
            )
        else:
            print("Validated distributions copied to dist/. No upload requested.")


if __name__ == "__main__":
    main()
