#!/usr/bin/env bash
set -euo pipefail

setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_dir="$(dirname -- "$setup_dir")/.venv"

"$env_dir/bin/python" - "$env_dir" <<'PY'
import pathlib
import site
import sys
import pip

expected = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(sys.prefix).resolve() == expected, sys.prefix
assert sys.prefix != sys.base_prefix, "Python is not isolated"
assert site.ENABLE_USER_SITE is False, "User packages are enabled"
config = (expected / "pyvenv.cfg").read_text()
assert "include-system-site-packages = false" in config
assert pathlib.Path(pip.__file__).resolve().is_relative_to(expected)
print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("Environment:", sys.prefix)
print("Isolation: passed")
PY

"$env_dir/bin/python" -m pip --version
git -C "$(dirname -- "$setup_dir")" check-ignore -- .venv/pyvenv.cfg
echo "Environment checks passed."
