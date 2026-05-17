"""Count lines of Python and C/C++ code under src/opd/."""
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "opd"

py_lines = 0
c_lines = 0

for f in SRC_ROOT.rglob("*"):
    if not f.is_file():
        continue
    if f.suffix == ".py":
        py_lines += sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))
    elif f.suffix in (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"):
        c_lines += sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))

print(f"Python: {py_lines} lines")
print(f"C/C++:  {c_lines} lines")
print(f"Total:  {py_lines + c_lines} lines")
