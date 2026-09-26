"""
Diagnostic-only kernel: inspects the actual current /kaggle/input mount
structure for the oil-spill-dartis-part1/-part2 datasets, since the v3
training kernel (train_kaggle_v3.py) hard-failed at STEP 2 with:
  ValueError: [!] HARD FAIL: Dataset path
  '/kaggle/input/oil-spill-dartis-part1/Oil/Oil' is missing or empty!
No training, no GPU work -- just os.walk() to find the real path convention
before spending another full run on a guess.
"""
import os

def show_tree(root, max_depth=4, max_entries_per_dir=30):
    if not os.path.exists(root):
        print(f"[!] Path does not exist: {root}")
        return
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/  ({len(dirnames)} dirs, {len(filenames)} files)")
        for f in sorted(filenames)[:5]:
            print(f"{indent}  - {f}")
        if len(filenames) > 5:
            print(f"{indent}  ... ({len(filenames) - 5} more files)")

print("=== /kaggle/input top level ===")
if os.path.exists("/kaggle/input"):
    for entry in sorted(os.listdir("/kaggle/input")):
        full = os.path.join("/kaggle/input", entry)
        print(f"  {entry}  (dir={os.path.isdir(full)})")
else:
    print("[!] /kaggle/input does not exist at all!")

print("\n=== Walking /kaggle/input (depth<=4) ===")
show_tree("/kaggle/input", max_depth=4)

print("\n=== Specific expected paths (as train_kaggle_v3.py hardcodes them) ===")
candidates = [
    "/kaggle/input/oil-spill-dartis-part1",
    "/kaggle/input/oil-spill-dartis-part1/Oil/Oil",
    "/kaggle/input/oil-spill-dartis-part1/Mask_oil/Mask_oil",
    "/kaggle/input/oil-spill-dartis-part2",
    "/kaggle/input/datasets/shuddhabrotabanerjee/oil-spill-dartis-part1",
    "/kaggle/input/datasets/shuddhabrotabanerjee/oil-spill-dartis-part1/Oil/Oil",
]
for c in candidates:
    exists = os.path.exists(c)
    n = len(os.listdir(c)) if exists and os.path.isdir(c) else None
    print(f"  {c}  exists={exists}  entries={n}")

print("\n[+] Diagnostic complete.")
