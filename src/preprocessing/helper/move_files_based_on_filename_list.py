"""
Move files from one directory to another based on a .txt file containing the file base keys such as: colon_0001-prone
"""

import shutil
from pathlib import Path

SOURCE_FOLDER = Path("")
DEST_FOLDER   = Path("")
BASE_KEYS_FILE = Path("")

base_keys = BASE_KEYS_FILE.read_text().splitlines()
base_keys = [k.strip() for k in base_keys if k.strip()]

DEST_FOLDER.mkdir(parents=True, exist_ok=True)

copied = 0
for f in SOURCE_FOLDER.iterdir():
    if any(key in f.name for key in base_keys):
        shutil.copy2(f, DEST_FOLDER / f.name)
        copied += 1

print(f"Copied {copied} files to {DEST_FOLDER}")
