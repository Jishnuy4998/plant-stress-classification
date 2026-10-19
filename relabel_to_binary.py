import os
import shutil
from pathlib import Path

SOURCE_DIR = "data/downloads/plantvillage"
DEST_DIR = "data/raw"

VALID_EXTS = {".jpg", ".jpeg", ".png"}

def main():
    source = Path(SOURCE_DIR)
    healthy_dir = Path(DEST_DIR) / "healthy"
    stressed_dir = Path(DEST_DIR) / "stressed"

    healthy_dir.mkdir(parents=True, exist_ok=True)
    stressed_dir.mkdir(parents=True, exist_ok=True)

    healthy_count = 0
    stressed_count = 0

    for class_folder in source.iterdir():

        if not class_folder.is_dir():
            continue

        is_healthy = "healthy" in class_folder.name.lower()

        for img in class_folder.glob("*"):

            if img.suffix.lower() not in VALID_EXTS:
                continue

            new_name = f"{class_folder.name}_{img.name}"

            if is_healthy:
                shutil.copy(img, healthy_dir / new_name)
                healthy_count += 1
            else:
                shutil.copy(img, stressed_dir / new_name)
                stressed_count += 1

    print("Done.")
    print("Total healthy images:", healthy_count)
    print("Total stressed images:", stressed_count)

if __name__ == "__main__":
    main()