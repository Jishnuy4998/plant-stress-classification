from PIL import Image
from pathlib import Path

# Input image
img_path = Path("data/downloads/plantvillage/Tomato_healthy/e9392a5e-2fac-48c5-a23d-df3aa3f61048___GH_HL Leaf 217.JPG")

from PIL import Image
from pathlib import Path

# Input image
img_path = Path("data/downloads/plantvillage/Tomato_healthy/e9392a5e-2fac-48c5-a23d-df3aa3f61048___GH_HL Leaf 217.JPG")

# Output folder
out_dir = Path("outputs/preprocessing_examples")
out_dir.mkdir(parents=True, exist_ok=True)

# Open original
img = Image.open(img_path).convert("RGB")

# Save original
img.save(out_dir / "original.jpg")

# Resize to CNN input size
resized = img.resize((224, 224))
resized.save(out_dir / "resized_224x224.jpg")

# Augmentation 1: rotate
rotated = resized.rotate(20, expand=False)
rotated.save(out_dir / "augmented_rotated.jpg")

# Augmentation 2: horizontal flip
flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
flipped.save(out_dir / "augmented_flipped.jpg")

print("Saved preprocessing images to:", out_dir)
