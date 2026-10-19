import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# -------- Rebuild model architecture --------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

# -------- Load trained weights --------
state_dict = torch.load("plant_stress_model.pth", map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

# -------- Load image --------
img_path = "data/downloads/plantvillage/Tomato_Early_blight/0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B 8389.JPG"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# -------- Capture feature maps --------
feature_maps = []

def hook_fn(module, input, output):
    feature_maps.append(output.detach())

# Hook first convolution layer
model.conv1.register_forward_hook(hook_fn)

# Forward pass
with torch.no_grad():
    model(input_tensor)

# -------- Plot feature maps --------
fm = feature_maps[0][0]

fig = plt.figure(figsize=(8,8))

for i in range(16):
    ax = fig.add_subplot(4,4,i+1)
    ax.imshow(fm[i].cpu(), cmap="viridis")
    ax.axis("off")

plt.tight_layout()

Path("outputs").mkdir(exist_ok=True)
plt.savefig("outputs/feature_maps.png")

print("Saved feature maps to outputs/feature_maps.png")
