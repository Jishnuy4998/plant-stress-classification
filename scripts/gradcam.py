import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

MODEL_PATH = "plant_stress_model.pth"
CLASS_NAMES = ["healthy", "stressed"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

def generate_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    original = np.array(image.resize((224, 224)))
    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward()

    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    activation_map = activations[0]

    for i in range(activation_map.shape[0]):
        activation_map[i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(activation_map, dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(
        cv2.cvtColor(original, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0
    )

    out_path = "outputs/gradcam_result.jpg"
    cv2.imwrite(out_path, superimposed)

    print("Prediction:", CLASS_NAMES[pred_class])
    print("Saved Grad-CAM image to", out_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gradcam.py <image_path>")
    else:
        generate_gradcam(sys.argv[1])
