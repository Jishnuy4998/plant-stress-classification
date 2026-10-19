import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

MODEL_PATH = "plant_stress_model.pth"
CLASS_NAMES = ["healthy", "stressed"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    print("Prediction:", CLASS_NAMES[predicted.item()])
    print("Confidence:", round(confidence.item(), 4))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_single.py <image_path>")
    else:
        predict_image(sys.argv[1])

