import cv2
import matplotlib.pyplot as plt

# Load a leaf image
img = cv2.imread("data/downloads/plantvillage/Tomato_healthy/e9392a5e-2fac-48c5-a23d-df3aa3f61048___GH_HL Leaf 217.JPG")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 100, 200)

# Show and save
plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.savefig("outputs/edges.png")

print("Saved edges image to outputs/edges.png")
