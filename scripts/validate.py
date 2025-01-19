import torch
import torchvision
from torch.utils.data import DataLoader
from scripts.dataset_loader import PascalVOCDataset
from torchvision import models, transforms
import os

# Paths to dataset
valid_images_dir = "L:/LEAFML/dataset/valid/images"
valid_annotations_dir = "L:/LEAFML/dataset/valid/annotations"

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create transformations (optional)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load dataset
valid_dataset = PascalVOCDataset(valid_images_dir, valid_annotations_dir, transforms=transform)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 2)  # 2 classes: leaf and background
model.load_state_dict(torch.load("L:/LEAFML/models/faster_rcnn_epoch_X.pth"))  # Replace with the correct epoch
model.to(device)

# Validation loop
model.eval()
with torch.no_grad():
    for images, targets in valid_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        print(loss_dict)

print("Validation complete!")
