import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_loader import PascalVOCDataset

def get_transform():
    # Transform to tensor and normalize the images
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def evaluate(model, valid_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_loss = 0
    with torch.no_grad():
        for images, targets in valid_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get the loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            all_loss += losses.item()

    return all_loss / len(valid_loader)

def main():
    # Set up the paths to the dataset
    dataset_root = "L:/LEAFML"

    # Initialize the dataset and dataloaders
    train_dataset = PascalVOCDataset(root=dataset_root, image_set='train', transform=get_transform())
    valid_dataset = PascalVOCDataset(root=dataset_root, image_set='valid', transform=get_transform())

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Load Faster R-CNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Replace the classifier with one specific to our dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=3)

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Set up the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)

            # Compute the total loss
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            losses.backward()

            # Update weights
            optimizer.step()

        # Print epoch loss
        print(f"Epoch #{epoch+1} Loss: {losses.item()}")

        # Validation step
        val_loss = evaluate(model, valid_loader, device)
        print(f"Validation Loss after Epoch {epoch+1}: {val_loss}")

    # Save the model after training
    torch.save(model.state_dict(), "faster_rcnn_leaf_model.pth")

if __name__ == "__main__":
    main()
