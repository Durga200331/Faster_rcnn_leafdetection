import torch
import torchvision
from PIL import Image
import os
from scripts.dataset_loader import PascalVOCDataset
from torchvision import transforms

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def test_model(image_path, model):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = get_transform()(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction

def main():
    # Load the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=3)

    # Load the trained model weights
    model.load_state_dict(torch.load("faster_rcnn_leaf_model.pth"))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Test on a new image
    test_image_path = "L:/LEAFML/dataset/test/images/IMG20250105154716_jpg.rf.c66a7a957642238eeeab3945ae38d4bd.jpg"
    prediction = test_model(test_image_path, model)
    print(prediction)

if __name__ == "__main__":
    main()
