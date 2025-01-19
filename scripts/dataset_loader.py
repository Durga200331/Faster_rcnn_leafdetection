import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.transforms import functional as F

class PascalVOCDataset(Dataset):
    def __init__(self, root, image_set='train', transform=None):
        """
        Args:
            root (str): Root directory of the dataset.
            image_set (str): One of 'train', 'valid', 'test' to specify which set to use.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.image_set = image_set
        self.transform = transform

        # Image and annotation file paths
        self.images_path = os.path.join(self.root, 'dataset', image_set, 'images')
        self.annotations_path = os.path.join(self.root, 'dataset', image_set, 'annotations')

        # Get all image file names
        self.image_files = [f for f in os.listdir(self.images_path) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get the image file
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_name)

        # Open the image
        image = Image.open(img_path).convert("RGB")

        # Parse the corresponding annotation
        annotation_path = os.path.join(self.annotations_path, img_name.replace('.jpg', '.xml'))
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        # Extract bounding boxes and labels from the annotation file
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name == 'leaves':
                label = 1
            elif name == 'half-leaf':
                label = 2
            else:
                continue  # Ignore any other classes

            # Get the bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Add the bounding box and label to the lists
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels}

        return image, target
