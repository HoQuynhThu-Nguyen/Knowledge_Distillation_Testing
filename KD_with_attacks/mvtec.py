from collections import Counter
import os
import shutil
import torch
import numpy as np
import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from tabulate import tabulate
from PIL import Image
from collections import Counter
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch import nn, optim
# from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU
# from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8Bias, Int4WeightPerTensorFloatDecoupled
from tqdm import tqdm
import matplotlib.pyplot as plt

# List of tools
tools = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
]

# Base path (dataset and notebook in the same directory)
base_path = "./mvtec_dataset"

# Target paths for organized datasets
train_output_path = os.path.join(base_path, "organized_train")
test_output_path = os.path.join(base_path, "organized_test")

os.makedirs(train_output_path, exist_ok=True)
os.makedirs(test_output_path, exist_ok=True)

print("Organizing dataset...\n")

table_data = [["Tool", "Training Images Moved", "Testing Images Moved"]]

for tool in tools:
    tool_path = os.path.join(base_path, tool)
    train_good_path = os.path.join(tool_path, "train", "good")
    test_good_path = os.path.join(tool_path, "test", "good")

    train_tool_output = os.path.join(train_output_path, tool)
    test_tool_output = os.path.join(test_output_path, tool)

    os.makedirs(train_tool_output, exist_ok=True)
    os.makedirs(test_tool_output, exist_ok=True)

    train_count = 0
    test_count = 0

    if os.path.exists(train_good_path):
        for img_file in os.listdir(train_good_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(
                    os.path.join(train_good_path, img_file),
                    os.path.join(train_tool_output, img_file)
                )
                train_count += 1

    if os.path.exists(test_good_path):
        for img_file in os.listdir(test_good_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(
                    os.path.join(test_good_path, img_file),
                    os.path.join(test_tool_output, img_file)
                )
                test_count += 1

    table_data.append([tool, train_count, test_count])

print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

classes = sorted(tools)
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

print("\nClass-to-Index Mapping:")
print(class_to_idx)

class FixedImageFolder(ImageFolder):
    def __init__(self, root, transform, class_to_idx):
        self.class_to_idx = class_to_idx
        self.classes = list(class_to_idx.keys())
        super().__init__(root, transform=transform)
        self.imgs = self.samples

    def find_classes(self, directory):
        return self.classes, self.class_to_idx

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = FixedImageFolder(
    root=train_output_path,
    transform=None,
    class_to_idx=class_to_idx
)

test_dataset = FixedImageFolder(
    root=test_output_path,
    transform=val_test_transforms,
    class_to_idx=class_to_idx
)

assert full_dataset.class_to_idx == class_to_idx, "Mismatch in dataset class_to_idx"
assert test_dataset.class_to_idx == class_to_idx, "Mismatch in test dataset class_to_idx"
assert full_dataset.classes == test_dataset.classes, "Mismatch in class names between datasets"

verification_table = [
    ["Dataset", "Class-to-Index Matches", "Class Names Match"],
    ["Training", full_dataset.class_to_idx == class_to_idx, full_dataset.classes == test_dataset.classes],
    ["Test", test_dataset.class_to_idx == class_to_idx, full_dataset.classes == test_dataset.classes],
]

print("\nVerification of Dataset Integrity:")
print(tabulate(verification_table, headers="firstrow", tablefmt="grid"))

torch.manual_seed(42)
np.random.seed(42)

train_labels = [s[1] for s in full_dataset.samples]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_indices, val_indices in split.split(np.zeros(len(train_labels)), train_labels):
    pass

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.dataset)

train_dataset = TransformedDataset(Subset(full_dataset, train_indices), transform=train_transforms)
val_dataset = TransformedDataset(Subset(full_dataset, val_indices), transform=val_test_transforms)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

train_class_distribution = Counter([full_dataset.samples[i][1] for i in train_indices])
val_class_distribution = Counter([full_dataset.samples[i][1] for i in val_indices])
test_class_distribution = Counter([s[1] for s in test_dataset.samples])

print("\nDataset Statistics:")
dataset_stats = [
    ["Dataset", "Total Samples", "Class Distribution"],
    ["Training", len(train_dataset), dict(train_class_distribution)],
    ["Validation", len(val_dataset), dict(val_class_distribution)],
    ["Test", len(test_dataset), dict(test_class_distribution)],
]
print(tabulate(dataset_stats, headers="firstrow", tablefmt="grid"))


