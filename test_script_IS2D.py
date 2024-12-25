import torch
import torch.nn as nn
import torch.optim as optim
from resnest.torch import resnest50
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

# Define the dataset class
class SegmentationDatasetWithMapping():
    def __init__(self, mapping_file, image_folder, mask_folder, image_size=(256, 256), transform=None,
                 target_transform=None):
        self.mapping = pd.read_excel(mapping_file)
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        image_name = self.mapping.iloc[idx, 1]
        mask_name = self.mapping.iloc[idx, 2]

        image_path = os.path.join(self.image_folder, image_name)
        mask_path = os.path.join(self.mask_folder, mask_name)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        mask = mask.squeeze(0)  # Remove extra dimensions if necessary
        return image, mask


# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define the model architecture (same as the one you used for training)
class ResNeStSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(ResNeStSegmentation, self).__init__()
        self.backbone = resnest50(pretrained=True)
        self.backbone.fc = nn.Identity()

        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)

        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)

        output = self.decoder(features)
        output = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
        return output


# Define the Dice coefficient calculation function for multi-class
def calculate_DSC(y_pred, y_true, num_classes=4):
    dice_scores = []

    for i in range(num_classes):
        y_pred_class = (y_pred == i).float()
        y_true_class = (y_true == i).float()

        intersection = torch.sum(y_pred_class * y_true_class)
        segmentation_pixels = torch.sum(y_pred_class)
        gt_label_pixels = torch.sum(y_true_class)

        dice_score = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)
        dice_scores.append(dice_score.item())

    return np.mean(dice_scores)


# Load the trained model (adjust the path to your model's checkpoint)
def load_model(model_path, device, num_classes):
    model = ResNeStSegmentation(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model


# Testing function
def test(model, loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    dice_score = 0.0
    num_batches = len(loader)

    # Prepare lists to store all images, masks, and predictions for visualization
    all_images = []
    all_masks = []
    all_preds = []

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)  # Get the model's predictions

            # Calculate loss (CrossEntropyLoss)
            loss = criterion(outputs, masks.long())
            total_loss += loss.item()

            # Convert outputs to class predictions (argmax over the class dimension)
            _, predicted = torch.max(outputs, 1)

            # Calculate number of correct predictions
            correct += (predicted == masks).sum().item()
            total += masks.numel()

            # Compute Dice coefficient
            dice_score += calculate_DSC(predicted, masks)

            # Collect all images, masks, and predictions
            all_images.extend(images.cpu())
            all_masks.extend(masks.cpu())
            all_preds.extend(predicted.cpu())

        avg_loss = total_loss / num_batches
        accuracy = correct / total
        avg_dice_score = dice_score / num_batches

    # Create a large grid to display all test images, ground truth masks, and predicted masks
    num_samples = len(all_images)
    grid_size = int(np.ceil(np.sqrt(num_samples)))  # Determine grid size

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    # Flatten the axes to easily iterate over them
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(all_images[i].permute(1, 2, 0).numpy())
        ax.set_title(f"GT / Pred {i+1}")
        ax.axis('off')

        # Optionally show a combination of GT and Pred for each image
        # You can display both GT and predicted masks next to the image or overlay them
        mask = all_masks[i].numpy()
        pred = all_preds[i].numpy()

        # Visualize the ground truth mask (shown in blue)
        ax.imshow(mask, alpha=0.5, cmap='Blues')

        # Visualize the predicted mask (shown in red)
        ax.imshow(pred, alpha=0.5, cmap='Reds')

    # Hide any empty subplots if there are not enough samples
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return avg_loss, accuracy, avg_dice_score


# Main function to run the test
def run_test():
    # Adjust these paths as per your dataset and trained model
    test_image_folder = "./dataset/BioMedicalDataset/COVID19/images"
    test_mask_folder = "./dataset/BioMedicalDataset/COVID19/masks"
    test_mapping_file = "./dataset/BioMedicalDataset/COVID19/test_frame.xlsx"
    model_path = "./model_weights/COVID19/model_weights/model_weight(EPOCH 20).pth.tar"  # Change this to your model path

    # Initialize the test dataset and dataloader
    test_dataset = SegmentationDatasetWithMapping(
        mapping_file=test_mapping_file,
        image_folder=test_image_folder,
        mask_folder=test_mask_folder,
        transform=transform,
        target_transform=target_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Define device, criterion and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, num_classes=4)
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class segmentation

    # Run the test
    test_loss, test_accuracy, test_dice_score = test(model, test_loader, criterion, device)

    # Print the test results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Dice Score: {test_dice_score:.4f}")


if __name__ == "__main__":
    run_test()
