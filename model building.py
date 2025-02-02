import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os

# Define paths
data_dir = r"C:\Users\jeferson fdo\PycharmProjects\Wheat Disease Detection\data\split_dataset"
train_dir = r"C:\Users\jeferson fdo\PycharmProjects\Wheat Disease Detection\data\split_dataset\training data"
val_dir = r"C:\Users\jeferson fdo\PycharmProjects\Wheat Disease Detection\data\split_dataset\validation data"

# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_classes = 5 

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations (ResNet requires 224x224 images)
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load dataset
train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms["train"])
val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms["val"])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Load Pretrained ResNet-18 Model
model = models.resnet18(pretrained=True)

# Freeze early layers (feature extractor)
for param in model.parameters():
    param.requires_grad = False

# Modify the fully connected layer for our 4-class classification
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU (if available)
model = model.to(device)

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss /= total
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_loss /= total
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model

# Train and fine-tune the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Save the trained model
torch.save(trained_model.state_dict(), "wheat_disease_resnet18.pth")
print("✅ Model training complete and saved!")
