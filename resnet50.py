import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image

class FER2013Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to the dataset directory (e.g., 'fer2013/train' or 'fer2013/test')
            transform (callable, optional): Optional transform to be applied on a sample
        """
        import os
        from glob import glob
        
        self.root_dir = root_dir
        self.transform = transform
        
        # Emotion labels mapping
        self.emotion_to_idx = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 
            'sad': 4, 'surprise': 5, 'neutral': 6
        }
        self.idx_to_emotion = {v: k for k, v in self.emotion_to_idx.items()}
        
        # Load all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for emotion_folder in os.listdir(root_dir):
            emotion_path = os.path.join(root_dir, emotion_folder)
            if os.path.isdir(emotion_path) and emotion_folder.lower() in self.emotion_to_idx:
                emotion_idx = self.emotion_to_idx[emotion_folder.lower()]
                
                # Get all image files in this emotion folder
                image_files = glob(os.path.join(emotion_path, '*.jpg')) + \
                             glob(os.path.join(emotion_path, '*.png')) + \
                             glob(os.path.join(emotion_path, '*.jpeg'))
                
                for img_path in image_files:
                    self.image_paths.append(img_path)
                    self.labels.append(emotion_idx)
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = FER2013Dataset('/kaggle/input/fer2013/train', transform=train_transform)
test_dataset = FER2013Dataset('/kaggle/input/fer2013/test', transform=val_transform)




train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

batch_size = 32

#Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class model(nn.Module):
    def __init__(self, model_name="resnet50", num_classes = 7):
        super(model, self).__init__()

        self.model = resnet50(pretrained=True)

        # Modify the first convolutional layer for grayscale (3 channels)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the final fully connected layer for 7 classes
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, num_classes)
    def forward(self, x):
        self.num_features(x)
        return self.model.fc(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. Freeze Pre-trained Layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final fully connected layer
for param in model.fc.parameters():
    param.requires_grad = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


def train(model, train_loader, criterion, optimizer, num_epochs=15):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

train(model, train_loader, criterion, optimizer, num_epochs=100)

# 9. Save the Model
torch.save(model.state_dict(), 'fer2013_resnet18_2.pth')