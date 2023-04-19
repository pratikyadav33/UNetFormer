import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from load import CustomDataset  # Custom dataset class
from unet import FTUnetformer  # FTUnetformer model implementation

# Define hyperparameters
num_classes = 16
batch_size = 4
learning_rate = 0.001
num_epochs = 10

# Instantiate the model
model = FTUnetformer(num_classes)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Choose optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create dataset and data loaders
train_dataset = CustomDataset()  # Replace with your custom dataset class instantiation for training data
val_dataset = CustomDataset()  # Replace with your custom dataset class instantiation for validation data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Zero the gradients of the model's parameters
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()

        # Update model's parameters
        optimizer.step()

        # Print training progress
        print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
              .format(epoch+1, num_epochs, batch_idx+1, len(train_loader), loss.item()))

    # Evaluate the model on validation dataset
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        total_loss = 0
        for val_batch_idx, (val_images, val_masks) in enumerate(val_loader):
            val_images = val_images.to(device)
            val_masks = val_masks.to(device)

            # Forward pass
            val_outputs = model(val_images)

            # Compute loss
            val_loss = criterion(val_outputs, val_masks)
            total_loss += val_loss.item()

        avg_val_loss = total_loss / len(val_loader)
        print('Validation Loss: {:.4f}'.format(avg_val_loss))

# Save the trained model's parameters
torch.save(model.state_dict(), 'ftunetformer.pth')
print('Model trained and saved!')

