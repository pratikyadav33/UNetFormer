import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os

class CustomDataset():
    def __init__(self, root_dir="data/train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = os.listdir(os.path.join(root_dir, "inputs"))  # List of image filenames

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.root_dir, "images", image_filename)
        mask_path = os.path.join(self.root_dir, "mask", image_filename)
        
        image = Image.open(image_path).convert('L')  # Load grayscale image
        mask = Image.open(mask_path).convert('L')  # Load grayscale mask

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define data loader function
def get_data_loader(root_dir, batch_size, image_size, normalize=True, num_workers=4, pin_memory=True):
    # Define image transformations
    transform_list = [
        Resize(image_size),
        ToTensor()
    ]
    if normalize:
        transform_list.append(Normalize(mean=[0.5], std=[0.5]))  # Normalize to [-1, 1]

    transform = Compose(transform_list)

    # Create dataset
    dataset = CustomDataset(root_dir, transform=transform)

    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )

    return data_loader

