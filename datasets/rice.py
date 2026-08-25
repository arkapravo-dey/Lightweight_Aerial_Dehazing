from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import re

def is_duplicate(filename):
    # Match filenames like '022 (1).png', 'sample (2).jpg', etc.
    return re.search(r'\(\d+\)', filename) is not None

class RICEDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.input_dir = os.path.join(root_dir, 'input')   # Hazy images
        self.target_dir = os.path.join(root_dir, 'target') # Ground truth images
        self.transform = transform
        self.image_filenames = [
            f for f in sorted(os.listdir(self.input_dir))
            if not is_duplicate(f) and os.path.exists(os.path.join(self.target_dir, f))
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.image_filenames[idx])
        target_path = os.path.join(self.target_dir, self.image_filenames[idx])
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            
        return {'source': input_img, 'target': target_img} 

transform = transforms.Compose([
    transforms.ToTensor()         # Convert to PyTorch tensor
])
