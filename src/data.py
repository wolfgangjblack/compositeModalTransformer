import os
import torch
from torch.utils.data import Dataset

class CustomImageTextDataset(Dataset):
    def __init__(self, data_dir, target_transform=None):
        self.data_dir = data_dir        
        self.files = self.get_files()
        self.labels = self.get_files(True)
        self.target_transform = target_transform
        
    def get_files(self, labels = False):
        files = []
        for i in os.listdir(self.data_dir):
            if labels:
                imgs = [i for j in os.listdir(os.path.join(self.data_dir,i)) if j.endswith(".jpg")]
            else:
                imgs = [os.path.join(self.data_dir, i, j.split('.')[0]) for j in os.listdir(os.path.join(self.data_dir,i)) if j.endswith(".jpg")]
            files.extend(imgs)
        return files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx] + '.jpg'
        txt_path = self.files[idx] + '.txt'
        tag_path = self.files[idx] + '_tags.txt'
        # image = Image.open(img_path)
        try:
            text = open(txt_path, 'r').read()
        except:
            text = ''
        try:
            tags = open(tag_path, 'r').read()
        except:
            tags = None
        label = self.labels[idx]
        if self.target_transform:
            label = self.target_transform[label]
            if isinstance(label, int):
                label = torch.tensor(label)
        return img_path, text, tags, label
    
    def __repr__(self):
        string = f"CustomImageTextDataset(num_samples={len(self.files)}, data_dir='{self.data_dir}')\n"
        return string + f"Unique Labels: {list(set(self.labels))} for phase: {os.path.basename(self.data_dir)}"
    
