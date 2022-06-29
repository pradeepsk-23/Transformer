import os
import pandas as pd

from skimage import io
from torch.utils.data import Dataset

class DriveandAct(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 4])
        images = io.imread(img_path)
        labels = str(self.annotations.iloc[index, 7])

        if self.transform:
            images = self.transform(images)

        return (images, labels)
      
       