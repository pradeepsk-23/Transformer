import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

# Dataset Class
class HAR(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class2index = {'closing_backpack':0, 'closing_bottle':1, 'closing_door_inside':2, 
                            'closing_door_outside':3, 'closing_laptop':4, 'drinking':5, 
                            'eating':6, 'entering_car':7, 'exiting_car':8, 
                            'fastening_seat_belt':9, 'fetching_an_object':10, 'interacting_with_phone':11, 
                            'looking_back_left_shoulder':12, 'looking_back_right_shoulder':13, 'looking_or_moving_around (e.g. searching)':14, 
                            'moving_towards_door':15, 'opening_backpack':16, 'opening_bottle':17, 
                            'opening_door_inside':18, 'opening_door_outside':19, 'opening_laptop':20, 
                            'placing_an_object':21, 'preparing_food':22, 'pressing_automation_button':23, 
                            'putting_laptop_into_backpack':24, 'putting_on_jacket':25, 'putting_on_sunglasses':26, 
                            'reading_magazine':27, 'reading_newspaper':28, 'sitting_still':29, 
                            'standing_by_the_door':30, 'taking_laptop_from_backpack':31, 'taking_off_jacket':32, 
                            'taking_off_sunglasses':33, 'talking_on_phone':34, 'unfastening_seat_belt':35, 
                            'using_multimedia_display':36, 'working_on_laptop':37, 'writing':38}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.df.iloc[index, 0]))
        label = self.class2index[self.df.iloc[index, 7]]

        if self.transform:
            image = self.transform(image)
   
        return image, label

import pandas as pd
train = pd.read_csv("D:\IRP\GitHub\Transformer\Drive&Act\Frame.csv")
classes = sorted(train["activity"].unique().tolist())
print(len(classes))