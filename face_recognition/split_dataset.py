import os
from pathlib import Path
import random
import shutil

def split_dataset(dataset_path="dataset", train_ratio=0.8):
    dataset_path = Path(dataset_path)
    train_path = Path("train")
    test_path = Path("test")
    train_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)

    for label_folder in dataset_path.iterdir():
        if label_folder.is_dir():
            # Create corresponding folders in train and test
            (train_path / label_folder.name).mkdir(exist_ok=True)
            (test_path / label_folder.name).mkdir(exist_ok=True)
            
            # Get all images in the label folder
            images = list(label_folder.glob("*.jpg"))
            random.shuffle(images)
            train_count = int(len(images) * train_ratio)
            
            # Split into train and test
            for i, img in enumerate(images):
                if i < train_count:
                    shutil.copy(img, train_path / label_folder.name / img.name)
                else:
                    shutil.copy(img, test_path / label_folder.name / img.name)
    print("Dataset split into train and test folders")

if __name__ == "__main__":
    split_dataset()