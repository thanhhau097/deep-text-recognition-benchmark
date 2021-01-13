import os
import json
import random

from create_lmdb_dataset import createDataset


with open('./data/cellcuts.json', 'r') as f:
    data = json.load(f)

data = list(data.items())
random.shuffle(data)
train_data = data[: int(0.8 * len(data))]
val_data = data[int(0.8 * len(data)):]

with open('./data/train_cellcuts.txt', 'w') as f:
    for image_path, label in train_data:
        f.write(image_path)
        f.write('\t')
        f.write(label)
        f.write('\n')

with open('./data/val_cellcuts.txt', 'w') as f:
    for image_path, label in val_data:
        f.write(image_path)
        f.write('\t')
        f.write(label)
        f.write('\n')

createDataset(inputPath='data/', gtFile='./data/train_cellcuts.txt', outputPath='data/train/cellcuts')
createDataset(inputPath='data/', gtFile='./data/val_cellcuts.txt', outputPath='data/val/cellcuts')