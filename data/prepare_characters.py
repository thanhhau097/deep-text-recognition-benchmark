# this file uses model to cut the character images from textline image: using craft/ctc
#!/usr/bin/env python
import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm


# we need to get only the characters and merge pickle file
def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))] # should return square? 


# original: TODO: wrong here, number of images and number of labels is not same
with open('auto_dataloader/database.pkl', 'rb') as f:
    origin = pickle.load(f)

print(len(origin[0]), len(origin[1]))
# etl: TODO: remove gray background
with open('auto_dataloader/character_dataset.pkl', 'rb') as f:
    etl = pickle.load(f)

for image, label in tqdm(zip(etl[0], etl[1])):
    # print(image)
    # break
    max_value = np.max(image)
    scale = 255 / max_value
    image = image.astype(np.float64)
    image = image * scale
    image[image > 255] = 255
    image[image > 180] = 255
    image = image.astype(np.uint8)
    origin[0].append(image)
    origin[1].append(label)
print(len(origin[0]), len(origin[1]))

# hw
with open('auto_dataloader/hw_font_characters.pkl', 'rb') as f:
    hw = pickle.load(f)

for image, label in tqdm(zip(hw[0], hw[1])):
    cropped_image = 255 - crop_image(255 - image, 0)
    origin[0].append(cropped_image)
    origin[1].append(label)

    # cv2.imwrite('char.png', cropped_image)
    # break
print(len(origin[0]), len(origin[1]))

# printed
with open('auto_dataloader/printed_font_characters.pkl', 'rb') as f:
    printed = pickle.load(f)

for image, label in tqdm(zip(printed[0], printed[1])):
    cropped_image = 255 - crop_image(255 - image, 0)
    origin[0].append(cropped_image)
    origin[1].append(label)
print(len(origin[0]), len(origin[1]))

with open('auto_dataloader/all_characters.pkl', 'wb') as f:
    pickle.dump(origin, f)
