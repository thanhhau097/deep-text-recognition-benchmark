import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm


def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))] # should return square? 


ROOT_FOLDER = './images'
IGNORE_CHARS = ['0x0000']

crop_percents = {
    'ETL1': 1,
    'ETL2': 1,
    'ETL3': 0.8,
    'ETL4': 1,
    'ETL5': 1,
    'ETL6': 1,
    'ETL7': 1,
    'ETL8G': 1,
    'ETL9G': 0.8,
}

total = 0
character_pkl = [], [] # 0: list of images, 1: list of labels
for dataset_folder in crop_percents.keys():
    crop_ratio = crop_percents[dataset_folder]
    # if '9' not in dataset_folder:
    #     continue
    for character_folder_name in tqdm(os.listdir(os.path.join(ROOT_FOLDER, dataset_folder))):
        if character_folder_name in IGNORE_CHARS:
            continue

        character_folder = os.path.join(ROOT_FOLDER, dataset_folder, character_folder_name)
        char = chr(int(character_folder_name, 16))
        for image_name in os.listdir(character_folder):
            if '.png' not in image_name:
                continue
            
            total += 1
            image_path = os.path.join(character_folder, image_name)
            image = cv2.imread(image_path, 0)
            if 0 in image.shape:
                continue
            if (0 < image.shape[0] < 10 or 0 < image.shape[1] < 10) and char not in '!^*()-_=+[]{}\|/?;:.,一二':
                continue
            h, w = image.shape[:2]
            image = image[int(h * (0.5 - crop_ratio / 2)):int(h * (0.5 + crop_ratio / 2)), int(w * (0.5 - crop_ratio / 2)):int(w * (0.5 + crop_ratio / 2))]
            image = 255 - crop_image(255 - image, 30)

            # verify the width and height of characters. and chars
            character_pkl[0].append(image)
            character_pkl[1].append(char)

            # cv2.imwrite(os.path.join('crop_image', dataset_folder + char + '_' + str(total) + '.png'), image)
            # break
        # if total == 10:
        #     break
with open('character_dataset.pkl', 'wb') as f:
    pickle.dump(character_pkl, f)

            
