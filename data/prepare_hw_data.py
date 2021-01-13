import os
import json
import random

from tqdm import tqdm

from create_lmdb_dataset import createDataset

IGNORE_CHARS = [' ', '	', '', '　']

trains = {
    "casia": {
        'annotation_path': '/mnt/ai_filestore/data/flax/HW_datasets/CASIA/train_labels_all_checked.txt',
        'root_dir': '/mnt/ai_filestore/data/flax/HW_datasets/CASIA/'
    },
    'iam': {
        'annotation_path': '/mnt/ai_filestore/data/flax/HW_datasets/IAM/lines_processed/train_checked.txt',
        'root_dir': '/mnt/ai_filestore/data/flax/HW_datasets/IAM/lines_processed/'
    },
    'scut': {
        'annotation_path': '/mnt/ai_filestore/data/flax/HW_datasets/SCUT-EPT_Dataset/SCUT-EPT_labels_train_checked.txt',
        'root_dir': '/mnt/ai_filestore/data/flax/HW_datasets/SCUT-EPT_Dataset/'
    },
    'ffg': {
        'annotation_path': '/mnt/ai_filestore/data/flax/ForFFG/HW_Printed_fixform/train_labels_all_checked.txt',
        'root_dir': '/mnt/ai_filestore/data/flax/ForFFG/HW_Printed_fixform/'
    },
    'invoice': {
        'annotation_path': '/mnt/ai_filestore/data/flax/HW_datasets/Invoice_all/train_835_files/ocr_extraction/ocr_labels_checked.txt',
        'root_dir': '/mnt/ai_filestore/data/flax/HW_datasets/Invoice_all/train_835_files/ocr_extraction/'
    },
}


tests = {
    "casia": {
        'annotation_path': '/mnt/ai_filestore/data/flax/HW_datasets/CASIA/CASIA_Competition_labels_test.txt',
        'root_dir': '/mnt/ai_filestore/data/flax/HW_datasets/CASIA/CASIA-Competition_textline/'
    },
    'iam': {
        'annotation_path': '/mnt/ai_filestore/data/flax/HW_datasets/IAM/lines_processed/val_checked.txt',
        'root_dir': '/mnt/ai_filestore/data/flax/HW_datasets/IAM/lines_processed/'
    },
    'scut': {
        'annotation_path': '/mnt/ai_filestore/data/flax/HW_datasets/SCUT-EPT_Dataset/SCUT-EPT_labels_test_checked.txt',
        'root_dir': '/mnt/ai_filestore/data/flax/HW_datasets/SCUT-EPT_Dataset/'
    },
    'ffg': {
        'annotation_path': '/mnt/ai_filestore/data/flax/ForFFG/HW_Printed_fixform/val_labels_all_checked.txt',
        'root_dir': '/mnt/ai_filestore/data/flax/ForFFG/HW_Printed_fixform/'
    },
    'invoice': {
        'annotation_path': '/mnt/ai_filestore/data/flax/HW_datasets/Invoice_all/test_338_files/ocr_extraction/ocr_labels_checked.txt',
        'root_dir': '/mnt/ai_filestore/data/flax/HW_datasets/Invoice_all/test_338_files/ocr_extraction/'
    },
}


def process_dataset(dataset, kind='train'):
    if kind == 'train':
        data_dict = trains
    else:
        data_dict = tests

    annotation_path = data_dict[dataset]['annotation_path']    
    root_dir = data_dict[dataset]['root_dir']

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    new_annotation_path = './data/hw/{}/{}_annotations.txt'.format(kind, dataset)
    with open(new_annotation_path, 'w') as f:
        for line in lines:
            index = line.index('|')
            label = ''.join([c for c in line[index + 1:] if c not in IGNORE_CHARS])
            if '.png' in label:
                continue
            new_line = line[:index] + '\t' + label
            f.write(new_line)

    createDataset(inputPath=root_dir, gtFile=new_annotation_path, outputPath='data/hw/{}/{}'.format(kind, dataset))
    

def get_all_charset():
    charset = set()
    for dataset in trains.keys():
        charset = charset.union(get_charset_of_annotation_file(trains[dataset]['annotation_path']))
        charset = charset.union(get_charset_of_annotation_file(tests[dataset]['annotation_path']))

    # with open('./data/auto_dataloader/text_to_gen.txt', 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         charset = charset.union(line.replace('\n', ''))

    print(charset)
    print(len(charset))

    charset = sorted(list(charset))
    with open('./data/project_charset.txt', 'w', encoding='utf-8') as f:
        f.write(''.join(charset))

    with open('./data/auto_dataloader/auto_generator_charset.txt', 'w', encoding='utf-8') as f:
        for c in charset:
            f.write(c)
            f.write('\n')


def get_charset_of_annotation_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.replace('\n', '') for line in lines]

    charset = set()
    for line in tqdm(lines):
        index = line.index('|')
        label = ''.join([c for c in line[index + 1:] if c not in IGNORE_CHARS])
        if '.png' in label:
            continue
        charset = charset.union(set(label))

    return charset


def get_text_to_gen_invoice():
    data_dict = trains
    dataset = 'invoice'
    annotation_path = data_dict[dataset]['annotation_path']    
    root_dir = data_dict[dataset]['root_dir']

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    with open('./data/auto_dataloader/text_to_gen.txt', 'w') as f:
        for line in lines:
            index = line.index('|')
            label = ''.join([c for c in line[index + 1:] if c not in IGNORE_CHARS])
            if '.png' in label:
                continue
            f.write(label)


# get_all_charset()
# get_text_to_gen_invoice()

for dataset in trains.keys():
    # try:
        process_dataset(dataset, 'train')
        process_dataset(dataset, 'test')
    # except:
    #     continue