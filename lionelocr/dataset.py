import os
import sys
import re
import six
import math
import lmdb
import torch

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms

from lionelocr.utils.rec_img_aug import RecAug
from lionelocr.utils.utils import normalize_char
try:
    from dataloader.utils.misc import read_config
    from dataloader.generate.line import create_data_generator
    from trdg.generators import GeneratorFromTextFile
except:
    print('Can not import auto dataloader modules')


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, training=True)  # this collate is only used for training set, no worry
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        # add auto generate dataloader
        if opt.use_auto_generate_dataloader:
            _AlignCollate_auto_generator = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, training=False) # training = False to ignore augmentation
            # change the choice of auto generator dataset here
            if opt.auto_generate_dataloader_type == 'trdg':
                _dataset = TRDGAutoGeneratorDataset(text_folder=opt.trdg_text_folder, background_type=opt.trdg_background_type, background_image_dir=opt.trdg_background_image_dir)
            else:
                _dataset = AutoGeneratorDataset(opt.auto_generate_dataloader_config_path)
            _batch_size = opt.auto_generate_dataloader_batch_size
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate_auto_generator, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


# do not augment images generated by auto generator dataset
class AutoGeneratorDataset(Dataset):
    def __init__(self, config_path):
        config = read_config(config_path)
        config['batch_size'] = 1
        self.batch_generator = create_data_generator(config)

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        image, label = self.loop_item()
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        label = [normalize_char(c) for c in label]
        return image, label
    
    def loop_item(self):
        try:
            images, _, labels = next(self.batch_generator)
            if len(labels[0]) > 50:
                return self.loop_item()
            return images[0], labels[0]
        except:
            return self.loop_item()


# update: generate by using background images: check code in TextRecognitionDataGenerator repo: DONE, need to pass image_dir into params, only if background_type == 3
class TRDGAutoGeneratorDataset(Dataset):
    def __init__(self, text_folder='./data/auto_dataloader/texts', background_type=0, background_image_dir=''):
        self.generator = GeneratorFromTextFile(folder=text_folder, count=-1, size=40, language='en',
                                               minimum_length=1, maximum_length=50,
                                               skewing_angle=2, random_skew=True, blur=1, random_blur=True,
                                               background_type=background_type, distorsion_type=-1, margins=(2, 2, 2, 2),
                                               image_dir=background_image_dir)

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        image, label = self.loop_item()
        image = image.convert('RGB')
        label = [normalize_char(c) for c in label]
        return image, label

    def loop_item(self):
        try:
            image, label = next(self.generator)
            if len(label) > 50:
                return self.loop_item()
            return image, label
        except:
            return self.loop_item()

class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.opt.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label): # if re.search(out_of_char, label.lower()):
                        print(re.search(out_of_char, label), label)
                        continue
                    # CURRENT ERROR: ignore some label with no reason

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            # if not self.opt.sensitive:  # update character list to comment out these 2 lines
            #     label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)
            label = [normalize_char(c) for c in label]
            
        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class StandardDataset(Dataset):
    def __init__(self, root, annotation_files, opt):
        self.root = root
        self.opt = opt
        if type(annotation_files) == str:
            annotation_files = [annotation_files]
        elif type(annotation_files) not in (list, set, tuple):
            raise ValueError("Error: Only support one or more annotation files. Type: str, list, set, tuple")
        
        self.image_paths, self.labels = self.read_data(annotation_files)

    def read_data(self, annotation_files):
        image_paths = []
        labels = []

        for file in annotation_files:
            with open(file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.replace('\n', '')
                image_paths.append(line.split('\t')[0])
                labels.append(line.split('\t')[1])

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        try:
            if self.opt.rgb:
                img = Image.open(os.path.join(self.root, self.image_paths[index])).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_paths[index]).convert('L')

            label = self.labels[index]

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))
            
            label = ''

        return img, label


class OCRDataset(object):
    def __init__(self, root, annotation_files, opt, training):
        collate_fn = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, training=training)  # this collate is only used for training set, no worry
        standard_dataset = StandardDataset(root, annotation_files, opt)
        self.standard_dataloader = torch.utils.data.DataLoader(
            standard_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.workers),
            collate_fn=collate_fn, pin_memory=True)
        
        self.data_loader_list = [self.standard_dataloader]
        self.dataloader_iter_list = [iter(self.standard_dataloader)]
        
        if opt.use_auto_generate_dataloader:
            _AlignCollate_auto_generator = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, training=False) # training = False to ignore augmentation
            # change the choice of auto generator dataset here
            if opt.auto_generate_dataloader_type == 'trdg':
                _dataset = TRDGAutoGeneratorDataset(text_folder=opt.trdg_text_folder, background_type=opt.trdg_background_type, background_image_dir=opt.trdg_background_image_dir)
            else:
                _dataset = AutoGeneratorDataset(opt.auto_generate_dataloader_config_path)
            _batch_size = opt.auto_generate_dataloader_batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate_auto_generator, pin_memory=True)

            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

            print('TOTAL BATCH_SIZE = {} + {} = {}'.format(opt.batch_size, 
                opt.auto_generate_dataloader_batch_size,
                opt.batch_size + opt.auto_generate_dataloader_batch_size))
        else:
             print('TOTAL BATCH_SIZE = {}'.format(opt.batch_size))

    def get_batch(self):
        batch_images = []
        batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                batch_images.append(image)
                batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                batch_images.append(image)
                batch_texts += text
            except ValueError:
                pass

        batch_images = torch.cat(batch_images, 0)

        return batch_images, batch_texts


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, training=False):
        """
        generated_data_type: "hw", "printed", "mix", None
        """
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.training = training
        self.aug = RecAug()

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                if self.training:
                    image = augment(self.aug, image)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            if self.training:
                images = [augment(self.aug, image) for image in images]
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def augment(aug_fn, image):
    image = np.asarray(image, dtype=np.float32)
    image = aug_fn(image)
    image = Image.fromarray(np.uint8(image))
    # import random
    # image.save('./data/augmented_images/{}.png'.format(str(random.randint(1, 100000))))
    return image


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
