# this file is use to process data

import string
import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
import numpy as np

from lionelocr.utils.utils import CTCLabelConverter, AttnLabelConverter
from lionelocr.dataset import RawDataset, AlignCollate
from lionelocr.model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
opt = parser.parse_known_args()[0]

cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()


class OCRModel():
    def __init__(self, weights_path):
        self.opt = opt

        # load value for opt
        self.opt.saved_model = weights_path
        weights_dict = torch.load(self.opt.saved_model, map_location=device)

        self.opt.batch_max_length = weights_dict['batch_max_length']
        self.opt.imgH = weights_dict['imgH']
        self.opt.imgW = weights_dict['imgW']
        self.opt.rgb = weights_dict['rgb']
        self.opt.character = weights_dict['character']
        self.opt.PAD = weights_dict['PAD']
        self.opt.Transformation = weights_dict['Transformation']
        self.opt.FeatureExtraction = weights_dict['FeatureExtraction']
        self.opt.SequenceModeling = weights_dict['SequenceModeling']
        self.opt.Prediction = weights_dict['Prediction']
        self.opt.num_fiducial = weights_dict['num_fiducial']
        self.opt.input_channel = weights_dict['input_channel']
        self.opt.output_channel = weights_dict['output_channel']
        self.opt.hidden_size = weights_dict['hidden_size']

        self.align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)

        self.opt.num_class = len(self.converter.character)

        self.model = Model(self.opt)
        self.model = torch.nn.DataParallel(self.model).to(device)
        self.model.load_state_dict(weights_dict['state_dict'])
        self.model.eval()

    def process(self, input_image):
        # use both string and numpy array as input
        if type(input_image) == str:
            if self.opt.rgb:
                image = Image.open(input_image).convert('RGB')  # for color image
            else:
                image = Image.open(input_image).convert('L')
        elif type(input_image) == np.array:
            image = Image.fromarray(input_image)
        else:
            raise ValueError('Only accept image path or numpy array as input')

        image_tensors, _ = self.align_collate([(image, '')])
        image_tensors = image_tensors.to(device)
        batch_size = image_tensors.size(0)

        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in opt.Prediction:
            preds = self.model(image_tensors, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = self.converter.decode(preds_index, preds_size)

        else:
            preds = self.model(image_tensors, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        pred_max_prob = preds_max_prob[0]
        pred = preds_str[0]
        if 'Attn' in opt.Prediction:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

        # calculate confidence score (= multiply of pred_max_prob)
        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

        print('predict:', pred)
        print('confidence_score:', confidence_score)
        return pred, confidence_score


if __name__ == "__main__":
    ocr = OCRModel(weights_path='./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth')
    ocr.process(image_path='./data/cellcuts/IMG_20201228_183149_037.png')
