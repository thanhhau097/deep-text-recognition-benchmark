# this file is use to process data

import string
import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from PIL import Image

from lionelocr.utils.utils import CTCLabelConverter, AttnLabelConverter
from lionelocr.dataset import RawDataset, AlignCollate
from lionelocr.model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
parser.add_argument('--saved_model', help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, help='Transformation stage. None|TPS', default='TPS')
parser.add_argument('--FeatureExtraction', type=str, help='FeatureExtraction stage. VGG|RCNN|ResNet', default='ResNet')
parser.add_argument('--SequenceModeling', type=str, help='SequenceModeling stage. None|BiLSTM', default='BiLSTM')
parser.add_argument('--Prediction', type=str, help='Prediction stage. CTC|Attn', default='Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

opt = parser.parse_args()

""" vocab / character number configuration """
if opt.sensitive:
    opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()


class OCRModel():
    def __init__(self, weights_path, character_file, transformation='TPS', feature_extraction='ResNet', sequence_modeling='BiLSTM', prediction='Attn'):
        self.opt = opt
        self.opt.saved_model = weights_path
        self.opt.Transformation = transformation
        self.opt.FeatureExtraction = feature_extraction
        self.opt.SequenceModeling = sequence_modeling
        self.opt.Prediction = prediction

        with open(character_file, 'r') as f:
            characters = f.readlines()[0].replace('\n', '')
        opt.character = characters

        self.align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)

        self.opt.num_class = len(self.converter.character)

        self.model = Model(self.opt)
        self.model = torch.nn.DataParallel(self.model).to(device)
        self.model.load_state_dict(torch.load(self.opt.saved_model, map_location=device))
        self.model.eval()

    def process(self, image_path):
        if self.opt.rgb:
            image = Image.open(image_path).convert('RGB')  # for color image
        else:
            image = Image.open(image_path).convert('L')

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
