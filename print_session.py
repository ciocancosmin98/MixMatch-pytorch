import sys
import argparse
import os
from torch import load
from torch import from_numpy
import pickle
import cv2
import numpy as np
import random

from dataset.custom_dataset import Preprocessor, resize_and_split
import models.wideresnet as models
from session import TrainState


short_consts   = ['dataset_name', 'session_id', 'enable_mixmatch', 'n_labeled']
verbose_consts = ['epochs', 'train_iteration', 'batch_size', 'lr', 'lambda_u', 'T', 'alpha']

short_vars = ['epoch', 'acc', 'best_acc']
verbose_vars = []

LABEL_LENGTH = 30

parser = argparse.ArgumentParser(description='Session information tool')

parser.add_argument('--dataset-name', '-name', default='animals10', type=str, metavar='NAME',
                    help='name of the dataset')
parser.add_argument('--session-id', '-id', default=-1, type=int, metavar='ID',
                    help='the id of the session whose information will be printed')
parser.add_argument('--verbose', '-v', action="store_true",
                    help='show more information')
parser.add_argument('--show-predictions', '-s', default=-1, type=int, metavar='N_IMAGES',
                    help='number of images per class to predict on')

args = parser.parse_args()

def get_wideresnet_models(n_classes):
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.WideResNet(num_classes=n_classes)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    return model, ema_model

def get_image_filenames(session_dir):
    filename_list_path = os.path.join(session_dir, 'preprocessing', 'filenames_list.pkl')

    if not os.path.exists(filename_list_path):
        raise Exception("Session exists but the list of filenames does not exist")

    save_file = open(filename_list_path, "rb")
    _, _, val, test = pickle.load(save_file)

    return test

def get_image(predictions):
    random.seed(42)

    SIZE = 80
    TEXT_LEN = int(SIZE // 2 * 3)

    n_classes = len(predictions.keys())

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5 * SIZE / 64
    color = (0, 0, 0)
    thickness = 2

    result = np.full((SIZE * n_classes, SIZE * args.show_predictions + TEXT_LEN, 3), fill_value=1.0)

    for class_index, _class in enumerate(predictions):
        if len(predictions[_class]) < args.show_predictions:
            raise Exception("Not enough predictions from class", _class, 
                    "to satisfy --show-predictions", args.show_predictions)
    
        index = 0

        random.shuffle(predictions[_class])
        for index, fname in enumerate(predictions[_class]):
            if index >= args.show_predictions:
                break

            start_y = SIZE * class_index
            start_x = SIZE * index + TEXT_LEN
            img = cv2.imread(fname)
            img = resize_and_split(img, careful=False, max_aspect_ratio=3, min_new_info=999999, size=SIZE)[0]
            #print(result.shape)
            #print(img.shape)
            #print(result[start_y : start_y + SIZE, start_x : start_x + SIZE, :].shape)
            result[start_y : start_y + SIZE, start_x : start_x + SIZE, :] = img / 255
            
            start = (SIZE // 6, start_y + int(3 * SIZE / 4))
            result = cv2.putText(result, _class, start, font, scale, color, thickness, cv2.LINE_AA)


    
    return result
    

def show_test_images():
    session_dir = os.path.join('sessions', args.dataset_name, str(_id))
    test_fn = get_image_filenames(session_dir)

    preprocessing_dir = os.path.join(session_dir, 'preprocessing')

    prep = Preprocessor(test_fn, None, None, None, save_dir=preprocessing_dir, size=32, extract_one=True)
    images, targets = prep.process_set(test_fn)

    model, ema_model = get_wideresnet_models(len(prep.get_class_names()))
        
    ts = TrainState(model, ema_model, prep.get_class_names(), 0, 0, 0, 0, session_dir)
    ts._handle_resume(load_best=True)

    model.cpu()

    images = from_numpy(images)
    logits = model(images)

    _, predictions = logits.topk(1, 1)
    predictions = predictions.view(-1).numpy()
    
    all_fnames = []
    for _class in test_fn:
        all_fnames.extend(test_fn[_class])

    preds_by_cat = {}
    for fname, prediction in zip(all_fnames, predictions):
        prediction = prep.int_2_str(prediction)
        fnames = preds_by_cat.get(prediction, [])
        fnames.append(fname)
        preds_by_cat[prediction] = fnames

    img = get_image(preds_by_cat)
    cv2.imshow('Predictions', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if args.session_id == -1:
    max_id = -1
    sessions_dir = os.path.join('sessions', args.dataset_name)
    for fname in os.listdir(sessions_dir):
        try:
            _id = int(fname)

            if _id >= max_id:
                max_id = _id
        except ValueError:
            pass

    if max_id == -1:
        raise Exception("Could not find any sessions using the dataset " + \
            args.dataset_name + ".")
    _id = max_id
else:
    session_dir = os.path.join('sessions', args.dataset_name, str(args.session_id))
    if not os.path.exists(session_dir):
        raise Exception("Could not find session " + str(args.session_id) + \
            " using the dataset " + args.dataset_name + ".")
    _id = args.session_id

if args.show_predictions > 0:
    show_test_images()
    exit(0)

const_path = os.path.join('sessions', args.dataset_name, str(_id), 'const.pth.tar')
if not os.path.exists(const_path):
    raise Exception("Session exists but is not properly initialized: missing constants")
    
checkpoint_path = os.path.join('sessions', args.dataset_name, str(_id), 'training', 'checkpoint.pth.tar')
if not os.path.exists(checkpoint_path):
    raise Exception("Session exists but hasn't saved a checkpoint yet")


def print_list(list_name, short_list, verbose_list, dictionary):
    print(' ' + (list_name + '\'S LABEL').ljust(LABEL_LENGTH) + list_name + '\'S VALUE')
    print('-' * (LABEL_LENGTH * 2))

    def print_value(name, active):
        if not active:
            return

        value = dictionary[name]
        if not isinstance(value, float):
            str_value = str(value)
        else:
            str_value = '%.3f' % value

        line = ' ' + name.ljust(LABEL_LENGTH) + str_value
        print(line)

    for name in short_list:
        print_value(name, True)

    for name in verbose_list:
        print_value(name, args.verbose)

constants = load(const_path)
constants['session_id'] = _id
print_list('CONSTANT', short_consts, verbose_consts, constants)

variables = load(checkpoint_path)
print(''); print('')
print_list('VARIABLE', short_vars, verbose_vars, variables)

