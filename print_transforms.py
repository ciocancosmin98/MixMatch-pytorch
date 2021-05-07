import sys
import argparse
import os
from torch import load
import pickle
import cv2
import numpy as np
import random
import sys
sys.path.append('dataset')

from transforms import load_transforms

from dataset.custom_dataset import Preprocessor, resize_and_split
import models.wideresnet as models
from session import TrainState

parser = argparse.ArgumentParser(description='Session information tool')

parser.add_argument('--dataset-name', '-n', default='animals10', type=str, metavar='NAME',
                    help='name of the dataset')
parser.add_argument('--session-id', '-i', default=-1, type=int, metavar='ID',
                    help='the id of the session whose information will be printed')
parser.add_argument('--verbose', '-v', action="store_true",
                    help='show more information')
parser.add_argument('--transforms', default='default.json', type=str)
parser.add_argument('--n-transformations', '-t', default=5, type=int, metavar='N_TR',
                    help='number of transformations per image')
parser.add_argument('--n-images', '-g', default=20, type=int, metavar='N_IMG',
                    help='number of images to transform')

args = parser.parse_args()

SIZE = 80

def get_image_filenames(session_dir):
    filename_list_path = os.path.join(session_dir, 'preprocessing', 'filenames_list.pkl')

    if not os.path.exists(filename_list_path):
        raise Exception("Session exists but the list of filenames does not exist")

    save_file = open(filename_list_path, "rb")
    _, _, val, test = pickle.load(save_file)

    return test

def get_transform_grid(images, preprocessor):
    random.seed(42)

    TEXT_LEN = int(SIZE // 2 * 3)

    n_rows = args.n_transformations + 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5 * SIZE / 64
    color = (0, 0, 0)
    thickness = 2

    train_transform, _ = load_transforms(args.transforms)
    result = np.full((SIZE * n_rows, SIZE * args.n_images + TEXT_LEN, 3), fill_value=1.0)

    for img_index in range(args.n_images):
        start_y = 0
        start_x = SIZE * img_index + TEXT_LEN

        img_prep = images[img_index]
        img_original = preprocessor.to_opencv(img_prep)

        result[start_y : start_y + SIZE, start_x : start_x + SIZE, :] = img_original

        for transform_index in range(1, args.n_transformations + 1):

            start_y = SIZE * transform_index
            start_x = SIZE * img_index + TEXT_LEN
            img_transf = train_transform(img_prep).numpy()
            img_transf = preprocessor.to_opencv(img_transf)

            result[start_y : start_y + SIZE, start_x : start_x + SIZE, :] = img_transf
    
    start = (SIZE // 6, int(3 * SIZE / 4))
    result = cv2.putText(result, "Originals", start, font, scale, color, thickness, cv2.LINE_AA)

    for transform_index in range(1, args.n_transformations + 1):
        start_y = SIZE * transform_index
        start = (SIZE // 6, start_y + int(3 * SIZE / 4))
        result = cv2.putText(result, str(transform_index), start, font, scale, color, thickness, cv2.LINE_AA)
    
    return result
    

def show_transforms():
    session_dir = os.path.join('sessions', args.dataset_name, str(_id))
    test_fn = get_image_filenames(session_dir)

    preprocessing_dir = os.path.join(session_dir, 'preprocessing')

    prep = Preprocessor(test_fn, None, None, None, save_dir=preprocessing_dir, size=32, extract_one=True)
    prep.size = SIZE
    images, _ = prep.process_set(test_fn)

    idxs = np.arange(images.shape[0])
    np.random.shuffle(idxs)
    images = images[idxs]

    img = get_transform_grid(images, prep)

    cv2.imshow('Transformations', img)
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

show_transforms()