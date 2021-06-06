import os
from torch import load
from torch import from_numpy
import pickle
import cv2
import numpy as np
import random

from dataset.custom_dataset import Preprocessor, resize
import models.wideresnet as models
from session import TrainState

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def top_k(values, k=3):
    indexed = [(value, index) for index, value in enumerate(values)]
    indexed.sort(key=lambda x : x[0], reverse=True)
    return indexed[:k]

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

def predict(image_path, session_dir):
    const_path = os.path.join(session_dir, 'const.pth.tar')
    if not os.path.exists(const_path):
        raise Exception("Session exists but is not properly initialized: missing constants")
    constants = load(const_path)

    test_fn = get_image_filenames(session_dir)

    preprocessing_dir = os.path.join(session_dir, 'preprocessing')
    prep = Preprocessor(test_fn, None, None, None, save_dir=preprocessing_dir, size=32, extract_one=True)
    predict_fn = {list(test_fn.keys())[0] : [image_path]}

    images, _ = prep.process_set(predict_fn)
    model, ema_model = get_wideresnet_models(len(prep.get_class_names()))
        
    ts = TrainState(model, ema_model, prep.get_class_names(), constants, session_dir)
    ts._handle_resume(load_best=True)

    model.cpu()
    model.eval()

    images = from_numpy(images)
    logits = model(images)

    _, predictions = logits.topk(1, 1)
    predictions = predictions.view(-1).numpy()

    probs = softmax(logits.detach().numpy()[0])
    top_preds = top_k(probs)
    result = []
    for prob, _class in top_preds:
        prob = "%.3f%%" % (prob * 100)
        _class = prep.int_2_str(_class)
        result.append((_class, prob))
    
    return result

def predict2(image, session_dir):
    """
    DELETE ME AND DO ME RIGHT, YOU SCOUNDREL
    """

    tmp_path = 'temporary_file0123456789.png'
    cv2.imwrite(tmp_path, image)
    result = predict(tmp_path, session_dir)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return result

def get_image(predictions, n_images_per_class=10):
    random.seed(42)

    SIZE = 64
    TEXT_LEN = int(SIZE // 2 * 3)

    n_classes = len(predictions.keys())

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5 * SIZE / 64
    color = (0, 0, 0)
    thickness = 2

    result = np.full((SIZE * n_classes, SIZE * n_images_per_class + TEXT_LEN, 3), fill_value=1.0)

    for class_index, _class in enumerate(predictions):
        index = 0
        random.shuffle(predictions[_class])

        if len(predictions[_class]) < n_images_per_class:
            predictions[_class].extend([None] * n_images_per_class)

        for index, fname in enumerate(predictions[_class]):
            if index >= n_images_per_class:
                break

            start_y = SIZE * class_index
            start_x = SIZE * index + TEXT_LEN

            if fname is None:
                img = np.ones((SIZE, SIZE, 3)) * 255
            else:
                img = cv2.imread(fname)
                img = resize(img, size=SIZE)
            
            result[start_y : start_y + SIZE, start_x : start_x + SIZE, :] = img / 255
            
            start = (SIZE // 6, start_y + int(3 * SIZE / 4))
            result = cv2.putText(result, _class, start, font, scale, color, thickness, cv2.LINE_AA)
    
    return result

def show_test_images(session_dir, model):
    test_fn = get_image_filenames(session_dir)

    preprocessing_dir = os.path.join(session_dir, 'preprocessing')
    prep = Preprocessor(test_fn, None, None, None, save_dir=preprocessing_dir, size=32, extract_one=True)
    images, _ = prep.process_set(test_fn)

    is_cuda = next(model.parameters()).is_cuda
    model.cpu()
    model.eval()

    images = from_numpy(images)
    logits = model(images)

    if is_cuda:
        model.cuda()

    _, predictions = logits.topk(1, 1)
    predictions = predictions.view(-1).numpy()
    
    all_fnames = []
    for _class in test_fn:
        all_fnames.extend(test_fn[_class])

    class_names = prep.get_class_names()
    preds_by_cat = {_class : [] for _class in class_names}
    for fname, prediction in zip(all_fnames, predictions):
        prediction = prep.int_2_str(prediction)
        preds_by_cat[prediction].append(fname)

    img = get_image(preds_by_cat)

    images_dir = os.path.join(session_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    cv2.imwrite(os.path.join(images_dir, 'grid.jpg'), img * 255)

"""
sess_dir = 'sessions\\oregon_wildlife\\1'
img_path = 'data\\oregon_wildlife\\images\\nutria\\1cb31f27c4685c6e0d.jpg'
img = cv2.imread(img_path)
result = predict(img_path, sess_dir)
print(result)
#predict2(img, sess_dir)
"""