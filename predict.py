import os
from torch import load
from torch import from_numpy
import pickle
import cv2
import numpy as np

from dataset.custom_dataset import Preprocessor
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
    predict(tmp_path, session_dir)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


"""
sess_dir = 'sessions\\oregon_wildlife\\1'
img_path = 'data\\oregon_wildlife\\images\\nutria\\1cb31f27c4685c6e0d.jpg'
img = cv2.imread(img_path)
result = predict(img_path, sess_dir)
print(result)
#predict2(img, sess_dir)
"""