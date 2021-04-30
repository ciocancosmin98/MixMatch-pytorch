from dataset.cifar10 import TransformTwice, transpose
import os
import cv2
import torch
import numpy as np
import torchvision
import random
import sys

from torch.utils.data import Dataset, DataLoader

def get_custom(root, labeled_fnames, unlabeled_fnames, transform_train=None, 
                transform_val=None):
    
    prep = Preprocessor(labeled_fnames, unlabeled_fnames, root=root, overwrite=False, size=32)
    _, tgts_lab, imgs_unl = prep.load_all()

    train_idxs, val_idxs, test_idxs = train_val_test_split(tgts_lab)
    unl_idxs = unlabeled_idxs(imgs_unl.shape[0])

    train_labeled_dataset   = CustomDataset(prep,  labeled=True, indexs=train_idxs,
                                            transform=transform_train)
    val_dataset             = CustomDataset(prep,  labeled=True, indexs=val_idxs,
                                            transform=transform_val)
    test_dataset            = CustomDataset(prep,  labeled=True, indexs=test_idxs,
                                            transform=transform_val)
    train_unlabeled_dataset = CustomDataset(prep, labeled=False, indexs=unl_idxs,
                                            transform=TransformTwice(transform_train))

    print (f"#Labeled: {len(train_idxs)} #Unlabeled: {len(unl_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, prep.get_class_names()
    

def train_val_test_split(labels, ratios=None, seed=42):
    train_idxs  = []
    val_idxs    = []
    test_idxs   = []
    
    if ratios is None:
        ratios = {'train': 0.20, 'val': 0.40, 'test': 0.40}
        
    assert ratios['train'] > 0 and ratios['val'] > 0 and ratios['train'] + ratios['val'] < 1.0
    
    np.random.seed(seed)
    
    indexs = {}
    n_classes = np.max(labels) + 1
    min_count = sys.maxsize
    
    for cls in range(n_classes):
        idxs = np.where(labels == cls)[0]
        
        np.random.shuffle(idxs)
        indexs[cls] = idxs
        min_count = min(min_count, len(idxs))
    
    train_start = 0
    train_end   = int(ratios['train'] * min_count)
    val_start   = train_end
    val_end     = val_start + int(ratios['val'] * min_count + 0.5)
    test_start  = val_end
    test_end    = min_count
    
    for cls in range(n_classes):
        count = len(indexs[cls])
        print(f'For class {cls} we are only using {min_count}/{count} images.')
        
        train_idxs.extend(indexs[cls][train_start:train_end])
        val_idxs.extend(indexs[cls][val_start:val_end])
        test_idxs.extend(indexs[cls][test_start:test_end])        
        
    np.random.shuffle(train_idxs)
    np.random.shuffle(test_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs, test_idxs

def unlabeled_idxs(n_unlabeled, seed=42):
    np.random.seed(seed)
    unl_idxs = np.arange(n_unlabeled)
    np.random.shuffle(unl_idxs)
    return unl_idxs

class CustomDataset(Dataset):
    def __init__(self, preprocessor, transform=None, target_transform=None,
                 labeled=True, indexs=None):
            
        self.transform = transform
        self.target_transform = target_transform
        
        imgs_lab, tgts_lab, imgs_unl = preprocessor.load_all()
        if labeled:
            self.data    = imgs_lab
            self.targets = tgts_lab
        else:
            self.data    = imgs_unl
            self.targets = np.full(imgs_unl.shape[0], -1, dtype=int)
        
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = self.targets[indexs]
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class Preprocessor:
    def __init__(self, labeled_fnames, unlabeled_fnames, root='.', size=32,
                    min_new_info=0.4, overwrite=False):
        
        assert (min_new_info >= 0 and min_new_info <= 1)
        assert (os.path.exists(root))
        
        # define label string to integer conversion
        classes = list(labeled_fnames.keys())
        self.classes = classes

        #print('GOT CLASSES:', classes)
        #for cls in classes:
        #    print(f'{cls}\t: {len(labeled_fnames[cls])}')

        self._int_2_str = dict(enumerate(classes))
        self._str_2_int = dict([(a, b) for b, a in enumerate(classes)])
        
        self.labeled_fnames   = labeled_fnames
        self.unlabeled_fnames = unlabeled_fnames
        
        self.root = root
        self.size = size
        self.min_new_info = min_new_info
        self.overwrite = overwrite 
        
        self._preprocess()
        
    def _preprocess(self):
        self.savepath = os.path.join(self.root, "preproc.npz")
        if os.path.exists(self.savepath) and not self.overwrite:
            print('Skipping preprocessing... (found saved file)')
            npzfile = np.load(self.savepath)
            self.mean = npzfile['mean']
            self.std  = npzfile['std']
            return
        else:
            print('Preprocessing started...')
        
        imgs_unl = []
        imgs_all = []
        for fname in self.unlabeled_fnames:
            img = cv2.imread(fname)
            imgs = resize_and_split(img, min_new_info=self.min_new_info,
                                size=self.size)
            imgs_unl.extend(imgs)
            imgs_all.extend(imgs)
            
        
        imgs_lab = []
        tgts_lab = []
        for cls in self.labeled_fnames:
            int_label = self.str_2_int(cls)
            imgs_cls = []
            for fname in self.labeled_fnames[cls]:
                img = cv2.imread(fname)
                imgs = resize_and_split(img, min_new_info=self.min_new_info,
                                size=self.size)
                imgs_cls.extend(imgs)
            tgts_lab.extend([int_label] * len(imgs_cls))
            imgs_lab.extend(imgs_cls)
            imgs_all.extend(imgs)
        
        imgs_all = np.array(imgs_all)
        self.mean = np.mean(imgs_all, axis=(0, 1, 2))
        self.std  = np.std(imgs_all, axis=(0, 1, 2))
        
        imgs_unl = np.array(imgs_unl)
        imgs_lab = np.array(imgs_lab)
        tgts_lab = np.array(tgts_lab)
        
        imgs_unl = self.standardize(imgs_unl)
        imgs_lab = self.standardize(imgs_lab)
        
        imgs_lab = transpose(imgs_lab)
        imgs_unl = transpose(imgs_unl)
        
        np.savez(self.savepath, imgs_lab=imgs_lab, tgts_lab=tgts_lab,
                 imgs_unl=imgs_unl, mean=self.mean, std=self.std)
        
    def standardize(self, images):
        images, mean, std = [np.array(a, np.float32) for a in
                             (images, self.mean, self.std)]
        images -= mean
        images *= 1.0/std
        return images
    
    def reverse_standardization(self, images):
        images, mean, std = [np.array(a, np.float32) for a in
                             (images, self.mean, self.std)]
        images *= std
        images += mean
        return images
    
    def to_opencv(self, image):
        image = transpose(image, source='CHW', target='HWC')
        image = self.reverse_standardization(image)
        image /= 255
        return image
        
    def load_all(self):
        npzfile = np.load(self.savepath)
        return (npzfile['imgs_lab'], npzfile['tgts_lab'],
                npzfile['imgs_unl'])
        
    def preprocess(self, img):
        return resize_and_split(img, min_new_info=self.min_new_info,
                                size=self.size)
        
    def get_class_names(self):
        return self.classes

    def int_2_str(self, int_label):
        return self._int_2_str[int_label]
    
    def str_2_int(self, str_label):
        return self._str_2_int[str_label]


def resize_and_split(img, careful=True, max_aspect_ratio=3, min_new_info=0.2, size=128):
    height, width = img.shape[0], img.shape[1]
    min_dim = min(height, width)
    scaling = min_dim / size
    new_height = int(height / scaling)
    new_width = int(width / scaling)
    img_1 = cv2.resize(img, (new_width, new_height))
    
    max_dim = max(new_height, new_width)
    ratio = max_dim / size
    if ratio > max_aspect_ratio and careful:
        # The aspect ratio of the image is too big; we cannot extract
        # a square portion of this image and have it still represent
        # the class as a whole
        return []
    
    def add_img(result_imgs, start_x, start_y):
        img_tmp = img_1[start_y:start_y+size, start_x:start_x+size]
        result_imgs.append(img_tmp)
    
    result_imgs = []
    if ratio > (1 + min_new_info):
        add_img(result_imgs, 0, 0)
        add_img(result_imgs, new_width - size, new_height - size)
    else:
        start_x = np.random.randint(0, new_width  - size + 1)
        start_y = np.random.randint(0, new_height - size + 1)
        add_img(result_imgs, start_x, start_y)
    
    return result_imgs