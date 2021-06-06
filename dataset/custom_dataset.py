from dataset.transforms import TransformTwice, transpose, load_transforms
import os
import cv2
import torch
import numpy as np
import torchvision
import random
import sys
import dataset.cifar10 as dataset
import torchvision.transforms as transforms
import torch.utils.data as data
import time

from torch.utils.data import Dataset, DataLoader

def load_custom(preprocessor, batch_size, transforms_name):
    print(f'==> Preparing the custom dataset')

    transform_train, transform_val = load_transforms(transforms_name)

    train_labeled_set, train_unlabeled_set, val_set, test_set = get_custom(
        preprocessor,
        transform_train=transform_train,
        transform_val=transform_val)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return labeled_trainloader, unlabeled_trainloader, val_loader, test_loader
    
def get_custom(preprocessor, transform_train=None, transform_val=None):
    train_labeled_dataset   = CustomDataset(preprocessor,  labeled=True, name='labeled', transform=transform_train)
    val_dataset             = CustomDataset(preprocessor,  labeled=True,     name='val', transform=transform_val)
    test_dataset            = CustomDataset(preprocessor,  labeled=True,    name='test', transform=transform_val)
    train_unlabeled_dataset = CustomDataset(preprocessor, labeled=False, transform=TransformTwice(transform_train))

    print (f"#Labeled: {len(train_labeled_dataset)} #Unlabeled: {len(train_unlabeled_dataset)} #Val: {len(val_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset

def shuffle_data(data, targets, seed=42):
    np.random.seed(seed)

    n_datapoints = data.shape[0]
    idxs = np.arange(n_datapoints)
    np.random.shuffle(idxs)
    
    return data[idxs], targets[idxs]

class CustomDataset(Dataset):
    def __init__(self, preprocessor, transform=None, target_transform=None,
                 labeled=True, name=None):
            
        self.transform = transform
        self.target_transform = target_transform
        
        if labeled:
            if name is None:
                raise TypeError('No name provided to identify the labeled set.')

            img_name = name + "_images"
            tgt_name = name + "_targets"

            self.data    = preprocessor.retrieve(img_name)
            self.targets = preprocessor.retrieve(tgt_name)
            self.data, self.targets = shuffle_data(self.data, self.targets)
        else:
            self.data    = preprocessor.retrieve('unlabeled_images')
            self.targets = np.full(self.data.shape[0], -1, dtype=int)
            self.data, self.targets = shuffle_data(self.data, self.targets)
        
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
    def __init__(self, labeled_fn, unlabeled_fn, val_fn, test_fn, save_dir='.', size=32,
                    min_new_info=0.4, overwrite=False, extract_one=False):
        
        assert (min_new_info >= 0 and min_new_info <= 1)
        assert (os.path.exists(save_dir))
        
        # define label string to integer conversion
        classes = list(labeled_fn.keys())
        self.classes = classes

        self._int_2_str = dict(enumerate(classes))
        self._str_2_int = dict([(a, b) for b, a in enumerate(classes)])
        
        self.labeled_fn   = labeled_fn
        self.unlabeled_fn = unlabeled_fn
        self.val_fn       = val_fn
        self.test_fn      = test_fn
        
        self.save_dir = save_dir
        self.size = size
        self.min_new_info = min_new_info
        self.overwrite = overwrite 
        self.npzfile = None

        self.extract_one = extract_one
        
        self._preprocess()
        
    def _read_and_resize_set(self, fnames, all_images):
        imgs_set = []
        tgts = []

        for cls in fnames:
            int_label = self.str_2_int(cls)
            imgs_cls = []
            for fname in fnames[cls]:
                try:
                    img = cv2.imread(fname)
                except:
                    print('EXCEPTION READING')
                    continue
                
                if img is None:
                    print('IMG IS NONE')
                    continue

                img = resize(img, size=self.size)
                imgs_cls.append(img)
            tgts.extend([int_label] * len(imgs_cls))
            imgs_set.extend(imgs_cls)
            all_images.extend(imgs_cls)

        return np.array(imgs_set), np.array(tgts)

    def process_set(self, fnames):
        imgs, tgts = self._read_and_resize_set(fnames, [])

        imgs = self.standardize(imgs)
        imgs = transpose(imgs)

        return imgs, tgts
        
        
    def _preprocess(self):
        self.savepath = os.path.join(self.save_dir, "preproc.npz")
        if os.path.exists(self.savepath) and not self.overwrite:
            print('Skipping preprocessing... (found saved file)')
            npzfile = np.load(self.savepath)
            self.mean = npzfile['mean']
            self.std  = npzfile['std']
            return
        else:
            print('Preprocessing started...')
        
        unlabeled_images = []
        all_images = []
        for fname in self.unlabeled_fn:
            try:
                img = cv2.imread(fname)
            except:
                print('EXCEPTION READING')
                continue
                
            if img is None:
                print('IMG IS NONE')
                continue

            img = resize(img, self.size)
            unlabeled_images.append(img)
            all_images.append(img)
            
        
        labeled_images, labeled_targets = self._read_and_resize_set(self.labeled_fn, all_images)
        val_images, val_targets         = self._read_and_resize_set(self.val_fn, all_images)
        test_images, test_targets       = self._read_and_resize_set(self.test_fn, all_images)
        
        all_images = np.array(all_images)
        self.mean = np.mean(all_images, axis=(0, 1, 2))
        self.std  = np.std(all_images, axis=(0, 1, 2))
        
        unlabeled_images = np.array(unlabeled_images)
        
        unlabeled_images = self.standardize(unlabeled_images)
        labeled_images   = self.standardize(labeled_images)
        val_images       = self.standardize(val_images)
        test_images      = self.standardize(test_images)
        
        unlabeled_images = transpose(unlabeled_images)
        labeled_images   = transpose(labeled_images)
        val_images       = transpose(val_images)
        test_images      = transpose(test_images)
        
        np.savez(self.savepath, 
                unlabeled_images = unlabeled_images, 
                labeled_images   = labeled_images,
                val_images       = val_images,
                test_images      = test_images,
                labeled_targets  = labeled_targets,
                val_targets      = val_targets,
                test_targets     = test_targets,
                mean=self.mean,
                std=self.std)
        
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
        
    def retrieve(self, name):
        if self.npzfile is None:
            self.npzfile = np.load(self.savepath)
        
        return self.npzfile[name]
        
    def preprocess(self, img):
        return resize(img, self.size)
        
    def get_class_names(self):
        return self.classes

    def int_2_str(self, int_label):
        return self._int_2_str[int_label]
    
    def str_2_int(self, str_label):
        return self._str_2_int[str_label]

def resize(img, size=128):
    height, width = img.shape[0], img.shape[1]
    min_dim = min(height, width)
    scaling = min_dim / size
    new_height = int(height / scaling)
    new_width = int(width / scaling)
    img = cv2.resize(img, (new_width, new_height))
    
    y_start = new_height // 2 - size // 2
    x_start = new_width // 2 - size // 2
    return img[y_start : y_start + size, x_start : x_start + size, :]

def get_filenames(dataset_path):
    import random

    dataset_path = os.path.join(dataset_path, 'images')

    filenames = {}
    classes = os.listdir(dataset_path)

    assert (len(classes) > 0)

    for cls in classes:
        filenames[cls] = []
        
        path_to_class = os.path.join(dataset_path, cls)
        class_fnames = os.listdir(path_to_class)

        assert (isinstance(class_fnames, list) and len(class_fnames) > 0)

        for fname in class_fnames:
            filenames[cls].append(os.path.join(path_to_class, fname))

        random.seed(42)
        random.shuffle(filenames[cls])

    return filenames, classes

def get_filenames_train_validate_test(dataset_name, session_path, n_labeled=1000, balance_unlabeled=False, n_test_per_class=100):
    import pickle, os

    dataset_path = os.path.join('data', dataset_name)
    save_path = os.path.join(session_path, 'filenames_list.pkl')

    if os.path.exists(save_path):
        print('==> Loading list of labeled and unlabeled data from memory')
        save_file = open(save_path, "rb")
        labeled, unlabeled, val, test = pickle.load(save_file)
        return labeled, unlabeled, val, test

    print('==> Calculating list of labeled and unlabeled data')

    #fnames_128, classes = get_fnames128(dataset_path)
    fnames, classes = get_filenames(dataset_path)

    n_classes = len(classes)
    n_labeled_per_class = int(n_labeled / n_classes)

    seed = 42
    random.seed(seed)

    unlabeled = {}
    labeled   = {}
    val       = {}
    test      = {}

    val_start     = 0
    val_end       = val_start + n_test_per_class
    test_start    = val_end
    test_end      = test_start + n_test_per_class
    labeled_start = test_end
    labeled_end   = labeled_start + n_labeled_per_class

    min_unl = 99999999

    for cls in fnames:        
        val[cls]       = fnames[cls][      val_start :     val_end]
        test[cls]      = fnames[cls][     test_start :    test_end]
        labeled[cls]   = fnames[cls][  labeled_start : labeled_end]
        unlabeled[cls] = fnames[cls][    labeled_end :]

        min_unl = min(min_unl, len(unlabeled[cls]))

    unlabeled_final = []
    if balance_unlabeled:
        for cls in unlabeled:
            unlabeled_final.extend(unlabeled[cls][:min_unl])
    else:
        for cls in unlabeled:
            unlabeled_final.extend(unlabeled[cls])

    unlabeled = unlabeled_final

    data = (labeled, unlabeled, val, test)
    save_file = open(save_path, "wb")
    pickle.dump(data, save_file)
    save_file.close()

    return labeled, unlabeled, val, test

class ImageQueueReader:
    def __init__(self, queue, base_path, categories, session_dir, min_train_per_class=50, train_perc=0.5):
        self.queue = queue
        self.categories = categories
        self.base_path = base_path
        self.train_perc = train_perc
        self.session_dir = session_dir

        # when all classes have at least min_per_class labeled images, the tick
        # function will return True, meaning you can do the train test val
        # split and continue with the training;
        self.min_train_per_class = min_train_per_class
        self.class_counts = [0] * len(categories)

        self.str_2_int = {cat : index for index, cat in enumerate(categories)}

        self.unlabeled_fn = set()

        self.all_labeled_fn = {cat : [] for cat in categories}

    def _read_queue(self):
        while len(self.queue) > 0:
            img_info = self.queue.pop(0)

            file_id   = img_info['id_file']
            folder_id = img_info['id_folder']
            category  = img_info['category']

            img_path = os.path.join(self.base_path, folder_id, file_id)

            if category == 'unclassified':
                self.unlabeled_fn.add(img_path)
            else:
                if img_path in self.unlabeled_fn:
                    self.unlabeled_fn.remove(img_path)
                
                index = self.str_2_int[category]
                self.all_labeled_fn[category].append(img_path)
                self.class_counts[index] += 1

    def tick(self):
        self._read_queue()

        if int(self.train_perc * min(self.class_counts)) < self.min_train_per_class:
            time.sleep(30)
            return False
        
        return True

    def split(self):
        import pickle

        labeled_fn = {cat : [] for cat in self.categories}
        test_fn = {cat : [] for cat in self.categories}
        val_fn = {cat : [] for cat in self.categories}

        min_count = min(self.class_counts)

        test_perc = (1.0 - self.train_perc) / 2
        n_test_per_class = min(int(min_count * test_perc), 50)

        val_start     = 0
        val_end       = val_start + n_test_per_class
        test_start    = val_end
        test_end      = test_start + n_test_per_class
        labeled_start = test_end
        labeled_end   = min_count

        for cls in self.all_labeled_fn:        
            val_fn[cls]     = self.all_labeled_fn[cls][      val_start :     val_end]
            test_fn[cls]    = self.all_labeled_fn[cls][     test_start :    test_end]
            labeled_fn[cls] = self.all_labeled_fn[cls][  labeled_start : labeled_end]

        unlabeled_fn = list(self.unlabeled_fn)

        save_path = os.path.join(self.session_dir, 'preprocessing', 'filenames_list.pkl')
        data = (labeled_fn, unlabeled_fn, val_fn, test_fn)
        save_file = open(save_path, "wb")
        pickle.dump(data, save_file)
        save_file.close()

        return labeled_fn, unlabeled_fn, val_fn, test_fn
