from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from util import BackgroundGenerator
import numpy as np


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            texts,
            labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label

    def __len__(self):
        count = len(self.images)
        return count


class SingleModalDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        count = len(self.data)
        return count


def get_loader(path, batch_size, INCOMPLETE=False):
    img_train = loadmat(path + "train_img.mat")['train_img']
    img_test = loadmat(path + "test_img.mat")['test_img']
    text_train = loadmat(path + "train_txt.mat")['train_txt']
    text_test = loadmat(path + "test_txt.mat")['test_txt']
    label_train = loadmat(path + "train_lab.mat")['train_lab']
    label_test = loadmat(path + "test_lab.mat")['test_lab']

    # Incomplete modal
    split = img_train.shape[0] // 5
    text_train[split * 1: split * 3] = np.zeros_like(text_train[split * 1: split * 3])
    img_train[split * 3: split * 5] = np.zeros_like(img_train[split * 3: split * 5])

    imgs = {'train': img_train, 'test': img_test}
    texts = {'train': text_train, 'test': text_test}
    labels = {'train': label_train, 'test': label_test}

    if INCOMPLETE:
        shuffle = {'train_complete': True, 'train_img': True, 'train_text': True, 'test': False}
        dataset = {'train_complete': CustomDataSet(images=imgs['train'][:split * 1],
                                                   texts=texts['train'][:split * 1],
                                                   labels=labels['train'][:split * 1]),
                   'train_img': SingleModalDataSet(data=imgs['train'][split * 1:split * 3],
                                                   labels=labels['train'][split * 1:split * 3]),
                   'train_text': SingleModalDataSet(data=texts['train'][split * 3: split * 5],
                                                    labels=labels['train'][split * 3: split * 5]),
                   'test': CustomDataSet(images=imgs['test'], texts=texts['test'], labels=labels['test'])}
        dataloader = {'train_complete': DataLoaderX(dataset['train_complete'], batch_size=batch_size // 5,
                                                    shuffle=shuffle['train_complete'], num_workers=0),
                      'train_img': DataLoaderX(dataset['train_img'], batch_size=batch_size // 5 * 2,
                                               shuffle=shuffle['train_img'], num_workers=0),
                      'train_text': DataLoaderX(dataset['train_text'], batch_size=batch_size // 5 * 2,
                                                shuffle=shuffle['train_text'], num_workers=0),
                      'test': DataLoaderX(dataset['test'], batch_size=batch_size,
                                          shuffle=shuffle['test'], num_workers=0),
                      }
    else:
        shuffle = {'train': True, 'test': False}
        dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
                   for x in ['train', 'test']}
        dataloader = {x: DataLoaderX(dataset[x], batch_size=batch_size,
                                     shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    img_dim = img_train.shape[1]
    text_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test
    input_data_par['img_train'] = img_train
    input_data_par['text_train'] = text_train
    input_data_par['label_train'] = label_train
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    return dataloader, input_data_par
