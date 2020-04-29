import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py

from transforms import Scale

class CLEVR_w_constraints(Dataset):
    def __init__(self, root,
                 split='train',
                 data_dir="../data/",
                 transform=None):
        with open(os.path.join(data_dir, "keywords_only", '{}.pkl'.format(split)), 'rb') as f:
            self.data = pickle.load(f)

        # self.transform = transform
        self.root = root
        self.split = split

        if split=='train':
            _constraints = h5py.File(os.path.join(data_dir, "keywords_only", "train_constraints.hdf5"), 'r')
            self.constraints = _constraints["constraints"]
        else:
            self.constraints = None

        self.h = h5py.File(os.path.join(data_dir, '{}_features.hdf5'.format(split)), 'r')
        self.img = self.h['data']

    def close(self):
        self.h.close()

    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[index]
        # img = Image.open(os.path.join(self.root, 'images',
        #                            self.split, imgfile)).convert('RGB')

        # img = self.transform(img)
        id = int(imgfile.rsplit('_', 1)[1][:-4])
        img = torch.from_numpy(self.img[id])

        if self.split == "train":
            c_mask = torch.from_numpy(self.constraints[index])
        else:
            c_mask = None

        return img, question, len(question), answer, family, c_mask

    def __len__(self):
        return len(self.data)

transform = transforms.Compose([
    Scale([224, 224]),
    transforms.Pad(4),
    transforms.RandomCrop([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])

def collate_data(batch):
    images, lengths, answers, families, c_masks = [], [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, family, c_mask = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        families.append(family)
        c_masks.append(c_mask)

    return torch.stack(images), torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers), families, torch.stack(c_masks)


def collate_data_validation(batch):
    images, lengths, answers, families = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, family, _ = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        families.append(family)

    return torch.stack(images), torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers), families
