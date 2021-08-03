import torch
from torch.utils.data import Dataset
import h5py
import json
import os

class CaptionDataset(Dataset):
    def __init__(self, data_folder, data_name, split, transform=None):      
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']
        self.cpi = self.h.attrs['captions_per_image']
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
        with open(os.path.join(data_folder, self.split + '_IMPATHS_' + data_name + '.json'), 'r') as j:
            self.image_paths = json.load(j)
        self.transform = transform
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)

        if self.transform is not None:
            img = self.transform(img)

        image_path = self.image_paths[i // self.cpi]
        image_name = image_path.split('/')[-1]

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen, image_name
        else:
            all_captions = torch.LongTensor(
                    self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions, image_name 

    def __len__(self):
        return self.dataset_size
