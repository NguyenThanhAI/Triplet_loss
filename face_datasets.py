import os
import numpy as np
import skimage
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils

data_transfrom = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.40],
                                                          std=[0.229, 0.224, 0.225])])

face_dataset = datasets.ImageFolder(root='laplacian',
                                    transform=data_transfrom)

data_loader = DataLoader(dataset=face_dataset, batch_size=12, shuffle=True,
                         num_workers=4)
print(type(face_dataset.class_to_idx))

class  TripletFaceDataset(Dataset):

    def __init__(self, face_dataset, is_train=True):
        self.face_dataset = face_dataset
        self.transform = face_dataset.transform
        self.is_train = is_train

        if self.is_train:
            self.train_labels = face_dataset.classes
            self.train_data = face_dataset.imgs
            self.labels_set = list(set(self.train_labels))
            self.label_to_indices = face_dataset.class_to_idx
        else:
            self.test_labels = face_dataset.classes
            self.test_data = face_dataset.images
            self.labels_set = list(set(self.test_labels))
            self.label_to_indices = face_dataset.class_to_idx

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]

            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.is_train:

            img1, label1 = self.train_data[index], self.train_labels[index].item()

            positive_index = index

            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])

            negative_label = list(self.labels_set - set(label1))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]

        else:

            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3


    def __len__(self):
        return len(self.face_dataset)