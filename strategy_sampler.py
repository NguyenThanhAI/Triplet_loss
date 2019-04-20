import random
import numpy as np
from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler(BatchSampler):
    '''
    BatchSampler: Samples n classes with m samples in these classes
    '''
    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.classes
        self.labels_set = list(set(self.labels))
        self.label_to_indices = dataset.class_to_idx

    #Shuffle instances in class

        for l in self.labels_set:
            random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_classes * self.n_samples

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])

                self.used_label_indices_count[class_] += self.n_samples

                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices

            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
