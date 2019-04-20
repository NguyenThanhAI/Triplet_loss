from itertools import combinations

import numpy as np
import torch


def pdist(vectors):
    distance_matrix = -2*vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix

class TripletSelector:
    def __int__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

class AllTripletSelector(TripletSelector):

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))

            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives for neg_ind in negative_indices]

            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hardest_negative = np.argmax(loss_values)
    return hardest_negative if loss_values[hardest_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    def __init__(self, margin, negative_selector_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.margin = margin
        self.negative_selector_fn = negative_selector_fn
        self.cpu = cpu

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()

        distance_matrix = pdist(embeddings)
        if self.cpu:
            distance_matrix  = distance_matrix.cpu()

        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]

            if len(label_indices) < 2:
                continue

            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]

            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distances - distance_matrix[torch.LongTensor(anchor_positive[0]), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selector_fn(loss_values)

                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

            if len(triplets) == 0:
                triplets.append(anchor_positive[0], anchor_positive[1], negative_indices[0])

            triplets = np.array(triplets)

            return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin, negative_selector_fn=hardest_negative, cpu=cpu)


def RandomHardNegativeTripletSelector(margin, cpu):
    return FunctionNegativeTripletSelector(margin=margin, negative_selector_fn=random_hard_negative, cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu):
    return FunctionNegativeTripletSelector(margin=margin, negative_selector_fn=semihard_negative, cpu=cpu)