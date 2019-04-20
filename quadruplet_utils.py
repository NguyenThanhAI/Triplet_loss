from itertools import combinations

import numpy as np
import torch

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class QuadrupletSelector:
    def __init__(self):
        pass

    def get_quadruplets(self, embeddings, labels):
        raise NotImplementedError

class AllQuadrupletSelector(QuadrupletSelector):
    def __init__(self):
        super(AllQuadrupletSelector, self).__init__()

    def get_quadruplets(self, embeddings, labels):
        quadrups = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]

            if len(label_indices) < 2:
                continue

            anchor_positives = list(combinations(label_indices, 2))

            for neg1_label in set(labels) - label:
                neg1_mask = (labels == neg1_label)

                neg1_indices = np.where(np.logical_not(label_mask))

                neg2_indices = np.where(np.logical_not(np.logical_or(label_mask, neg1_mask)))[0]

                temp_quadrups = [[anchor_positive[0], anchor_positive[1], neg1_ind, neg2_ind] for anchor_positive in anchor_positives
                                 for neg1_ind in neg1_indices for neg2_ind in neg2_indices]


                quadrups += temp_quadrups

        return torch.LongTensor(np.array(quadrups))


def random_hard_negative(loss_1, loss_2):
    '''
    :param loss_1: loss between anchor, positive, negative 1
    :param loss_2: loss between anchor, positive, negative 1, negative 2
    :return: index of quadrup
    '''

    hard_negatives = np.where(np.logical_and(loss_1 > 0, loss_2 > 0))[0]

    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def hardest_negative(loss_1, loss_2):
    #loss_1[np.logical_or(loss_1 < 0, loss_2 < 0)] = 0
    #loss_2[np.logical_or(loss_1 < 0, loss_2 < 0)] = 0

    loss = loss_1 + loss_2

    loss[np.logical_or(loss_1 < 0, loss_2 < 0)] = 0

    hardest_negative = np.argmax(loss)

    return hardest_negative if loss[hardest_negative] > 0 else None


def semi_hard_negative(loss_1, loss_2, margin_1, margin_2):
    semi_negatives = np.where(np.logical_and(loss_1 > 0, loss_1 < margin_1, loss_2 > 0, loss_2 < margin_2))[0]
    return np.random.choice(semi_negatives) if len(semi_negatives) > 0 else None


class FunctionHardNegtiveQuadrupSelector(QuadrupletSelector):
    def __init__(self, margin1, margin2, negative_selector_fn, cpu=True):
        super(FunctionHardNegtiveQuadrupSelector, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.negative_selector_fn = negative_selector_fn
        self.cpu = cpu

    def get_quadrups(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()

        distance_matrix = pdist(embeddings)

        if self.cpu:
            distance_matrix = distance_matrix.cpu()

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]

            if len(label_indices) < 2:
                continue

            anchor_positives = list(combinations(label_indices, 2))

            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]

            for neg1_label in set(labels) - labels:
                neg1_mask = (labels == neg1_label)
                neg1_indices = np.where(neg1_mask)[0]
                neg2_indices = np.where(np.logical_or(label_mask, neg1_mask))[0]
                for ap_distance, anchor_positive in zip(ap_distances, anchor_positives):
                    loss_1 = ap_distance - distance_matrix(anchor_positive[0], neg1_indices) + self.margin1
                    loss_1 = loss_1.data.cpu().numpy()

                    for neg1_ind in neg1_indices:
                        loss_2 = ap_distance - distance_matrix(neg1_ind, neg2_indices)  + self.margin2
                        loss_2 = loss_2.data.cpu().numpy()

                        hard_negative = self.negative_selector_fn(loss_1, loss_2)
