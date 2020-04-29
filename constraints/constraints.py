from abc import abstractmethod
from typing import List, Dict
from image_feature import CLEVR

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Constraint:
    """
    Base Class for constraints
    """
    @classmethod
    @abstractmethod
    def generate_constraints_masks(cls,
                                   question_toks: List[str],
                                   scene_data: Dict,
                                   img: torch.tensor) -> torch.tensor:
        raise NotImplementedError


def constraint_loss_fn_calc(attn_output: torch.tensor,
                            constraint_masks: torch.tensor,
                            alpha=0.1):
    """
    TODO: maybe change this if we have time to train an upsampled attention to full image
    :param attn_output: Attention distribution over KB, of size batch_size x 14 x 14
    :param constraint_masks: of size bs x K x 14 x 14, penalty for
    :param alpha:
    :return:
    """
    attn_output = attn_output.unsqueeze(1).view(-1, 1, 14, 14)
    attn_w_loss_mask = torch.matmul(constraint_masks, attn_output)
    return torch.sum(attn_w_loss_mask)


class RightConstraint(Constraint):
    @classmethod
    def generate_constraints_masks(cls,
                                   question_toks: List[str],
                                   scene_data: Dict,
                                   img: torch.tensor  # Orig img after transformation - 3 x 224 x 224
                                   ) -> torch.tensor:

        _mask = np.zeros((14, 14))

        if 'right' in question_toks:
            _mask[:, 0:7] = 1  # Assign non-zero loss to the left side of the img

        return _mask


class LeftConstraint(Constraint):
    @classmethod
    def generate_constraints_masks(cls,
                                   question_toks: List[str],
                                   scene_data: Dict,
                                   img: torch.tensor) -> torch.tensor:

        _mask = np.zeros((14, 14))

        if 'left' in question_toks:
            _mask[:, 7:14] = 1  # Assign non-zero loss to the left side of the img

        return _mask


if __name__ == '__main__':
    import sys
    import pickle as pkl
    processed_clevr = sys.argv[1]

    # clevr = CLEVR(sys.argv[1], 'val')
    # print(clevr[0])
    # print(clevr[0].size())
    # # dataloader = DataLoader(CLEVR(sys.argv[1], 'val'), batch_size=4, num_workers=4)
    #
    # print(len(clevr))

    with open('data/train.pkl', 'rb') as fin:
        question = pkl.load(fin)

    with open('data/dic.pkl', 'rb') as fin:
        dic = pkl.load(fin)
    print(dic['word_dic'])
    print(dic['answer_dic'])
    print(len(question))
    print(question[0])