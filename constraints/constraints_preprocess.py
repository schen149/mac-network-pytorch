import os
import sys
import json
import pickle as pkl
import constraints
from typing import List, Type, Dict
from transforms import Scale

import torch
import h5py
from PIL import Image
from tqdm import tqdm

from torchvision import transforms


transform = transforms.Compose([
    Scale([224, 224]),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_scene_into_dict(scene_file_path: str,
                          split: str = "train") -> Dict[str, Dict]:

    scene_dict = {}

    with open(scene_file_path) as fin:
        scene_data = json.load(fin)

    for scene in scene_data["scenes"]:
        _img_fname = "CLEVR_{}_{}.png".format(split, str(scene["image_index"]).zfill(6))
        scene_dict[_img_fname] = scene

    return scene_dict


def generate_constraints(constraints: List[Type[constraints.Constraint]],
                         scene_file_path: str,
                         dict_path: str,
                         question_path: str,
                         image_dir: str,
                         output_path: str) -> None:
    """
    generate loss masks for each constraints on the training set of CLEVR
    :param constraint: List of constraint classes (defined in constraints.py) that you want to apply in training
    :param output_path:
    :return:
    """

    scene_dict = parse_scene_into_dict(scene_file_path)

    with open(dict_path, 'rb') as fin:
        qa_dict = pkl.load(fin)

    with open(question_path, 'rb') as fin:
        questions = pkl.load(fin)

    tok_id_to_tok_dict = {}
    for tok, tok_id in qa_dict["word_dic"].items():
        tok_id_to_tok_dict[tok_id] = tok

    out_f = h5py.File(output_path, 'w', libver='latest')
    dset = out_f.create_dataset('constraints', (len(questions), len(CONSTRAINTS_TO_APPLY), 14, 14), dtype='f4')

    for idx, q in tqdm(enumerate(questions)):
        img_fname = q[0]
        question_toks = [tok_id_to_tok_dict[tok_id] for tok_id in q[1]]

        # scene = scene_dict[img_fname]
        #
        # img_path = os.path.join(image_dir, img_fname)
        # img = Image.open(img_path).convert('RGB')
        # img = transform(img) # Note that we don't do normalization here

        for c_idx, c_class in enumerate(constraints):
            c_mask = c_class.generate_constraints_masks(
                question_toks=question_toks,
                scene_data=None,
                img=None
            )

            dset[idx, c_idx] = c_mask

    out_f.close()


CONSTRAINTS_TO_APPLY = [
    constraints.RightConstraint,
    constraints.LeftConstraint,
]

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python constraints_preprocess.py [clevr-directory]", file=sys.stderr)
        exit(1)

    data_dir = "data/keywords_only"

    clevr_dir = sys.argv[1]
    output_path = os.path.join(data_dir, "train_constraints.hdf5")
    dict_path = os.path.join(data_dir, "dic.pkl")
    question_path = os.path.join(data_dir, "train.pkl")
    image_dir = os.path.join(clevr_dir, "images", "train")
    scene_file_path = os.path.join(clevr_dir, "scenes", "CLEVR_train_scenes.json")

    generate_constraints(constraints=CONSTRAINTS_TO_APPLY,
                         scene_file_path=scene_file_path,
                         dict_path=dict_path,
                         question_path=question_path,
                         image_dir=image_dir,
                         output_path=output_path)