import os
import sys
import json
import pickle

import nltk
import tqdm
# from torchvision import transforms
# from PIL import Image
# from transforms import Scale

"""
Only keep questions with certain keywords
"""
keywords = ["left", "right"]


def process_question(root, split, word_dic=None, answer_dic=None):

    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    with open(os.path.join(root, 'questions',
                        f'CLEVR_{split}_questions.json')) as f:
        data = json.load(f)

    result = []
    word_index = 1
    answer_index = 0

    for question in tqdm.tqdm(data['questions']):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        contains_keywords = False
        for kw in keywords:
            if kw in words:
                contains_keywords = True
                break

        if not contains_keywords:
            continue

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question['answer']

        try:
            answer = answer_dic[answer_word]

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append((question['image_filename'], question_token, answer,
                    question['question_family_index']))

    if not os.path.exists('data/keywords_only'):
        os.makedirs('data/keywords_only')

    print("Number of exmaples in {} split with keywords {}:\t{}".format(split, keywords, len(result)))
    with open(f'data/keywords_only/{split}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python ... [CLEVR-dir]", file=sys.stderr)
        exit(1)

    root = sys.argv[1]

    word_dic, answer_dic = process_question(root, 'train')
    process_question(root, 'val', word_dic, answer_dic)

    with open('data/dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)