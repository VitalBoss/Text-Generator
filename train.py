import numpy as np
import pickle as pkl
from collections import Counter
import argparse
from os import listdir
from os.path import isfile, join
import re


def generate_from_dict(source_dict):
    total_sum = sum(source_dict.values())
    values = []
    probabilities = []
    for val, count in source_dict.items():
        values.append(val)
        probabilities.append(count / total_sum)
    return np.random.choice(values, p=probabilities)


def read_from_dir(directory):
    only_files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    res = ''
    for file_path in only_files:
        f = open(file_path, 'r')
        res += f.read()
        f.close()
    return res


class Model:

    def __init__(self):
        self.dictionary = dict()
        self.prior = None

    def fit(self, data):
        self.prior = Counter(data)
        for i in range(len(data) - 1):
            d = self.dictionary.setdefault(data[i], dict())
            if d.get(data[i + 1]) is None:
                d[data[i + 1]] = 1
            else:
                d[data[i + 1]] += 1

    def generate(self, previous):
        if self.dictionary.get(previous) is None:
            return generate_from_dict(self.prior)
        else:
            return generate_from_dict(self.dictionary.get(previous))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir')
    parser.add_argument('--model')
    args = parser.parse_args()

    if args.input_dir is None:
        text = input()
    else:
        text = read_from_dir(args.input_dir)
    model = Model()
    model.fit(re.sub('[^a-z0-9 ]', ' ', text.lower(), count=0).split())
    file = open(args.model, 'wb')
    pkl.dump(model, file)
    file.close()
