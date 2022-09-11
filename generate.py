import pickle as pkl
import argparse
from train import generate_from_dict
from train import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--prefix')
    parser.add_argument('--length')
    args = parser.parse_args()

    file = open(args.model, 'rb')
    model = pkl.load(file)
    file.close()
    if args.prefix is None:
        previous = generate_from_dict(model.prior)
    else:
        previous = args.prefix.lower()
    result = args.prefix
    for i in range(int(args.length) - 1):
        previous = model.generate(previous)
        result += ' ' + previous
    print(result)
