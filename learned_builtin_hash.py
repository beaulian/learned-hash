import click

import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable

import os
import random
import math
from bisect import bisect_left

url_to_english_words = \
    'https://raw.githubusercontent.com/dwyl/english-words/master/words.txt'


def get_english_words():
    if not os.path.isfile('words.txt'):
        import subprocess
        subprocess.call(['wget', url_to_english_words])

    with open('words.txt') as f:
        english_words = []
        for line in f:
            english_words.append(line.strip())

    return english_words


def sorted_hash_map(N=1000):
    # ignore the map for now, just get random hashes
    english_words = get_english_words()
    english_words = english_words[:N]
    hashes = [abs(hash(word)) >> 32 for word in english_words]
    hashes = sorted(hashes)  # pseudo hash map
    def random_fun():
        index = random.randint(0, N - 1)
        hash_ = hashes[index]
        return hash_, index
    return hashes, random_fun


def get_model(dim=32):
    model = torch.nn.Sequential(
          torch.nn.Linear(1, dim),
          torch.nn.ReLU(),
          torch.nn.Linear(dim, 1),
        )
    return model


def _featurize(x):
    return torch.unsqueeze(Variable(torch.Tensor(x)), 1)


def naive_index_search(x, numbers):
    for idx, n in enumerate(numbers):
        if n > x:
            break
    return idx - 1


def bisect_search(x, numbers):
    i = bisect_left(numbers, x)
    if i:
        return i - 1
    raise ValueError


@click.command()
@click.option('--n', default=10000, type=int,
    help='Size of sorted array.')
@click.option('--lr', default=9e-3, type=float,
    help='Learning rate of DL model (only parameter that matters!)')
def main(n, lr):
    """CLI for creating machine learned index.
    """
    N = n
    numbers, random_fun = sorted_hash_map(N)

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    try:
        while True:

            batch_x = []
            batch_y = []
            for _ in range(32):
                x, y = random_fun()
                print(x, y)
                batch_x.append(x)
                batch_y.append(y)

            batch_x = _featurize(batch_x)
            batch_y = _featurize(batch_y)

            pred = model(batch_x) * N

            output = F.smooth_l1_loss(pred, batch_y)
            loss = output.data[0]

            print(loss)

            optimizer.zero_grad()
            output.backward()
            optimizer.step()

    except KeyboardInterrupt:
        pass

    def _test(x):
        pred = model(_featurize([x])) * N
        # idx = naive_index_search(x, numbers)
        idx = bisect_search(x, numbers)
        print('Real:', idx, 'Predicted:', float(pred.data[0]))

    _test(984666871)


if __name__ == '__main__':
    main()