import click

import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable

import os
import random
import math
from bisect import bisect_left


def exponential_distribution(lambda_=1.0):
    u = random.random()
    x = - math.log(u) / lambda_
    return x


# with open("dataset2.txt", 'w') as f:
#     hashes = [exponential_distribution() for _ in range(19000)]
#     hashes = sorted(hashes)
#     for i in range(19000):
#         f.write("%f\n" % hashes[i])


def read_keys(file):
    with open(file, "r") as f:
        for line in f:
            yield float(line.strip())


def sorted_hash_map(N=19000):
    hashes = list(read_keys("dataset2.txt"))
    def random_fun():
        index = random.randint(0, N - 1)
        hash_ = hashes[index]
        return hash_, index
    return hashes, random_fun


def get_model(dim=128):
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


@click.command()
@click.option('--n', default=19000, type=int,
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
        idx = bisect_left(numbers, x)
        print('Real:', idx, 'Predicted:', float(pred.data[0]))

    _test(1.5)


if __name__ == '__main__':
    main()
