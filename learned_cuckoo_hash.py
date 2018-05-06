import click

import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda')


def get_dataset():
    data = {}
    with open('dataset.txt') as f:
        for line in f:
            in_, out_ = line.strip().split()
            data[int(in_)] = int(out_)
    data_tuple = sorted(data.items(), key=lambda i: i[0])
    return list(map(lambda i: i[0], data_tuple)), list(map(lambda i: i[1], data_tuple))


def get_model(dim=128):
    model = torch.nn.Sequential(
          torch.nn.Linear(1, dim),
          torch.nn.ReLU(),
          torch.nn.Linear(dim, 1),
        )
    return model


def _featurize(x):
    return torch.unsqueeze(Variable(torch.Tensor(x)), 1)


x, y = get_dataset()


@click.command()
@click.option('--n', default=19000, type=int,
    help='Size of sorted array.')
@click.option('--lr', default=9e-3, type=float,
    help='Learning rate of DL model (only parameter that matters!)')
def main(n, lr):
    """CLI for creating machine learned index.
    """
    N = n

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    try:
        count = 0
        while True:

            batch_x = []
            batch_y = []
            for _ in range(32):
                x1, y1 = x[count], y[count]
                batch_x.append(x1)
                batch_y.append(y1)
                count += 1

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

    except IndexError:
        pass

    def _test(x2):
        pred_ = model(_featurize([x2])) * N
        # idx = naive_index_search(x, numbers)
        idx = 524
        print('Real:', idx, 'Predicted:', float(pred_.data[0]))

    _test(0)


if __name__ == '__main__':
    main()