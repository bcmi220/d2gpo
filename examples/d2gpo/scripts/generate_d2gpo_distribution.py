import scipy.stats as stats
import sys
import numpy as np
import tqdm
from sklearn.utils.extmath import softmax
import h5py
import argparse

def scatter(a, dim, index, b): # a inplace
    expanded_index = tuple([index if dim==i else np.arange(a.shape[i]).reshape([-1 if i==j else 1 for j in range(a.ndim)]) for i in range(a.ndim)])
    a[expanded_index] = b

if __name__ == '__main__':
    '''
    '''

    parser = argparse.ArgumentParser(description='Prior Distribution Generation')
    parser.add_argument("--d2gpo_mode", type=str, default="")
    parser.add_argument("--d2gpo_order_idx", type=str, default="")

    parser.add_argument("--d2gpo_softmax_position", type=str, default="")
    parser.add_argument("--d2gpo_softmax_temperature", type=float, default=1.0)
    
    parser.add_argument("--d2gpo_distribution_output", type=str, default="")
    parser.add_argument("--d2gpo_sample_width", type=int, default=200)

    parser.add_argument("--d2gpo_gaussian_std", type=float, default=1)
    parser.add_argument("--d2gpo_gaussian_offset", type=int, default=0)

    parser.add_argument("--d2gpo_linear_k", type=float, default=-1)

    parser.add_argument("--d2gpo_cosine_max_width", type=int, default=200)
    parser.add_argument("--d2gpo_cosine_offset", type=int, default=0)

    args = parser.parse_args()

    mode = args.d2gpo_mode
    assert mode in ['gaussian', 'linear', 'cosine']

    if mode == 'gaussian':
        std = args.d2gpo_gaussian_std
        offset = args.d2gpo_gaussian_offset
        mean = 0
        distribution_func = stats.norm(mean, std)
    elif mode == 'linear':
        k = args.d2gpo_linear_k
        assert k < 0
        b = 1.0
        offset = 0
        assert (-b / k) >= (offset + args.d2gpo_sample_width)
    elif mode == 'cosine':
        max_width = args.d2gpo_cosine_max_width
        offset = args.d2gpo_cosine_offset
        assert max_width >= (offset + args.d2gpo_sample_width)

    assert args.d2gpo_softmax_position in ['presoftmax', 'postsoftmax']

    # load the order information
    with open(args.d2gpo_order_idx, 'r', encoding='utf-8') as fin:
        data = fin.readlines()
    data = [[int(item) for item in line.strip().split()] for line in data if len(line.strip())>0]

    assert len(data) == len(data[0])

    if args.d2gpo_sample_width == 0:
        args.d2gpo_sample_width = len(data)

    x = np.arange(args.d2gpo_sample_width) + offset

    if mode == 'gaussian':
        y_sample = distribution_func.pdf(x)
    elif mode == 'linear':
        y_sample = k * x + b
    else:
        y_sample = np.cos(np.pi / 2 * x / max_width)

    if args.d2gpo_softmax_position == 'presoftmax':
        y_sample = y_sample / args.d2gpo_softmax_temperature 
        y_sample = softmax(np.expand_dims(y_sample,0)).squeeze(0)

    y = np.zeros(len(data))

    y[:args.d2gpo_sample_width] = y_sample

    print(y[:args.d2gpo_sample_width])

    label_weights = np.zeros((len(data), len(data)), dtype=np.float32)

    for idx in tqdm.tqdm(range(len(data))):
        sort_index = np.array(data[idx])
        resort_index = np.zeros(len(data), dtype=np.int)
        natural_index = np.arange(len(data))
        scatter(resort_index, 0, sort_index, natural_index)
        weight = y[resort_index]
        label_weights[idx] = weight

    f = h5py.File(args.d2gpo_distribution_output,'w')
    f.create_dataset('weights', data=label_weights)
    f.close()


