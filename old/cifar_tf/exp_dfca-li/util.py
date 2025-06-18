import itertools
import numpy as np
import time


def test():
    pass


def dict_string(my_dict):
    str_list = []
    for key in my_dict:
        if type(my_dict[key]) == float:
            str_list.append(key + f"_{my_dict[key]:.6f}")
        else:
            str_list.append(key + f"_{my_dict[key]}")

    return "_".join(str_list)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

    # use: list(product_dict(**mydict))


def chunk(a, i, n):
    a2 = chunkify(a, n)
    return a2[i]

def chunkify(a, n):
    # splits list into even size list of lists
    # [1,2,3,4] -> [1,2], [3,4]

    k, m = divmod(len(a), n)
    gen = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return list(gen)

def chunkify_uneven(a, n):
    # print("len: ", len(a))
    # splits list into uneven size list of lists
    # e.g [1,2,3,4] -> [1], [2,3,4]

    sizes = np.random.randint(low=1, high=len(a) // n*2, size=n)
    # print("sizes: ", sizes)
    # normalize to length of a and ensure right sizes
    sizes = sizes / np.sum(sizes) * len(a)
    sizes = sizes.astype(int)
    sizes[-1] = len(a) - np.sum(sizes[:-1])

    sizes = np.maximum(sizes, 1)

    # print("sizes: ", sizes)
    gen = []
    s = 0
    for chunk in sizes:
        e = s + chunk
        gen.append(a[s:e])
        s = e
    # print("gen: ", gen)
    return gen


if __name__ == '__main__':
    start_time = time.time()
    test()
    duration = (time.time() - start_time)
    print("---train_cluster Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))