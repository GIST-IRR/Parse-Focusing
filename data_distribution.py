import pickle
import argparse
from collections import OrderedDict
from numpy import vsplit
from torch.utils.tensorboard import SummaryWriter


def split_by_key(data, key):
    # split data to ordered dictionary by key
    result = {}
    for d in data:
        v = d[key]
        if v in result:
            result[v].append(d)
        else:
            result[v] = [d]
    return OrderedDict(sorted(result.items()))

def count_value(data, key):
    result = {}
    for d in data:
        n = d[key]
        if n in result:
            result[n] += 1
        else:
            result[n] = 1
    return OrderedDict(sorted(result.items()))


def main(args):
    data_dir = args.dir
    lang = 'french'
    split = 'standard'
    form = 'nocnf'
    data_type = f'{lang}.{split}.{form}'
    # data_type = f'{split}.{form}'
    data_split = 'test'
    data_name = f'{data_dir}/{data_type}/{lang}-{split}-{data_split}.pkl'
    # data_name = f'{data_dir}/{data_type}/chinese-{data_split}.pkl'
    with open(data_name, 'rb') as f:
        dataset = pickle.load(f)

    sort_type = 'depth'

    # dataset transform to dict for each sentence.
    data = []
    for i in range(len(dataset['word'])):
        data.append({})

    for k, vs in dataset.items():
        for i, v in enumerate(vs):
            if k == 'word':
                data[i].update({'len': len(v)})
            data[i].update({k: v})

    # criterion = 40
    # data = list(filter(lambda x: len(x['word']) <= criterion, data))
    # data = list(filter(lambda x: x['len'] <= criterion, data))
    # data = dataset['depth']

    # depth distribution for total data
    # depth_dist = count_value(data, 'len')

    # length distribution for total data
    # length_dist = count_value(data, 'depth')

    # depth distribution for each length
    split_dict = split_by_key(data, 'len')
    n_dict = {}
    for k in split_dict.keys():
        n_dict[k] = count_value(split_dict[k], 'depth')

    log_name = 'log/filter_depth_dist'
    writer = SummaryWriter(log_name)

    for k, vs in n_dict.items():
        tag = f'{data_type}/{data_split}/length_{k}'
        for k, v in vs.items():
            writer.add_scalar(tag, v, k)

    writer.flush()
    writer.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, default='data')
    parser.add_argument('--lang', default='english')
    args = parser.parse_args()

    main(args)