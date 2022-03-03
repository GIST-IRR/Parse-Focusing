import pickle
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter


data_dir = 'data'
lang = 'english'
split = 'standard'
form = 'nocnf'
# data_type = f'{lang}.{split}.{form}'
data_type = f'{split}.{form}'
data_split = 'train'
data_name = f'{data_dir}/{data_type}/{lang}-{split}-{data_split}.pkl'
# data_name = f'{data_dir}/{data_type}/chinese-{data_split}.pkl'
with open(data_name, 'rb') as f:
    dataset = pickle.load(f)

sort_type = 'depth'

data = []
for i in range(len(dataset['word'])):
    data.append({})

for k, vs in dataset.items():
    for i, v in enumerate(vs):
        data[i].update({k: v})

criterion = 5
# data = list(filter(lambda x: len(x['word']) <= criterion, data))
data = list(filter(lambda x: x['depth'] == criterion, data))
# data = dataset['depth']

n_dict = {}
size = len(data)
for d in data:
    n = d['depth']
    if n in n_dict:
        n_dict[n] += 1
    else:
        n_dict[n] = 1
n_dict = OrderedDict(sorted(n_dict.items()))

log_name = 'log/filter_depth_dist'
tag = f'{data_type}/{data_split}'
writer = SummaryWriter(log_name)
for k, v in n_dict.items():
    writer.add_scalar(tag, v/size, k)
writer.flush()
writer.close()