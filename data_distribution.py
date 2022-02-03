import pickle
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

data_dir = 'data'
data_type = 'chinese.standard.nocnf'
# data_type = 'data.clean'
data_split = 'test'
data_name = f'{data_dir}/{data_type}/chinese-standard-{data_split}.pkl'
# data_name = f'{data_dir}/{data_type}/chinese-{data_split}.pkl'
with open(data_name, 'rb') as f:
    dataset = pickle.load(f)

n_depth = {}
size = len(dataset['depth'])
for d in dataset['depth']:
    if d in n_depth:
        n_depth[d] += 1
    else:
        n_depth[d] = 1
n_depth = OrderedDict(sorted(n_depth.items()))

log_name = 'log/depth_dist'
tag = f'{data_type}/{data_split}'
writer = SummaryWriter(log_name)
for k, v in n_depth.items():
    writer.add_scalar(tag, v/size, k)
writer.flush()
writer.close()