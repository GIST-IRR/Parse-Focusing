import os

dir = 'data/data.clean'
file = 'chinese-valid.txt'
path = os.path.join(dir, file)

with open(path, 'r') as f:
    data = []
    for line in f:
        data.append('(ROOT ' + line[:-1] + ')\n')

save_file = 'edit-' + file
save_path = os.path.join(dir, save_file)

with open(save_path, 'w') as f:
    for d in data:
        f.write(d)