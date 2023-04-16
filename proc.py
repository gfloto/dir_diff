import os, sys

# find files
path = '/drive2/celeba/imgs'
files = os.listdir(path)

r = 0.8
n = int(len(files) * 0.8)
# make test train split
train = files[:n]
test = files[n:]

# make savepath dir
save_path = 'taming-transformers/custom_data'
os.makedirs(save_path, exist_ok=True)

# save names to text files
for (names, file) in [(train, 'train.txt'), (test, 'test.txt')]:
    with open(os.path.join(save_path, file), 'w') as f:
        for name in names:
            line = f'{path}/{name}\n'
            f.write(line)
