import random
random.seed(12345)
train = []
test = []
test_ratio=0.2
with open('ratings.txt') as f:
    for line in f:
        items = line.strip().split()
        if random.random()>test_ratio:
            train.append(line)
        else:
            test.append(line)

with open('train.txt','w') as f:
    f.writelines(train)

with open('test.txt','w') as f:
    f.writelines(test)

