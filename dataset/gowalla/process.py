record = []
with open('train.txt') as f:
    for line in f:
        items = line.strip().split()
        for i in items[1:]:
            record.append(items[0]+' '+i+' 1\n')
with open('train.txt','w') as f:
    f.writelines(record)