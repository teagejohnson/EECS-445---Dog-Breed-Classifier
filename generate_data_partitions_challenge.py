import pandas as pd
import numpy as np


a = pd.read_csv('/Users/teagejohnson/Desktop/Michigan/Senior Year/Fall Semester/EECS 445/Projects/Project 2/data/dogs.csv')


target = a.loc[(a['task'] == 'target') & (a['partition'] != 'challenge')]
source = a.loc[(a['task'] == 'source') & (a['partition'] != 'challenge')]
challenge = a.loc[a['partition'] == 'challenge']

train = 0.75 * 0.75
val = 0.75 * 0.25
test = 0.25

target_partition_challenge = ['val'] * int(len(target) * val) + ['test'] * int(len(target) * test)
source_partition_challenge = ['val'] * int(len(source) * val) + ['test'] * int(len(source) * test)

target_partition_challenge += ['train'] * (len(target) - len(target_partition_challenge))
source_partition_challenge += ['train'] * (len(source) - len(source_partition_challenge))

np.random.shuffle(target_partition_challenge)
np.random.shuffle(source_partition_challenge)

target['partition_challenge'] = target_partition_challenge
source['partition_challenge'] = source_partition_challenge
challenge['partition_challenge'] = 'challenge'

b = pd.concat([target, source, challenge], axis=0)

print(target)
print(source)
print(challenge)

print(b)

b.to_csv('/Users/teagejohnson/Desktop/Michigan/Senior Year/Fall Semester/EECS 445/Projects/Project 2/data/dogs_challenge.csv')

