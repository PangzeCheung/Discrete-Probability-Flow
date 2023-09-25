from pypianoroll import Multitrack
import os
import numpy as np

root = './piano_1'
dir = os.listdir('./piano_1')

for d in dir:
    if '.npy' in d:
        m = Multitrack(os.path.join(root, d))
        m.write(os.path.join(root, d[:-3]+'mid'))

