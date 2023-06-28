import cupy as cp
import time
import os
import sys
import json
from QuantumSimulator import *


if __name__ == '__main__':
    os.makedirs('./outputs', exist_ok=True)
    repeat = 10
    start_qubit = 20
    end_qubit = 30
    x_list = []
    y_list = []
    for n in range(start_qubit, end_qubit+1):
        cp.cuda.Stream.null.synchronize()
        start_time = time.time()
        for _ in range(repeat):
            # test用の量子回路を用いる
            pass
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        print('qubit: {}, time: {}'.format(n, end_time - start_time))
        x_list.append(n)
        y_list.append((end_time - start_time) / repeat)
    with open('./outputs/{}.json'.format('test'), 'w') as f:
        json.dump({'x': x_list, 'y': y_list}, f)

