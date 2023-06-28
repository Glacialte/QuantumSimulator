# outputsから画像を作成する
import json
import matplotlib.pyplot as plt

with open('./outputs/test.json', 'r') as f:
    data = json.load(f)
    x = data['x']
    y = data['y']
    plt.plot(x, y)
    plt.xlabel('qubit')
    plt.ylabel('time')
    plt.savefig('./outputs/test.png')
    plt.show()
