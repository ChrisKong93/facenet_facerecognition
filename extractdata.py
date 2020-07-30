import numpy as np


def detect(data):
    a = data.tolist()
    print(a)
    exit()
    i = 0
    for m in a:
        if i == 12:
            print(m)
            print(a[m])
            exit()
        i += 1


def extract(data, net, filename):
    a = data.tolist()
    # print(a)
    x = []
    for n in net:
        # print(n)
        # print(a[n])
        for b in a[n]:
            # print(b)
            # print(a[n][b].flatten())
            for d in a[n][b].flatten():
                print(d)
                x.append(d)
        # for b in a:
        #     print(b)
        #     print(a[b])
        #     if b == n:
        #         for c in a[b]:
        #             print(c)
        #             for d in a[b][c].flatten():
        #                 print(d)
        #                 x.append(d)
        #         break
    for y in x:
        s = '[' + str(y) + ']'
        # print(s)
        with open(filename, 'a+', encoding='utf-8') as f:
            f.write(s + '\n')


def run(filename):
    if filename == 'Pnet.txt':
        extract(data1, pnet, filename)
    elif filename == 'Rnet.txt':
        extract(data2, rnet, filename)
    elif filename == 'Onet.txt':
        extract(data3, onet, filename)


pnet = ['conv1', 'PReLU1', 'conv2', 'PReLU2', 'conv3', 'PReLU3', 'conv4-1', 'conv4-2']
rnet = ['conv1', 'prelu1', 'conv2', 'prelu2', 'conv3', 'prelu3', 'conv4', 'prelu4', 'conv5-1', 'conv5-2']
onet = ['conv1', 'prelu1', 'conv2', 'prelu2', 'conv3', 'prelu3', 'conv4', 'prelu4', 'conv5', 'prelu5', 'conv6-1',
        'conv6-2', 'conv6-3']
data1 = np.load('./align/det1.npy', allow_pickle=True, encoding="latin1")
data2 = np.load('./align/det2.npy', allow_pickle=True, encoding="latin1")
data3 = np.load('./align/det3.npy', allow_pickle=True, encoding="latin1")

if __name__ == '__main__':
    detect(data1)
    # run(filename='Pnet.txt')
    # run(filename='Rnet.txt')
    # run(filename='Onet.txt')
