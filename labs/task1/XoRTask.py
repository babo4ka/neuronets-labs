import numpy as np
import matplotlib.pyplot as plt


dots0 = list()
dots0.append(list())
dots0.append((list()))

dots1 = list()
dots1.append(list())
dots1.append((list()))


def func_act(x):
    return 0 if x <= 0 else 1


def neuron(input_data):
    x = np.array([input_data[0], input_data[1], 1])
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    hidden_neuron = np.array([w1, w2])
    w_out_neuron = np.array([-1, 1, -0.5])

    sum = np.dot(hidden_neuron, x)
    out = [func_act(x) for x in sum]
    out.append(1)
    out = np.array(out)

    sum = np.dot(w_out_neuron, out)
    return func_act(sum)


input_data = [(0, 0), (0, 1), (1, 0), (1, 1)]

for data in input_data:
    result = neuron(data)
    if result <= 0:
        dots0[0].append(data[0])
        dots0[1].append(data[1])
    else:
        dots1[0].append(data[0])
        dots1[1].append(data[1])
    print(data[0], 'XOR', data[1], '=', result)

start1, end1 = [0, 0.5], [0.5, 0]
start2, end2 = [0.5, 1], [1, 0.5]


plt.scatter(dots0[0][:], dots0[1][:], c='red')
plt.scatter(dots1[0][:], dots1[1][:], c='green')
plt.plot(start1, end1, start2, end2)
plt.show()
