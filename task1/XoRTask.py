import numpy as np

b = 1

dots0 = list()
dots0.append(list())
dots0.append((list()))

dots1 = list()
dots1.append(list())
dots1.append((list()))

def func_act(x):
    return 0 if x <= 0 else 1


def neuron(input_data):
    x = np.array([input_data[0], input_data[1], b])
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
    print(neuron(data))