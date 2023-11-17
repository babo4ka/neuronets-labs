import socket
import json
import numpy as np
import matplotlib.pyplot as plt

HOST = ('localhost', 10000)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(HOST)

x1 = np.random.random(20)
x2 = x1 + [np.random.randint(10)/10 for i in range(20)]
x1 = list(x1)
x2 = list(x2)

data = json.dumps({"x1": x1, "x2": x2})
client.send(data.encode())

res = client.recv(4096)
res = json.loads(res.decode())
print(res)
answers = res.get("answers")
dots0 = res.get("dots0")
dots1 = res.get("dots1")

i = 0
a = [x1, x2]


start, end = [0, 0.5], [0.5, 0]

xs0 = list()
ys0 = list()
for el in dots0:
    xs0.append(el[0])
    ys0.append(el[1])

xs1 = list()
ys1 = list()
for el in dots1:
    xs1.append(el[0])
    ys1.append(el[1])

plt.scatter(xs0, ys0, c='red')
plt.scatter(xs1, ys1, c='green')

plt.plot(start, end)
plt.show()
