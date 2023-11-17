import socket
import numpy as np
from my_own.task1.ORTask import Neuron_OR
import json

neuron = Neuron_OR()

HOST = ('localhost', 10000)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(HOST)
server.listen()
print("here")

while True:
    conn, addr = server.accept()
    print(addr)

    data = conn.recv(4096)
    if data:
        data = json.loads(data.decode())
        answers = list()

        x1 = data.get("x1")
        x2 = data.get("x2")
        inputData = [x1, x2]

        for i in range(len(x1)):
            d = np.array([inputData[0][i], inputData[1][i]])
            answers.append(neuron.operate_or(d))

        dots0, dots1 = neuron.get_dots()

        response = json.dumps({"answers": answers, "dots0": dots0, "dots1": dots1})
        conn.send(response.encode())

    conn.close()
