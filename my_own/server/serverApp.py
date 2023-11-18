import socket
import numpy as np
from my_own.task1.ORTask import Neuron_OR
from my_own.task1.XoRTask import Neuron_XOR
from my_own.task2.gradient import Gradient
import json


OR_task = 1
XOR_task = 2
GRAD_dot_task = 3
GRAD_desc_task = 4
LIN_REG_task = 5

neuron_or = Neuron_OR()
neuron_xor = Neuron_XOR()
gradient = Gradient()

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
        task = data.get("task")

        if task == OR_task:
            answers = list()

            x1 = data.get("x1")
            x2 = data.get("x2")
            inputData = [x1, x2]

            for i in range(len(x1)):
                d = np.array([inputData[0][i], inputData[1][i]])
                answers.append(neuron_or.operate_or(d))

            dots0, dots1 = neuron_or.get_dots()

            response = json.dumps({"answers": answers, "dots0": dots0, "dots1": dots1})
            conn.send(response.encode())
        elif task == XOR_task:
            answers = list()

            x1 = data.get("x1")
            x2 = data.get("x2")
            inputData = [x1, x2]

            for i in range(len(x1)):
                d = np.array([inputData[0][i], inputData[1][i]])
                answers.append(neuron_xor.operate_xor(d))
            dots0, dots1 = neuron_xor.get_dots()

            response = json.dumps({"answers": answers, "dots0": dots0, "dots1": dots1})
            conn.send(response.encode())
        elif task == GRAD_dot_task:
            dot1 = data.get("dot1")
            dot2 = data.get("dot2")
            w = [dot1, dot2]
            gradient.set_w(w)
            result = gradient.get_grad_in_dot()
            result = result.tolist()
            response = json.dumps({"dot1": result[0], "dot2": result[1]})
            conn.send(response.encode())
        elif task == GRAD_desc_task:
            dot1 = data.get("dot1")
            dot2 = data.get("dot2")
            steps = data.get("steps")
            w = [dot1, dot2]
            gradient.set_w(w)
            result = gradient.get_grad_after_descent(steps)
            result = result.tolist()
            response = json.dumps({"dot1": result[0], "dot2": result[1]})
            conn.send(response.encode())


    conn.close()
