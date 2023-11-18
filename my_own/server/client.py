import socket
import json
import numpy as np
import matplotlib.pyplot as plt

HOST = ('localhost', 10000)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(HOST)

dot1 = [5., 10.]
dot2 = [1., 2.]

data = json.dumps({"dot1": dot1, "dot2": dot2, "steps":500, "task": 4})
print(data)
client.send(data.encode())

res = client.recv(4096)
res = json.loads(res.decode())
print(res)
dot0 = res.get("dot1")
dot1 = res.get("dot2")
print([dot0, dot1])

