import socket
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

HOST = ('localhost', 10000)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(HOST)

x = [2., 3., 4., .5, 6., 7., -1., -3., 12.]

data = json.dumps({"x": x, "task":5})
print(data)
client.send(data.encode())

res = client.recv(4096)
res = json.loads(res.decode())
answers = res.get("answers")
print(answers)

awaits = torch.pow((torch.cos(torch.tensor(x) + 3)), 2) * torch.pow(2, torch.tensor(x) - 3)
print(awaits)

