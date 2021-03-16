
from temp_task import RihkyeTask
from temp_model import FullNetwork

dataset = RihkyeTask()
model = FullNetwork()

for i in range(10):
    input, target = dataset()
    output = model(input, target)