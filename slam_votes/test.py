import numpy
import datetime

now = datetime.datetime.now()
timeStr = now.strftime("%y_%m_%d-%H_%M_%S")
filename = "votes_" + timeStr + ".txt"
print(filename)
print("A")
