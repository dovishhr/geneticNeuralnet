import sys
import os
from threading import Thread
import subprocess

def fire():
    while True:
        subprocess.run(["python3", "car.py"])

if len(sys.argv)-1:
    for i in range(int(sys.argv[1])):
        print(i)
        t = Thread(target=fire)
        t.start()
else:
    for i in range(os.cpu_count()):
        print(i)
        t = Thread(target=fire)
        t.start()
