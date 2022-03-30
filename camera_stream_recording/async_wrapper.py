import time
from subprocess import Popen
while True:
    print("starting new process with video_recording_main.py")
    p = Popen('python main.py -u -p -vd 10', shell=True)
    time.sleep(600)
