import dlib
import cv2
import sys
import numpy as np
import time

from facer import NeuralEnhancer





def init_enhancer():
    enhance = NeuralEnhancer(loader=False)


def single_frame():
    vid = init_video()
    _,frame = vid.read()
    enhancer = NeuralEnhancer(loader=False)
    out = enhancer.process(frame)
    out.save('vid/jay.png')
    #cv2.imshow('',out)



def main_loop():
    vid = init_video()
    while True:
        ret,frame = vid.read()
        if ret == True:
            cv2.imshow('',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    vid.release()
    cv2.destroyAllWindows()


def user_input():
    if len(sys.argv) >= 2:
        # user input here
        return 'vid/jay.mp4'
    else:
        return 'vid/jay.mp4'


def init_video():
    filepath = user_input()
    vid = cv2.VideoCapture(filepath)
    return vid


if __name__ == '__main__':
    #main_loop()
    single_frame()


# enhance and recognize images
