import dlib
import cv2
import sys
import numpy as np
import time
import scipy
from facer import neural_enhance



def single_frame():
    enhancer = neural_enhance()
    img = scipy.ndimage.imread('img/bruce.jpg', mode='RGB')
    out = enhancer.process(img)
    out.save('img/bruce_ne2x.png')
    print(flush=True)



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
