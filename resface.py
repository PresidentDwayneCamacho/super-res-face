from dlibface import encode_face,recognize_face
from facer import neural_enhance
import numpy as np
import scipy
import dlib
import time
import cv2
import sys


def init_img(filepath):
    return scipy.ndimage.imread(filepath)


def single_frame():
    enhancer = neural_enhance()
    img = scipy.ndimage.imread('img/bruce.jpg', mode='RGB')
    out = enhancer.process(img)
    out.save('img/bruce_neur2x.png')
    print(flush=True)


def enhancement():
    enhancer = neural_enhance()
    frames = full_video()
    images = []
    # this needs to be halved for avi files
    frame_num = int(len(frames)/2)
    for i in range(frame_num):
        images.append(np.array(enhancer.process(frames[i])))
        print('\n',i+1,'/',frame_num)
    h,w,l = images[0].shape
    writer = cv2.VideoWriter("vid/rich_out.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10,(w,h))
    for img in images:
        writer.write(img)
    writer.release()

# attempt at real time face recognition
def recognize():
    ground_img = init_img('img/rich.jpg')
    known = encode_face(ground_img)[0]
    #vid = init_video()
    vid = cv2.VideoCapture('vid/rich.mp4')
    while vid.isOpened():
        ret,frame = vid.read()
        if ret:
            unknown = encode_face(frame)[0]
            results = recognize_face([known],unknown)
            result = results[0]
            if result:
                subtitle = 'Rich is here!'
            else:
                subtitle = 'Not here!'
            cv2.putText(frame,subtitle,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv2.imshow('',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    vid.release()
    cv2.destroyAllWindows()


def user_input():
    if len(sys.argv) >= 2:
        return sys.argv[1]
    else:
        return 'vid/rich.mp4'


def pathway():
    if len(sys.argv) >= 3:
        if sys.argv[2] == 'encode':
            return True
    return False


def init_video():
    filepath = user_input()
    vid = cv2.VideoCapture(filepath)
    return vid


def full_video():
    filepath = user_input()
    vid = cv2.VideoCapture(filepath)
    size = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = [vid.read()[1] for x in range(size)]
    return frames


if __name__ == '__main__':
    if pathway():
        enhancement()
    else:
        recognize()



# enhance and recognize images
