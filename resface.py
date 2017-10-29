from dlibface import encode_face,recognize_face
from facer import neural_enhance
import numpy as np
import scipy
#import dlib
import time
import cv2
import sys


def init_img(filepath):
    return scipy.ndimage.imread(filepath)


def enhancement(filepath):
    frames = full_video(filepath)
    images = []
    enhancer = neural_enhance()
    frame_num = len(frames)
    if '.avi' in filepath:
        frame_num = int(frame_num/2)
    for i in range(int(frame_num/20)):
        images.append(np.array(enhancer.process(frames[i])))
        print('\n',i+1,'/',frame_num)
    h,w,l = images[0].shape
    writer = cv2.VideoWriter("vid/rich_out.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15, (w,h))
    for img in images:
        writer.write(img)
    writer.release()


# attempt at real time face recognition
def recognize(ground_file,unknown_file,subject):
    ground_img = init_img(ground_file)
    known = encode_face(ground_img)[0]
    vid = cv2.VideoCapture(unknown_file)
    while vid.isOpened():
        ret,frame = vid.read()
        if ret:
            unknown = encode_face(frame)[0]
            results = recognize_face([known],unknown)
            result = results[0]
            if result:
                subtitle = ' '.join(subject)
            else:
                subtitle = 'Unknown person'
            cv2.putText(frame,subtitle,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv2.imshow('',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    vid.release()
    cv2.destroyAllWindows()


def driver():
    if len(sys.argv) >= 4:
        # ground, test video, name
        recognize(sys.argv[1],sys.argv[2],sys.argv[3:])
    elif len(sys.argv) == 2:
        enhancement(sys.argv[1])
    else:
        print('Improper number of arguments')
        print('Recognition: ground, test video, subject name')


def full_video(filepath):
    vid = cv2.VideoCapture(filepath)
    size = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = [vid.read()[1] for x in range(size)]
    return frames


if __name__ == '__main__':
    driver()



# enhance and recognize images
