# import the face encoding and recognizing
# functionality from python-wrapped dlib library
from dlibface import encode_face,recognize_face
# import super resolution object
from sres import neural_enhance
# number manipulation function
import numpy as np
# scientific computation
import scipy
# time used for testing purposes
import time
# computer vision library
import cv2
# used for terminal interfacing
import sys


def init_img(filepath):
    '''
        import images as numpy arrays
    '''
    return scipy.ndimage.imread(filepath)


def enhancement(filepath):
    '''
        function which calls the super-resolution
        image enhancement functionality
    '''
    # import full video at once
    # TODO limit number of frames
    #      that can be imported
    frames = full_video(filepath)
    # init super-resolution object
    # to increase resolution of video
    enhancer = neural_enhance()
    # empty image array of
    # resolution-increased images
    images = []
    # get number of video frames
    frame_num = len(frames)
    # alter framerate if avi file format
    if '.avi' in filepath:
        # halve frame rate if avi
        frame_num = int(frame_num/2)
    # iterate through each frame of video
    # TODO remove the divide by 20 when testing is complete
    for i in range(int(frame_num/20)):
        # enhance each image of the video
        # convert to np array
        # collect into array of enhance images
        images.append(np.array(enhancer.process(frames[i])))
        # output information regarding current
        # frame number to user
        print('\n',i+1,'/',frame_num)
    # get height and width attributes from each frame
    h,w,l = images[0].shape
    # TODO change output directory
    # write frames to video file with mjpg encoding
    writer = cv2.VideoWriter("vid/rich_out.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15, (w,h))
    # iterate through each enhanced frame
    for img in images:
        # write single frame each iteration
        writer.write(img)
    # finish the writing process
    writer.release()


# wroking near real time face recognition
def recognize(ground_file,unknown_file,subject):
    '''
        function to determine if the dlib face recognition
        is able to match the ground subject with the unknown
    '''
    # initialize ground image
    ground_img = init_img(ground_file)
    # an encoding of the ground subject
    # against which to match
    known = encode_face(ground_img)[0][0]
    # import the video stream
    vid = cv2.VideoCapture(unknown_file)
    # iterate through frames there are still frames
    while vid.isOpened():
        # read a single frame of the video
        ret,frame = vid.read()
        # condition if the current frame is readable
        if ret:
            # get face encoding of images 'in the wild'
            # including spatial positions of each face
            unknowns,locations = encode_face(frame)
            # iterate through each encoding and location
            for unknown,rect in zip(unknowns,locations):
                # return array of whether subject/known face
                # matches the unknown faces
                results = recognize_face([known],unknown)
                # get first element of results, true or false
                result = results[0]
                # if subject recognized, print subject name
                if result:
                    # join the subject name into single string
                    subtitle = ' '.join(subject)
                # if subject not recognized, print unknown
                else:
                    # output the string indicating
                    # that a person is not known
                    subtitle = 'Unknown person'
                # get the corners of the face locations
                top,bottom,left,right = rect.top(),rect.bottom(),rect.left(),rect.right()
                # output green rectangle around
                # the area the face is recognized
                cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
                # output placard at base of recognition rectangle
                cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0,0,255),cv2.FILLED)
                # set font to predefined opencv font
                font = cv2.FONT_HERSHEY_DUPLEX
                # output text with subject name to bottom of rectangle
                cv2.putText(frame,subtitle,(left+6,bottom-6),font,1.0,(255,255,255),1)
            # display the current frame to ffmpeg console
            cv2.imshow('',frame)
            # condition allows exiting of the loop
            # if 'q' is pressed or 'x' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # break out of loop if condition is true
                break
        # if frame is unreadable, break from while
        else:
            # break if frame is unreadable
            break
    # closes alread opened file or camera
    vid.release()
    # terminate the windows from opencv
    cv2.destroyAllWindows()


def driver():
    '''
        condition to call recognition or enhancement
    '''
    # if long enough arguments
    # to do face recognition
    if len(sys.argv) >= 4:
        # attempt to recognize a face from ground vs test
        # ground, test video, name
        recognize(sys.argv[1],sys.argv[2],sys.argv[3:])
    # if long enough to increase resolution
    elif len(sys.argv) == 2:
        # enhance pixelation of window
        enhancement(sys.argv[1])
    # if imporper arguments, exit program
    else:
        # print improper number args to user
        print('Improper number of arguments')
        # tell how to input terminal window args
        print('Recognition: ground, test video, subject name')


def full_video(filepath):
    '''
        full video importation
    '''
    # capture video with default opencv function
    vid = cv2.VideoCapture(filepath)
    # quantify number of frames
    size = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # capture all frames in video
    frames = [vid.read()[1] for x in range(size)]
    # return total frames
    return frames


if __name__ == '__main__':
    '''
        entry point for program
    '''
    # call beginning of program
    driver()


# enhance and recognize images
