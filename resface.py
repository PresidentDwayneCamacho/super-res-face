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
def recognize_video(ground_file,unknown_file,subject):
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


def image_demo(ground_file,unknown_file,zoom,subject):
    # TODO comment this code, fresh new code!

    enhancer = neural_enhance()

    ground_img = init_img(ground_file)
    unknown_img = init_img(unknown_file)
    unknown_img_x2 = init_img(unknown_file)
    unknown_img_x2 = np.array(enhancer.process(unknown_img_x2))

    ground_encode = encode_face(ground_img)[0][0]
    unknown_encode,unknown_locations = encode_face(unknown_img)
    unknown_encode_x2,unknown_locations_x2 = encode_face(unknown_img_x2)

    print('\n')

    ground_img_copy = cv2.cvtColor(ground_img.copy(),cv2.COLOR_BGR2RGB)
    cv2.imshow('',ground_img_copy)
    outpath_subject = '_'.join(subject)
    scipy.misc.imsave('img/ground_img'+outpath_subject+'.jpg',ground_img)
    cv2.waitKey(0)

    for unknown,rect in zip(unknown_encode,unknown_locations):
        results = recognize_face([ground_encode],unknown)
        result = results[0]
        if result:
            if isinstance(subject,list):
                subtitle = ' '.join(subject)
            else:
                subtitle = subject
        else:
            subtitle = 'Unknown'
        print(subtitle)
        top,bottom,left,right = rect.top(),rect.bottom(),rect.left(),rect.right()
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(unknown_img,subtitle,(10,len(unknown_img)-10),font,0.7,(255,255,255),1)
    unknown_img_copy = cv2.cvtColor(unknown_img.copy(),cv2.COLOR_BGR2RGB)
    cv2.imshow('',unknown_img_copy)
    scipy.misc.imsave('img/1x_img'+outpath_subject+'.jpg',unknown_img)
    cv2.waitKey(0)

    for unknown,rect in zip(unknown_encode_x2,unknown_locations_x2):
        results = recognize_face([ground_encode],unknown)
        result = results[0]
        if result:
            if isinstance(subject,list):
                subtitle = ' '.join(subject)
            else:
                subtitle = subject
        else:
            subtitle = 'Unknown person'
        top,bottom,left,right = rect.top(),rect.bottom(),rect.left(),rect.right()
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(unknown_img_x2,subtitle,(10,len(unknown_img_x2)-10),font,0.7,(255,255,255),1)
        print(subtitle)
    unknown_img_x2_copy = cv2.cvtColor(unknown_img_x2.copy(),cv2.COLOR_BGR2RGB)
    cv2.imshow('',unknown_img_x2_copy)
    scipy.misc.imsave('img/2x_img'+outpath_subject+'.jpg',unknown_img_x2)
    cv2.waitKey(0)


def driver():
    '''
        condition to call recognition or enhancement
    '''
    # TODO allow multiple names to be input for subject
    if len(sys.argv) >= 5:
        # ground, test, zoom, subject name
        image_demo(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4:])
    # if long enough arguments
    # to do face recognition
    elif len(sys.argv) >= 4:
        # attempt to recognize a face from ground vs test
        # ground, test video, name
        recognize_video(sys.argv[1],sys.argv[2],sys.argv[3:])
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


def mutex():
    if len(sys.argv) >= 5:
        filetype = sys.argv[1].lower()
        if filetype == 'image':
            pass
        elif filetype == 'video':
            pass
        else:
            print('')
    elif len(sys.argv) >= 3:
        # go to image enhancement
        pass
    else:
        print('Imporper number of arguments')


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
