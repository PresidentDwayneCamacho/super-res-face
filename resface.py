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
    for i in range(frame_num):
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
    writer = cv2.VideoWriter("vid/rich_out.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 23, (w,h))
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


def video_demo(ground_file,unknown_file,subject):
    '''
        create a video demo with
        face recognition for any video file,
        whether ehnahced or not
    '''
    # import training image as numpy array
    ground_img = init_img(ground_file)
    # return encoding scheme for training image
    # used to be compared to all other images
    # encoding is a 128D array of given face
    known = encode_face(ground_img)[0][0]
    # import full video as a set of frames
    # frames is array of unedited video images
    frames = full_video(unknown_file)
    # init empty list to be modified video images
    images = []
    # get the length of frames
    frame_num = len(frames)
    # iterate through each unedited frame
    for i in range(frame_num):
        # get current frame
        frame = frames[i]
        # create recognition and encoding frames
        # create face encodings and locations
        # encoding is a 128D array of given face
        unknowns,locations = encode_face(frame)
        # iterate through unknown encodings and locations
        for unknown,rect in zip(unknowns,locations):
            # recognize unknown face against the ground
            # return truthiness array of which faces are recognized
            results = recognize_face([known],unknown)
            # get first results
            # usually result array is 1 element list
            result = results[0]
            # condition if face is recognized
            if result:
                # make a name of the subject recognized
                # if they face is recognized
                # based on the above encoding
                subtitle = ' '.join(subject)
            # condition if face is not recognized
            else:
                # add a default, unknown subtitle
                # if face is not recognized
                subtitle = '???'
            # get each corner of the rectangle
            # which indicates if a face is recognized
            top,bottom,left,right = rect.top(),rect.bottom(),rect.left(),rect.right()
            # draw the rectangle bordering the face with a red box
            cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
            # draw solid rectangle at bottom of face box
            cv2.rectangle(frame,(left,bottom+20),(right,bottom),(0,0,255),cv2.FILLED)
            # get the font of the text
            font = cv2.FONT_HERSHEY_DUPLEX
            # draw the text to the solid bordering
            # beneath the face if recognized
            cv2.putText(frame,subtitle,(left+4,bottom+16),font,1.0,(255,255,255),1)
        # add processed frame to the array of processed frames
        images.append(frame)
        # print the frame number that is printed to console
        # this gives a primitive timescale
        print(i+1,'/',frame_num)
    # get the dimensions of the processed image array
    # for use of resolution output of processed video
    h,w,l = images[0].shape
    # get filepath of videofile
    base = unknown_file.split('.')[0]+'-face-rec.mp4'
    # create VideoWriter object
    # at base video file with specified encoding scheme
    # at specific frame rate with specific frame size
    writer = cv2.VideoWriter(base, cv2.VideoWriter_fourcc(*"MJPG"), 23, (w,h))
    # iterate through array of images
    for img in images:
        # write processed file image by image with writer obj
        writer.write(img)
    # terminate the video writer
    writer.release()


def image_demo(ground_file,unknown_file,zoom,subject):
    '''
        function which enhances the resolution of image files
    '''
    # init image enhancer (Neural Enhance) object
    enhancer = neural_enhance()
    # intialize the ground, training image
    ground_img = init_img(ground_file)
    # init an unknown image
    # against which to compare the ground image
    unknown_img = init_img(unknown_file)
    # initialize image which will become doubled image
    unknown_img_x2 = init_img(unknown_file)
    # enhance image: double resolution of unknown image
    # convert image to np array
    unknown_img_x2 = np.array(enhancer.process(unknown_img_x2))
    # encode training image to ground encoding
    ground_encode = encode_face(ground_img)[0][0]
    # newline to console
    print('\n')
    # create filename based on input name of subject
    outpath_subject = '_'.join(subject)
    # same the ground image to the filepath
    scipy.misc.imsave('img/ground_img_'+outpath_subject+'.jpg',ground_img)
    # call function to encode and recognize unenhanced face image
    unknown_img = recognize_img(ground_encode,subject,unknown_img,2)
    # same image with output information to directory
    scipy.misc.imsave('img/1x_img_'+outpath_subject+'.jpg',unknown_img)
    # call function to encode and recognize enhanced face image
    unknown_img_x2 = recognize_img(ground_encode,subject,unknown_img_x2,1)
    # same image with output information to directory
    scipy.misc.imsave('img/2x_img_'+outpath_subject+'.jpg',unknown_img_x2)


def recognize_img(ground_encode,subject,unknown_img,text_factor):
    '''
        function to recognize faces in images
        based on inputs of the training encoding
        the name of the subject
        the unknown image
    '''
    # create coding and location of unknown image
    unknown_encode,unknown_locations = encode_face(unknown_img)
    # iterate through encodings and locations of faces
    for unknown,rect in zip(unknown_encode,unknown_locations):
        # determine if training face is in unknown image
        results = recognize_face([ground_encode],unknown)
        # get first element of result list
        result = results[0]
        # condition to determine if face is recognized in unknown
        if result:
            # condition if result is a list
            if isinstance(subject,list):
                # join array into string representing name
                subtitle = ' '.join(subject)
            # condition if result is string
            else:
                # set string name to name
                subtitle = subject
        # condition if face is not recognized in unknown
        else:
            # subject 'name' is unknown
            subtitle = 'Unknown'
        # output the name of the subject if recognized
        # or unknown if the subject is not recognized
        print(subtitle)
        # get corner of the rectangle drawn around recognized face
        top,bottom,left,right = rect.top(),rect.bottom(),rect.left(),rect.right()
        # set font for subtitle of subject or unknown
        font = cv2.FONT_HERSHEY_DUPLEX
        # draw rectangle around the face recognized
        cv2.rectangle(unknown_img,(left,top),(right,bottom),(0,255,0),2//text_factor)
        # draw name to the image, either subject or unknown
        cv2.putText(unknown_img,subtitle,
            (40//text_factor,len(unknown_img)-10//text_factor),
            font,1.0/text_factor,(255,255,255),1)
    # return the processed image
    return unknown_img



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
        #recognize_video(sys.argv[1],sys.argv[2],sys.argv[3:])
        video_demo(sys.argv[1],sys.argv[2],sys.argv[3:])
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
