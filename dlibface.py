# import image manipulation capability
from PIL import Image
# import number manipulation ability
# used for high throughput array manipulation
import numpy as np
# import scientific functionality
import scipy
# import neural network capability
# used for face recognition
import dlib
# import computer vision library
import cv2

# try/catch for importing recognition models
try:
    # trained models for face recognition capability
    import face_recognition_models
# raise exception when any error occurs
except:
    # output error message to terminal
    print('Face recognition models not importable')
    # exit program if model cannot be imported
    quit()

# dlib.get_frontal_face_detector returns object_detector
# configured for human faces that are looking at the camera
# created using scan_fhog_pyramid
# linear classifier slides over hog pyramid
face_detector = dlib.get_frontal_face_detector()
# model used by the face_recognition package
# also created by the owner of the face_recognition library
# predictor_68_point_model estimates pose of face
# points are the corner of the mouth, along the eyebrows, on the eyes...
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
# shape_predictor takes image region and outputs set of
# point locations that define pose of an object
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)
# wrapper function to call location of face landmarks data in models folder
face_recognition_model = face_recognition_models.face_recognition_model_location()
# maps human faces to 128D vectors where pictures of same person
# mapped near each other and pictures of different people
# mapped far apart, is a .dat.bz file
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# encoding face
def encode_face(face_image):
    '''
        takes face image, encodes with hog
        maps human face to 128D vectors
    '''
    # creates object_detector with histogram of oriented gradients
    face_locations = face_detector(face_image,1)
    #
    raw_landmarks = [pose_predictor_68_point(face_image, face_location) for face_location in face_locations]
    #
    encode = [np.array(face_encoder.compute_face_descriptor(face_image,raw_landmark_set,1)) for raw_landmark_set in raw_landmarks]
    # 
    return encode,face_locations


def recognize_face(face_encodings,face_to_compare):
    tolerance = 0.6
    distance = np.linalg.norm(face_encodings - face_to_compare,axis=1)
    return list(distance <= tolerance)





# naked dlib
