#import face_recognition
from PIL import Image
import scipy
import dlib
import cv2
import numpy as np

try:
    import face_recognition_models
except:
    print('Face recognition models not importable')
    quit()


face_detector = dlib.get_frontal_face_detector()
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)
pose_predictor = pose_predictor_68_point
face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def init_img(filepath):
    img = np.array(Image.open(filepath))
    return img


def encode_face(face_image):
    face_locations = face_detector(face_image,1)
    raw_landmarks = [pose_predictor(face_image, face_location) for face_location in face_locations]
    return [np.array(face_encoder.compute_face_descriptor(face_image,raw_landmark_set,1)) for raw_landmark_set in raw_landmarks]


def recognize_face(face_encodings,face_to_compare):
    tolerance = 0.6
    distance = np.linalg.norm(face_encodings - face_to_compare,axis=1)
    return list(distance <= tolerance)


'''
def test():
    img01 = init_img('img/araya.jpg')
    encoding01 = encode_face(img01)[0]
    img02 = init_img('img/unknown.jpg')
    encoding02 = encode_face(img02)[0]
    return recognize_face([encoding01],encoding02)
'''

#print(test())



# naked dlib
