from PIL import Image
import dlib
import scipy
import cv2

try:
    import face_recognition_models
except:
    print('Face recognition models not importable')
    quit()


def encoding():
    pass





ground_image = scipy.misc.imread('img/araya.jpg',mode='RGB')
face_detector = dlib.get_frontal_face_detector()

detector = face_detector(ground_image,1)

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)
pose_predictor = pose_predictor_68_point

raw_face_landmarks = [pose_predictor(ground_image,face_location) for face_location in detector]
ground_encoding = [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks][0]






face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)




ground_encoding =











# naked dlib
