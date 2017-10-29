# face recognition example
import face_recognition
from PIL import Image
import dlib
import cv2

ground_picture = face_recognition.load_image_file('img/araya.jpg')
ground_encoding = face_recognition.face_encodings(ground_picture)[0]

unknown_picture = face_recognition.load_image_file('img/unknown.jpg')
unknown_encoding = face_recognition.face_encodings(unknown_picture)[0]

face_locations = face_recognition.face_locations( \
    unknown_picture, number_of_times_to_upsample=0,model='cnn')
results = face_recognition.compare_faces([ground_encoding],unknown_encoding)

for location in face_locations:
    top,right,bottom,left = location
    face_array = unknown_picture[top:bottom,left:right]
    picture_array = unknown_picture[:,:]
    cv2.rectangle(picture_array,(left,top),(right,bottom),(255,255,255),3)

    if results[0] == True:
        subject = 'Tom Araya'
    else:
        subject = 'Unknown Person'

    cv2.putText(picture_array,subject,(left-20,bottom+20), \
        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),0,cv2.LINE_AA)

    cv2.imshow('a face',picture_array)
    cv2.waitKey()
