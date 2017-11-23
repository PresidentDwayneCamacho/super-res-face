# import the face encoding and recognizing
# functionality from python-wrapped dlib library
from dlibface import encode_face,recognize_face
# import super resolution object
from sres import neural_enhance
# import image class
from PIL import Image
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
# used for directory checking
import os


def init_img(filepath):
    '''
        import images as numpy arrays
    '''
    return scipy.ndimage.imread(filepath)


def show_img(img):
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def dir_list(path):
    return [path+x+'/' for x in sorted(os.listdir(path))]


'''
def gen_imgs(subject,grd_file,exp_file):
    grd_img = init_img(subject+grd_file)
    hres_img = init_img(subject+exp_file)
    lres_img = cv2.resize(hres_img,None,fx=0.21,fy=0.21,interpolation=cv2.INTER_CUBIC)
    return grd_img,hres_img,lres_img
'''

'''
def gen_imgs(subject,grd_file,exp_file):
    grd_img = init_img(subject+grd_file)
    hres_img = init_img(subject+exp_file)
    lres_img = init_img(subject+exp_file)
    return grd_img,hres_img,lres_img
'''

def gen_imgs(path):
    grd_img = init_img(path+'ground.jpg')
    hres_img = init_img(path+'high.jpg')
    lres_img = init_img(path+'low.jpg')
    return grd_img,hres_img,lres_img


def encode_img(ground,img,encodings):
    if not encodings:
        return False
    for enc in encodings:
        results = recognize_face([ground],enc)
        res = results[0]
        if res:
            return True
        else:
            return False


def determine_subject(grd,hres,lres,x2res):
    grd_enc = encode_face(grd)[0][0]
    hres_enc,hres_loc = encode_face(hres)
    lres_enc,lres_loc = encode_face(lres)
    x2res_enc,x2res_loc = encode_face(x2res)
    hres_out = encode_img(grd_enc,hres,hres_enc)
    lres_out = encode_img(grd_enc,lres,lres_enc)
    x2res_out = encode_img(grd_enc,x2res,x2res_enc)




def run(img_dir):
    enhancer = neural_enhance()
    for subject in img_dir:
        grd,hres,lres = gen_imgs(subject)
        x2res = enhancer.process(lres)
        determine_subject(grd,hres,lres,x2res)





def driver():
    img_dir = dir_list('img/')
    run(img_dir)



if __name__ == '__main__':
    driver()



# end of file
