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


def gen_imgs(path):
    grd_img = init_img(path+'ground.jpg')
    hres_img = init_img(path+'high.jpg')
    lres_img = init_img(path+'low.jpg')
    hresf_img = init_img(path+'fhigh.jpg')
    lresf_img = init_img(path+'flow.jpg')
    return grd_img,hres_img,lres_img,hresf_img,lresf_img


def encode_img(ground,encodings):
    for enc in encodings:
        results = recognize_face([ground],enc)
        res = results[0]
        if res:
            return True
        else:
            return False
    return False


def compare_subject(grd,res):
    enc,_ = encode_face(res)
    out = encode_img(grd,enc)
    return out


def compile_data(f_obj,subject,*args):
    num = subject.split('/')[-1]
    vals = ','.join([str(int(x)) for x in args])
    f_obj.write(num+vals+'\n')


def compile_data_list(running_data,subject,*args):
    running_data.append(list(args))


def summarize_stats(data):
    pass


def run(img_dir):
    enhancer = neural_enhance()
    cur_data = []
    for path in img_dir[:2]:
        print(path)
        grd,hres,lres,hresf,lresf = gen_imgs(path)
        grd_enc = encode_face(grd)[0][0]
        x2res = enhancer.process(lres)
        x2resf = enhancer.process(lresf)
        ht = compare_subject(grd_enc,hres)
        lt = compare_subject(grd_enc,lres)
        x2t = compare_subject(grd_enc,x2res)
        hf = compare_subject(grd_enc,hresf)
        lf = compare_subject(grd_enc,lresf)
        x2f = compare_subject(grd_enc,x2resf)
        compile_data_list(cur_data,path,ht,lt,x2t,hf,lf,x2f)
    summarize_stats(cur_data)


def driver():
    img_dir = dir_list('img/')
    run(img_dir)


if __name__ == '__main__':
    driver()



# end of file
