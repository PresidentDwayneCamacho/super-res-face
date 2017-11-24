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


from pprint import pprint


def init_img(filepath):
    return scipy.ndimage.imread(filepath)


def show_img(img):
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_img_bgr(img):
    cv2.imshow('',cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def dir_list(filepath):
    return [x for x in sorted(os.listdir(filepath))]


def gen_cond():
    return ['ground','high','highf','low','lowf']


def gen_exp():
    return ['ground','high','highf','low','lowf','x2','x2f']


def gen_criteria():
    return ['match','size']

def gen_path(sub,cond):
    return 'img/'+sub+'/'+cond+'.jpg'


def gen_img(sub,cond):
    return init_img('img/'+sub+'/'+cond+'.jpg')


def print_data(exp_res):
    sub_dir = dir_list('img/')
    exp_dir = gen_exp()
    criteria = ['match','size']
    for sub in sub_dir:
        for exp in exp_dir:
            print(sub,exp,end=' ')
            for crit in criteria:
                print(crit,exp_res[sub][exp][crit],end=' ')
            print('')


def gen_exp_res(sub_dir,exp_dir):
    return {
        s:{ e:{
            'size':0.0,'match':0.0
        } for e in exp_dir } for s in sub_dir
    }


def compare_subject(grd,exp):
    encs,locs = encode_face(exp)
    if encs:
        encoding = encs[0]
        results = recognize_face([grd],encoding)
        res = results[0]
        loc = locs[0]
        if res:
            match = True
        else:
            match = False
        top,bottom,left,right = loc.top(),loc.bottom(),loc.left(),loc.right()
        area = (right-left)*(bottom-top)
        return match,area
    else:
        return False,0


def run_experiment():
    sub_dir = dir_list('img/')
    exp_dir = gen_exp()
    enhancer = neural_enhance()
    exp_res = gen_exp_res(sub_dir,exp_dir)
    # remove slicing
    for sub in sub_dir:
        print(sub)
        grd_img = gen_img(sub,'ground')
        imgs = {c:gen_img(sub,c) for c in exp_dir[1:-2]}
        imgs['x2'] = enhancer.process(imgs['low'])
        imgs['x2f'] = enhancer.process(imgs['lowf'])
        grd_enc = encode_face(grd_img)[0][0]

        #print('ground')
        #show_img_bgr(grd_img)

        for key,img in imgs.items():
            match,size = compare_subject(grd_enc,img)
            exp_res[sub][key]['match'] += match
            exp_res[sub][key]['size'] += size

            #print(key,match)
            #show_img_bgr(img)

    return exp_res


def compile_data(exp_res):

    sub_dir = dir_list('img/')
    exp_dir = gen_exp()[1:]
    res = {x:{'match':[],'size':[]} for x in exp_dir}
    criteria = gen_criteria()

    for exp in exp_dir:
        for sub in sub_dir:
            for crit in criteria:
                res[exp][crit].append(exp_res[sub][exp][crit])

    ms = {'mean':0.0,'std':0.0}
    stat = {x:{'match':dict(ms),'size':dict(ms)} for x in exp_dir}
    for exp in exp_dir:
        for crit in criteria:
            stat[exp][crit]['mean'] = np.mean(res[exp][crit])
            stat[exp][crit]['std'] = np.std(res[exp][crit])

    return stat


def display_data(stat):
    exp_dir = gen_exp()[1:]
    criteria = gen_criteria()
    for exp in exp_dir:
        print(exp)
        for crit in criteria:
            print('     {:6} {:0.4f} +/- {:0.4f}'.format(
                crit,stat[exp][crit]['mean'],stat[exp][crit]['std']
            ))



def driver():
    exp_res = run_experiment()
    sum_stat = compile_data(exp_res)
    display_data(sum_stat)



if __name__ == '__main__':
    driver()



# end of file
