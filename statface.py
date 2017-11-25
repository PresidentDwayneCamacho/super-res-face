# import the face encoding and recognizing
# functionality from python-wrapped dlib library
from dlibface import encode_face,recognize_face

from dlibface import tolerance_face
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


import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt


def init_img(filepath):
    return scipy.ndimage.imread(filepath)


def show_img(img):
    cv2.imshow('',cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def dir_list(filepath):
    return [x for x in sorted(os.listdir(filepath))]


def gen_cond():
    return ['high','low']


def gen_exp():
    return ['ground','high','highf','low','lowf','x2','x2f']


def gen_criteria():
    return ['match','dist','size']


def gen_criteria_dict():
    return {'match':0.0,'dist':0.0,'size':0.0}


def gen_path(sub,cond):
    return 'img/'+sub+'/'+cond+'.jpg'


def gen_img(sub,cond):
    return init_img('img/'+sub+'/'+cond+'.jpg')


def print_data(exp_res):
    sub_dir = dir_list('img/')
    exp_dir = gen_exp()
    criteria = gen_criteria()
    for sub in sub_dir:
        for exp in exp_dir:
            print(sub,exp,end=' ')
            for crit in criteria:
                print(crit,exp_res[sub][exp][crit],end=' ')
            print('')


def gen_exp_res(sub_dir,exp_dir):
    return {
        s:{ e: gen_criteria_dict()
            for e in exp_dir } for s in sub_dir
    }


def compare_subject(grd,exp):
    encs,locs = encode_face(exp)
    if encs:
        encoding = encs[0]
        results,distance = tolerance_face([grd],encoding)
        res,loc,dist = results[0],locs[0],distance[0]
        top,bottom,left,right = loc.top(),loc.bottom(),loc.left(),loc.right()
        area = (right-left)*(bottom-top)
        return res,dist,area
    else:
        return False,1.0,0


def run_experiment():
    sub_dir = dir_list('img/')
    cond_dir = gen_cond()
    exp_dir = gen_exp()
    enhancer = neural_enhance()
    exp_res = gen_exp_res(sub_dir,exp_dir)
    for sub in sub_dir:
        print(sub)
        grd_img = gen_img(sub,'ground')
        imgs = {c:gen_img(sub,c) for c in cond_dir}
        off = (int(sub)+(len(sub_dir))//2)%len(sub_dir)
        if off < 10:
            off = '0'+str(off)
        else:
            off = str(off)
        imgs['highf'] = gen_img(off,'high')
        imgs['lowf'] = gen_img(off,'low')
        imgs['x2'] = enhancer.process(imgs['low'])
        imgs['x2f'] = enhancer.process(imgs['lowf'])
        grd_enc = encode_face(grd_img)[0][0]

        '''
        # below is for testing, remove later
        x2_match = False
        x2_size = 0
        not_low_match = False
        # above is for testing, remove later
        '''

        for key,img in imgs.items():
            match,dist,size = compare_subject(grd_enc,img)

            '''
            # below is for testing, remove later
            if key == 'x2':
                x2_match = match
                x2_size = size
            if key == 'low':
                not_low_match = not match
            if x2_match and not_low_match:
                print('')
                print('increase match here!')
                print('x2 =',x2_match,'and low =',not not_low_match)
                print('at size',x2_size)
                print('')
            # above is for testing, remove later
            '''

            exp_res[sub][key]['match'] += match
            exp_res[sub][key]['dist'] += dist
            exp_res[sub][key]['size'] += size

            #print(key)
            #show_img(img)

    return exp_res


def compile_data(exp_res):

    sub_dir = dir_list('img/')
    exp_dir = gen_exp()[1:]
    criteria = gen_criteria()
    res = {x:gen_criteria_dict() for x in exp_dir}

    for exp in exp_dir:
        for crit in criteria:
            res[exp][crit] = []

    for exp in exp_dir:
        for sub in sub_dir:
            for crit in criteria:
                res[exp][crit].append(exp_res[sub][exp][crit])

    ms = {'mean':0.0,'std':0.0}
    stat = {x:gen_criteria_dict() for x in exp_dir}
    for exp in exp_dir:
        for crit in criteria:
            stat[exp][crit] = dict(ms)

    for exp in exp_dir:
        for crit in criteria:
            stat[exp][crit]['mean'] = np.mean(res[exp][crit])
            stat[exp][crit]['std'] = np.std(res[exp][crit])

    return stat


def criteria_mean(stat,exp_dir,criteria):
    return [stat[x][criteria]['mean'] for x in exp_dir]


def get_label(abbr):
    if abbr == 'high':
        label = 'high res'
    elif abbr == 'highf':
        label = 'high ctrl'
    elif abbr == 'low':
        label = 'low res'
    elif abbr == 'lowf':
        label = 'low ctrl'
    elif abbr == 'x2':
        label = 'enhanced res'
    elif abbr == 'x2f':
        label = 'upres ctrl'
    return label






def filter_data(stat):
    sub_dir = dir_list('img/')
    for sub in sub_dir:
        







def display_data(stat):

    exp_dir = gen_exp()[1:]
    criteria = gen_criteria()

    matches = criteria_mean(stat,exp_dir,'match')
    distance = criteria_mean(stat,exp_dir,'dist')
    size = criteria_mean(stat,exp_dir,'size')

    for i,e in enumerate(distance):
        distance[i] = 1.0-e

    labels = [get_label(x) for x in exp_dir]
    x_pos = np.arange(len(exp_dir))

    plt.bar(x_pos,distance,align='center',alpha=0.5)
    plt.xticks(x_pos,labels)
    plt.ylabel('vector distance')
    plt.title('threshold across groups')

    plt.figure()
    plt.bar(x_pos,matches,align='center',alpha=0.5)
    plt.xticks(x_pos,labels)
    plt.ylabel('proportion of matches')
    plt.title('matches across groups')

    plt.figure()
    plt.bar(x_pos,size,align='center',alpha=0.5)
    plt.xticks(x_pos,labels)
    plt.ylabel('sizes')
    plt.title('size across groups')

    plt.show()


def print_data(stat):
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


    # remove below for testing purposes
    filter_data(sum_stat)
    # remove above for testing purposes


    display_data(sum_stat)


if __name__ == '__main__':
    driver()



# end of file
