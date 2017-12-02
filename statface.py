# import the face encoding and recognizing
# functionality from python-wrapped dlib library
from dlibface import encode_face,recognize_face
# import a function like recognize_face
# which also returns the magnitude
# of the difference between training and unknown face
from dlibface import tolerance_face
# import super resolution object
from sres import neural_enhance
# import deepcopy to copy statistics data structures
from copy import deepcopy
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
# import matplotlib for data display
import matplotlib.pyplot as plt
# reset matplotlib to default values
plt.rcdefaults()
# import matplotlib for data display
import matplotlib.pyplot as plt


def init_img(filepath):
    '''
        import images as numpy arrays
    '''
    return scipy.ndimage.imread(filepath)


def show_img(img):
    '''
        shows the image to user
        for testing purposes
    '''
    # show the image with the proper color encoding scheme
    cv2.imshow('',cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    # prevent the window from closing prematurely
    cv2.waitKey(0)
    # destroy all the open windows
    cv2.destroyAllWindows()


def dir_list(filepath):
    '''
        get the folders in the specified directory
    '''
    # create list of files in the specified directory
    return [x for x in sorted(os.listdir(filepath))]


def gen_cond():
    '''
        return list of high resolution and low
        resolution of images in directory
    '''
    return ['high','low']


def gen_exp():
    '''
        return list of the experimental and control conditions
    '''
    return ['ground','high','highf','low','lowf','x2','x2f']


def gen_criteria():
    '''
        return the three quantities by which
        the images are measured
    '''
    return ['match','dist','size']


def gen_criteria_dict():
    '''
        return the criteria which will be measured
        as the means of each group
    '''
    return {'match':0.0,'dist':0.0,'size':0.0}


def gen_path(sub,cond):
    '''
        return a filepath in at sub directory
        with a given condition as jpeg image
    '''
    return 'img/'+sub+'/'+cond+'.jpg'


def gen_img(sub,cond):
    '''
        initialize an image at the specified filepath
    '''
    return init_img('img/'+sub+'/'+cond+'.jpg')


def gen_exp_res(sub_dir,exp_dir):
    '''
        generate a data structure which stores results
        from the experiments
    '''
    # return comprehension with number for each folder
    # that stores experimental categories for each measurement
    return {
        s:{ e: gen_criteria_dict()
            for e in exp_dir } for s in sub_dir
    }


def compare_subject(grd,exp):
    '''
        compare the training image with an unknown image
    '''
    # return the encoding and locations of an unknown face
    encs,locs = encode_face(exp)
    # condition to determine if face was found in image
    # if no encoding, skip comparisons
    if encs:
        # get first encoding from list of face encodings
        encoding = encs[0]
        # make a comparison between training and experimental face
        # returns true if face is recognized, false if not
        # returns magnitude of difference between
        # training and unknown face
        results,distance = tolerance_face([grd],encoding)
        # unpack first values of lists of if face is recognized,
        # and location of face, and magnitude of similarity
        res,loc,dist = results[0],locs[0],distance[0]
        # unpack each corner of the rectangle
        top,bottom,left,right = loc.top(),loc.bottom(),loc.left(),loc.right()
        # find area of face
        area = (right-left)*(bottom-top)
        # return tuple of
        # face is recognized
        # magnitude between faces
        # and area of recognized face
        return res,dist,area
    # condition if no face is found in image
    else:
        # return false, size of face is 0
        return False,1.0,0


def run_experiment():
    '''
        the experiment to determine rate of similarity between
        low resolution and high resolution images and returns
        the stats related to rate of recognition
    '''
    # generate list of subjects in specified directory
    sub_dir = dir_list('img/')
    # get conditions of resolution (high,low)
    cond_dir = gen_cond()
    # get experiment conditions of various resolutions
    # including increased resolution and ground, etc...
    exp_dir = gen_exp()
    # instanciate object which enhances resolution of image
    enhancer = neural_enhance()
    # gen data structure which holds experimental results
    exp_res = gen_exp_res(sub_dir,exp_dir)
    # iterate through every file which represents subject
    # in specified file directory
    for sub in sub_dir:
        # print the subject number to the terminal
        print(sub)
        # initialize the ground truth training image
        grd_img = gen_img(sub,'ground')
        # create dictionary of images at various exp conditions
        imgs = {c:gen_img(sub,c) for c in cond_dir}
        # create offset by which to query control test image
        off = (int(sub)+(len(sub_dir))//2)%len(sub_dir)
        # if subject is less than ten, append with zero digit
        # to make file dir string two chars long
        if off < 10:
            # add zero digit to front of string less than 10
            off = '0'+str(off)
        # condition if file less than 10
        else:
            # convert int of filepath to str
            off = str(off)
        # generate false positive, high resolution control image
        imgs['highf'] = gen_img(off,'high')
        # generate false positive, low resolution control image
        imgs['lowf'] = gen_img(off,'low')
        # generate enhanced resolution image from low res image
        imgs['x2'] = enhancer.process(imgs['low'])
        # generate enhanced resolution image from low res false positive
        imgs['x2f'] = enhancer.process(imgs['lowf'])
        # encode image from ground truth training image
        grd_enc = encode_face(grd_img)[0][0]
        # iterate through each image in experimental groups
        for key,img in imgs.items():
            # comparison between ground truth training image
            # and image in each experimental group
            match,dist,size = compare_subject(grd_enc,img)
            # add match to match experimental results
            # if true, adds 1 to sum
            # if false, adds 0 to sum
            exp_res[sub][key]['match'] += match
            # increment magnitude comparison
            # to determine 'closeness' of each face
            exp_res[sub][key]['dist'] += dist
            # add size of each face
            exp_res[sub][key]['size'] += size
    # return structure of data results contains
    # all data across experimental categories
    return exp_res


def compile_data(exp_res):
    '''
        compiles the data into mean and standard
        deviation across experimental categories
    '''
    # generate directory list for each subject
    sub_dir = dir_list('img/')
    # get list of experimental conditions
    # including high res, low res, increased res, etc...
    exp_dir = gen_exp()[1:]
    # get size, match, distance
    criteria = gen_criteria()
    # creates dict of statistics
    # has match, dist, size
    res = {x:gen_criteria_dict() for x in exp_dir}
    # iterate through experimental conditions
    for exp in exp_dir:
        # iterate through criteria used
        for crit in criteria:
            # create empty list of samples
            res[exp][crit] = []
    # iterate through experimental result types
    for exp in exp_dir:
        # iterate through subject directory
        for sub in sub_dir:
            # iterate through criteria of stats used
            for crit in criteria:
                # append data element from results
                # to the statistic totaling data structure
                res[exp][crit].append(exp_res[sub][exp][crit])
    # dictionary of mean and standard deviation
    ms = {'mean':0.0,'std':0.0}
    # create dict of stats (mean, std) for each experimental condition
    stat = {x:gen_criteria_dict() for x in exp_dir}
    # iterate through experimental conditions
    for exp in exp_dir:
        # iterate through criteria used
        for crit in criteria:
            # create dictionary of mean and std
            # for each exp category
            stat[exp][crit] = dict(ms)
    # iterate through experimental condition
    for exp in exp_dir:
        # iterate through criteria (match,dist,size) measured
        for crit in criteria:
            # get mean of data list at
            # experimental conditions
            # and criteria (match,dist,size)
            stat[exp][crit]['mean'] = np.mean(res[exp][crit])
            # get std dev of data list at experimental
            # conditions and at criteria
            stat[exp][crit]['std'] = np.std(res[exp][crit])
    # return data structure with statistics
    return stat


def compile_filtered_data(exp_res):
    '''
        compile data structure where results
        are filtered by selected criteria
    '''
    # get list of subjects in directory
    sub_dir = dir_list('img/')
    # get list of experimental conditions without ground category
    exp_dir = gen_exp()[1:]
    # get list of measurement criteria
    criteria = gen_criteria()
    # generate dict of results for display purposes
    res = {x:gen_criteria_dict() for x in exp_dir}
    # iterate through experimental results list
    for exp in exp_dir:
        # iterate through criteria
        for crit in criteria:
            # init empty list at experimental category and criteria
            res[exp][crit] = []
    # iterate through experimental category
    for exp in exp_dir:
        # iterate through dir of subject numbers
        for sub in sub_dir:
            # condition if sub key in experimental res
            # if not, skip
            if sub in exp_res:
                # iterate through criteria
                for crit in criteria:
                    # add datum to list at specified categories
                    res[exp][crit].append(exp_res[sub][exp][crit])
    # creat dict of mean and std dev
    ms = {'mean':0.0,'std':0.0}
    # init statistics dict with criteria (match,dist,size)
    stat = {x:gen_criteria_dict() for x in exp_dir}
    # iterate through experimental categories
    for exp in exp_dir:
        # iterate through criteria
        for crit in criteria:
            # init new dict with mean and std at
            # experimental category and criteria
            stat[exp][crit] = dict(ms)
    # iterate through experimental categories
    for exp in exp_dir:
        # iterate through criteria
        for crit in criteria:
            # get mean of data list at
            # experimental conditions
            # and criteria (match,dist,size)
            # of filtered data
            stat[exp][crit]['mean'] = np.mean(res[exp][crit])
            # get std dev of data list at experimental
            # conditions and at criteria
            # of filtered data
            stat[exp][crit]['std'] = np.std(res[exp][crit])
    # return data structure for filtered statistics
    return stat


def criteria_mean(stat,exp_dir,criteria):
    '''
        return a list of mean values at different experimental categories
    '''
    return [stat[x][criteria]['mean'] for x in exp_dir]


def criteria_std(stat,exp_dir,criteria):
    '''
        return a list of std dev values at different experimental categories
    '''
    return [stat[x][criteria]['std'] for x in exp_dir]


def get_label(abbr):
    # if category is high,
    # set to high resolution
    if abbr == 'high': label = 'high res'
    # if category is high false positive,
    # set to high resolution ctrl
    elif abbr == 'highf': label = 'high ctrl'
    # if category is low
    # set to low resolution
    elif abbr == 'low': label = 'low res'
    # if category is low false positive,
    # set to low resolution control
    elif abbr == 'lowf': label = 'low ctrl'
    # if category is enhanced
    # set label to low ctrl
    elif abbr == 'x2': label = 'enhanced res'
    # if category is enhanced false positive,
    # set label to upres ctr
    elif abbr == 'x2f': label = 'upres ctrl'
    # return the label based on input of key
    return label


def filter_results(exp_res):
    '''
        remove the experimental category
        specified by function
    '''
    # init list of sub directory files at path
    sub_dir = dir_list('img/')
    # get list of experimental conditions without ground category
    conditions = gen_exp()[1:]
    # bool if subject must be removed
    remove_subject = False
    # iterate through every file which represents subject
    # in specified file directory
    for sub in sub_dir:
        # size of image for enhanced res categories
        x2size = exp_res[sub]['x2']['size']
        # size of image for low res categories
        lowsize = exp_res[sub]['low']['size']
        # if size greater than preselected value
        if x2size > 4000:
            # print to terminal that value was removed
            print(sub,'of size',x2size,'deleted')
            # delete exp res if size greater than preselected value
            del exp_res[sub]
        # if size is less than preselected value
        else:
            # print that val was kept to terminal
            print(sub,'size',x2size,'kept')


def filter_match_low(exp_res):
    '''
        filter out subjects wherein a match was
        not found in data structure list
    '''
    # init list of sub directory files at path
    sub_dir = dir_list('img/')
    # get list of experimental conditions without ground category
    conditions = gen_exp()[1:]
    # iterate through every file which represents subject
    # in specified file directory
    for sub in sub_dir:
        # get bool representing if match between
        # training image and low res image
        low_match = exp_res[sub]['low']['match']
        # condition if match is true
        if low_match:
            # print subject if it was deleted
            print(sub,'low is recognized, deleted')
            # delete the subject at the sub key
            del exp_res[sub]
        # condition if match not true
        else:
            # print subject if it was deleted
            print(sub,'low unrecognized, kept')



def display_data(stat,suffix):
    '''
        output graphs representing data found previously
    '''
    # get list of experimental conditions without ground category
    exp_dir = gen_exp()[1:]
    # get list of measurement criteria
    criteria = gen_criteria()
    # get list of matches
    matches = criteria_mean(stat,exp_dir,'match')
    # get list of magnitude of similarity
    distance = criteria_mean(stat,exp_dir,'dist')
    # get list of size of faces
    size = criteria_mean(stat,exp_dir,'size')
    # get list of err for matches
    err_matches = criteria_std(stat,exp_dir,'match')
    # get list of err for magnitude of matches
    err_distance = criteria_std(stat,exp_dir,'dist')
    # get list of err for size of faces
    err_size = criteria_std(stat,exp_dir,'size')
    # iterate through list of distance values
    for i,e in enumerate(distance):
        # invert values of magnitude
        distance[i] = 1.0-e
    # create labels for graph from predefined labels
    labels = [get_label(x) for x in exp_dir]
    # get number of positions for x axis
    x_pos = np.arange(len(exp_dir))
    # create new matplotlib figure
    plt.figure()
    # init bar chart with x-axis labels by y-axis as magnitude
    plt.bar(x_pos,distance,align='center',alpha=0.5,yerr=err_distance)
    # set ticks for x-axis
    plt.xticks(x_pos,labels)
    # label y axis title
    plt.ylabel('vector distance')
    # set title of graph
    plt.title('threshold across groups'+suffix)
    # create new matplotlib figure
    plt.figure()
    # init bar chart with x-axis labels by y-axis as rate matches
    plt.bar(x_pos,matches,align='center',alpha=0.5,yerr=err_matches)
    # set ticks for x-axis
    plt.xticks(x_pos,labels)
    # label y axis title
    plt.ylabel('proportion of matches')
    # set title of graph
    plt.title('matches across groups'+suffix)
    # create new matplotlib figure
    plt.figure()
    # init bar chart with x-axis labels by y-axis as size of faces
    plt.bar(x_pos,size,align='center',alpha=0.5,yerr=err_size)
    # set ticks for x-axis
    plt.xticks(x_pos,labels)
    # label y axis title
    plt.ylabel('sizes')
    # set title of graph
    plt.title('size across groups'+suffix)
    # show the matplotlib graphs
    plt.show()


def display_match_filter_data(stat):
    '''
        output graphs representing data found previously
    '''
    # get list of experimental conditions without ground category
    exp_dir = gen_exp()[1:]
    # get list of measurement criteria
    criteria = gen_criteria()
    # get list of matches
    matches = criteria_mean(stat,exp_dir,'match')
    # get magnitude of matching
    distance = criteria_mean(stat,exp_dir,'dist')
    # get std dev of matches
    err_matches = criteria_std(stat,exp_dir,'match')
    # get std dev of magnitude of matches
    err_distance = criteria_std(stat,exp_dir,'dist')
    # iterate through distances
    for i,e in enumerate(distance):
        # invert value of distances of match between faces
        distance[i] = 1.0-e
    # generate list of labels for graph
    labels = [get_label(x) for x in exp_dir]
    # get x posiiton
    x_pos = np.arange(len(exp_dir))
    # create new matplotlib figure
    plt.figure()
    # init bar chart with x-axis labels by y-axis as magnitude matching
    plt.bar(x_pos,distance,align='center',alpha=0.5,yerr=err_distance)
    # set ticks for x-axis
    plt.xticks(x_pos,labels)
    # label y axis title
    plt.ylabel('vector distance')
    # set title of graph
    plt.title('threshold across groups, filtered by unrecognized')
    # create new matplotlib figure
    plt.figure()
    # init bar chart with x-axis labels by y-axis as rate of matches
    plt.bar(x_pos,matches,align='center',alpha=0.5,yerr=err_matches)
    # set ticks for x-axis
    plt.xticks(x_pos,labels)
    # label y axis title
    plt.ylabel('proportion of matches')
    # set title of graph
    plt.title('matches across groups, filtered by unrecognized')
    # show the matplotlib graph
    plt.show()


def driver():
    '''
        driver of program
    '''
    # run the experiment of matches
    exp_res = run_experiment()
    # copy results
    exp_res_filter = deepcopy(exp_res)
    # filter out results based on size
    filter_results(exp_res_filter)
    # copy experimental results
    exp_res_match_filter = deepcopy(exp_res)
    # filter out images based on matching
    # if low res, remove from data dict
    filter_match_low(exp_res_match_filter)
    # change back to compile_data if you want
    sum_stat = compile_filtered_data(exp_res)
    # display statistics with matplotlib plots
    display_data(sum_stat,'')
    # compile the data filtered by size
    sum_stat_filter = compile_filtered_data(exp_res_filter)
    # display statistics with matplotlib plots
    display_data(sum_stat_filter,' filtered')
    # filtered by matchiness
    sum_stat_match = compile_filtered_data(exp_res_match_filter)
    # display statistics filtered by matches with matplotlib
    display_match_filter_data(sum_stat_match)



if __name__ == '__main__':
    # entry point of the program
    driver()



# end of file
