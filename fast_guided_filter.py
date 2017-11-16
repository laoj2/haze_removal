import cv2
import numpy as np


def fast_guided_filter(I, p, r, upsampling_ratio, epsilon):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))

    corr_I = cv2.boxFilter(I*I, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))

    var_I = corr_I - mean_I*mean_I
    cov_Ip = corr_Ip - mean_I*mean_p

    a = cov_Ip/(var_I + epsilon)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    return mean_a, mean_b


def fast_matting (hazy_img, transmission_map, subsampling_ratio=4, r=20, epsilon=0.001):

    subsampling_ratio = 1.0/subsampling_ratio
    gray = cv2.cvtColor(hazy_img,cv2.COLOR_BGR2GRAY)

    I = cv2.resize(gray, dsize=(0,0), fx=subsampling_ratio, fy=subsampling_ratio)
    p = cv2.resize(transmission_map, dsize=(0,0), fx=subsampling_ratio, fy=subsampling_ratio)
    r = int(r*subsampling_ratio)

    mean_a, mean_b = fast_guided_filter(I/255.0, p, r, 1/subsampling_ratio, epsilon)

    x,y,_ = np.shape(hazy_img)
    mean_a_up = cv2.resize(mean_a,dsize=(y,x))
    mean_b_up = cv2.resize(mean_b,dsize=(y,x))

    q = mean_a_up*(gray/255.0) + mean_b_up

    return q
