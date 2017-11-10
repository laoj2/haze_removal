import heapq as hp

import numpy as np
import cv2
from numpy.ma.core import log2


def get_dark_channel (img_in, patch_size=15):

    rows, cols, channels = np.shape(img_in)

    img_min = np.empty(shape=[rows,cols])

    #Minimum value among the channels
    for i in range (0, rows):
        for j in range (0, cols):
            img_min[i,j] = min([img_in[i,j,0], img_in[i,j,1], img_in[i,j,2]])

    #Minimum filter
    strel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size,patch_size))
    img_dark = cv2.erode(img_min,strel)

    return img_dark

def get_atmospheric_light (rgb_img_in, dark_img_in):
    rows, cols = np.shape(dark_img_in)
    pixels_amount = int(0.001*3*rows*cols)

    heap = []

    for i in range (0, rows):
        for j in range (0, cols):
            hp.heappush(heap,(dark_img_in[i,j],i,j))

            if len(heap) > pixels_amount:
                hp.heappop(heap)

    max_intensity = [0,0,0]
    for i in range (0,len(heap)):
        _,x,y = heap[i]

        intensity = int(rgb_img_in[x,y,0]) + int(rgb_img_in[x,y,1]) + int(rgb_img_in[x,y,2])
        if max_intensity[0] < intensity:
            max_intensity = [intensity,x,y]

    _,x,y = max_intensity
    return rgb_img_in[x,y]

def get_transmission_map (hazy_img, atmospheric_light, w=0.95, patch_size=15):

    normalized_hazy_img = hazy_img.astype(np.float32)/atmospheric_light.astype(np.float32)

    transmission_map = 1 - w*get_dark_channel(normalized_hazy_img,patch_size)

    return transmission_map

def get_scene_radiance(hazy_img, atmospheric_light, transmission_map, t0=0.1):
    x,y,ch = np.shape(hazy_img)

    radiance = hazy_img
#994929797
    for i in range (0,x):
        for j in range (0,y):
            for c in range (0,ch):
                t = max(transmission_map[i,j].astype(np.float32),t0)
                radiance[i,j,c] = atmospheric_light[c] + (hazy_img[i,j,c].astype(np.float32) - atmospheric_light[c])/t

    return  radiance

def get_depth_map (hazy_img, transmission_map, beta=3):
    depth = np.log(transmission_map)/-beta
    depth = depth*255


    return depth.astype(np.uint8)


def main():

    hazy_img = cv2.imread("images/cityscape.png")

    dark_channel = get_dark_channel(hazy_img).astype(np.uint8)


    atmospheric_light = get_atmospheric_light(hazy_img, dark_channel)

    transmission_map = get_transmission_map(hazy_img, atmospheric_light)


    radiance = get_scene_radiance(hazy_img, atmospheric_light,transmission_map)

    #cv2.imshow("Image", get_depth_map(hazy_img,transmission_map))

    #cv2.waitKey(0)



#main()

