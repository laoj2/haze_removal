import heapq as hp

import numpy as np
import cv2

def get_dark_channel (img_in, patch_size=15):

    rows, cols, channels = np.shape(img_in)

    img_min = np.empty(shape=[rows,cols])

    #Minimum value among the channels
    for i in range (0, rows):
        for j in range (0, cols):
            img_min[i,j] = min([img_in[i,j,0], img_in[i,j,1], img_in[i,j,2]])

        img_min = img_min.astype(np.uint8)

    #Minimum filter
    strel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size,patch_size))
    img_dark = cv2.erode(img_min,strel)

    return img_dark

def get_atmospheric_light (rgb_img_in, dark_img_in):
    rows, cols = np.shape(dark_img_in)
    pixels_amount = int(0.001*3*rows*cols)

   # print pixels_amount

    heap = []

    for i in range (0, rows):
        for j in range (0, cols):
            hp.heappush(heap,(dark_img_in[i,j],i,j))

            if len(heap) > pixels_amount:
                hp.heappop(heap)

    max_intensity = [0,0,0]
    for i in range (0,len(heap)):
        _,x,y = heap[i]

        intensity = rgb_img_in[x,y,0] + rgb_img_in[x,y,1] + rgb_img_in[x,y,2]
        if max_intensity[0] < intensity:
            max_intensity = [intensity,x,y]

    #_,x,y = max_intensity
    #dark_img_in[x,y] = 0


    return max_intensity


def main():

    img_in = cv2.imread("images/cityscape.png")

    dark_channel = get_dark_channel(img_in)

    #print np.mean(img_dark)

    atmospheric_light = get_atmospheric_light(img_in, dark_channel)

    cv2.imshow("Image", dark_channel)

    cv2.waitKey(0)



main()

