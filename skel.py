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


def main():

    image_in = cv2.imread("images/d.webp")

    img_dark = get_dark_channel(image_in)

    cv2.imshow("Image", img_dark)

    cv2.waitKey(0)



main()

