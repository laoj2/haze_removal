import os

import argparse
from haze_removal import *
from fast_guided_filter import fast_matting
from soft_matting import *

parser = argparse.ArgumentParser(description='Display image depth map')
parser.add_argument('images_path', type=str)
parser.add_argument('patch_size', type=int)
args = parser.parse_args()

#image = args.images_path
#for image in os.listdir(args.images_path):


hazy_img = cv2.imread(args.images_path)
dark_channel = get_dark_channel(hazy_img, patch_size=args.patch_size).astype(np.uint8)
atmospheric_light = get_atmospheric_light(hazy_img, dark_channel)

transmission_map_old = get_transmission_map(hazy_img, atmospheric_light,patch_size=args.patch_size)
transmission_map = fast_matting(hazy_img,transmission_map_old)

radiance = get_scene_radiance(hazy_img, atmospheric_light, transmission_map)
depth_map = get_depth_map(hazy_img,transmission_map,beta=np.e)
depth_map = cv2.applyColorMap(depth_map,cv2.COLORMAP_HOT)

image = args.images_path.split('/')[1]

cv2.imwrite('dark_channel_' + image, dark_channel)
cv2.imwrite('transmission_map_' + image, float_image_to_uint(transmission_map_old*255.0))
cv2.imwrite('refined_transmission_map_' + image, float_image_to_uint(transmission_map*255.0))
cv2.imwrite('depth_map_' + image, depth_map)
cv2.imwrite('radiance_' + image,radiance)

#cv2.imshow("depth", (transmission_map*255).astype(np.uint8))
#cv2.imshow("depth", transmission_map)

#cv2.namedWindow("radiance",cv2.WINDOW_AUTOSIZE)
#cv2.imshow("radiance", radiance)

#cv2.waitKey(0)