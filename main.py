import argparse
from skel import *


parser = argparse.ArgumentParser(description='Display image depth map')
parser.add_argument('image_path',type=str)
args = parser.parse_args()


hazy_img = cv2.imread(args.image_path)


dark_channel = get_dark_channel(hazy_img,15).astype(np.uint8)

atmospheric_light = get_atmospheric_light(hazy_img, dark_channel)

transmission_map = get_transmission_map(hazy_img, atmospheric_light,patch_size=15)

radiance = get_scene_radiance(hazy_img, atmospheric_light, transmission_map)

depth_map = np.copy(hazy_img)

depth_map[:,:,1] = get_depth_map(hazy_img,transmission_map)
depth_map[:,:,0] = ~depth_map[:,:,1]


#depth_map[:,:,0] = transmission_map*255# get_depth_map(hazy_img,transmission_map)
#depth_map[:,:,2] = ~depth_map[:,:,1]

cv2.namedWindow("depth",cv2.WINDOW_AUTOSIZE)
#cv2.imshow("depth", (transmission_map*255).astype(np.uint8))
cv2.imshow("depth", get_depth_map(hazy_img,transmission_map))

cv2.namedWindow("radiance",cv2.WINDOW_AUTOSIZE)
cv2.imshow("radiance", radiance)

cv2.waitKey(0)