

import argparse

from HSICubeBuilder import HSIImageHandler, CoordSys, HSIPixelCube, HSIPixelCubeVariation
from HSICubeBuilder import HSICubeTransformImageExtractor as CTE

import scipy.io as scp
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', help='image file')
parser.add_argument('--ground_truth', '-gt', help='ground truth file')
parser.add_argument('--save_path', '-sp', help='save path')
parser.add_argument('--window_size', '-ws', help='window size - odd number', type = int)
parser.add_argument('--transformations', '-t', help='transforms \'fv\' where f = function in (ro, sc, sh) and v = value', nargs = '+')

args = parser.parse_args()

data_file = args.image
gt_file = args.ground_truth

#image = scp.loadmat(data_file)['data']
#ground_truth = scp.loadmat(gt_file)['data']

HSI_IH = HSIImageHandler(data_file, gt_file, args.window_size, "HWDC")

TM = CTE.getIdentity()

for t in args.transformations:
	func = t[0:2]
	vals = t[2:].split('_')
	vals = [float(x) for x in vals]
	if func == 'ro':
		print('rotating:', vals[0])
		TM = CTE.addRotation(TM, np.deg2rad(vals[0]))
	if func == 'sc':
		print('scaling:', vals[0], vals[1])
		TM = CTE.addScale(TM, vals[0], vals[1])
	if func == 'sh':
		print('sheering:', vals[0], vals[1])
		TM = CTE.addSheer(TM, vals[0], vals[1])

HSI_IH.addTransformExtractor(TM)
start = time.time()
all_cubes = HSI_IH.populatePixels(False)
end = time.time()
print('total time:', end - start)
print('time/sample:', (end - start)/len(HSI_IH.labeled_pixels))

all_transforms = []
TMb = TM.tobytes()

for pixel in HSI_IH.labeled_pixels:
	all_transforms.append(pixel.cubes[TMb].getOriginal()[0])

all_transforms = np.stack(all_transforms, axis = 0)

print(all_transforms.shape)

mat = dict()
mat['data'] = all_transforms

scp.savemat(args.save_path, mat)
