import SimpleITK as sitk
import numpy as np
import cv2 as cv
import os
import skimage.io as io
from natsort import natsorted
from tqdm import tqdm
IN = "./training"
OUT = "./camus-images"

def mhd2jpg(mhdPath, outFolder):
    img = io.imread(mhdPath, plugin='simpleitk')
    img = np.reshape(cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U), (img.shape[1], img.shape[2], img.shape[0]))
    file_name = os.path.basename(mhdPath)
    
    try:
        if "_gt" in file_name:
            mask = "mask"
            if not os.path.exists(os.path.join(outFolder, mask)):
                os.mkdir(os.path.join(outFolder, mask))
            cv.imwrite(os.path.join(outFolder, mask, f"{file_name[:-4]}.jpg"), img)
        else:
            image = "image"
            if not os.path.exists(os.path.join(outFolder, image)):
                os.mkdir(os.path.join(outFolder, image))
            cv.imwrite(os.path.join(outFolder, image, f"{file_name[:-4]}.jpg"), img)
    except:
        print(f"Error with: {file_name}")

def main():
    folders = natsorted(filter(lambda d: "patient" in d, os.listdir(IN)))

    for folder in tqdm(folders):
        # print(folder)
        files = natsorted(filter(lambda f: ".mhd" in f and "_sequence" not in f, os.listdir(os.path.join(IN, folder))))
        for file in files:
            mhd2jpg(os.path.join(IN, folder, file), OUT)



if __name__ == "__main__":
    main()
