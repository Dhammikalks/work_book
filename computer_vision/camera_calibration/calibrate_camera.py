#!/bin/python -f
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

images = glob.glob('./calibration_images/calibration*.jpg')
#plt.imshow(img)
objpoints = []
imgpoints = []

objp = np.zeros((6*8,3),np.float32)

objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

for fname in images:
    print("got image")
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        img = cv2.drawChessboardCorners(img,(8,6), corners, ret)
img = cv2.imread('./calibration_images/test_image.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=50)
#ax2.imshow(dst)
#ax2.set_title('Undistorted Image', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.show()

img = cv2.imread('./calibration_images/test_image2.png')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
undistort = cv2.undistort(img, mtx, dist, None, mtx)

img_size = (undistort.shape[1] , undistort.shape[0])
src = np.float32(
[[966,206],
[962,305],
[884,300],
[887,192]
]
)
dst = np.float32(
[[987,192],
[987,292],
[887,292],
[887,192]
])

M = cv2.getPerspectiveTransform(src,dst)
warped = cv2.warpPerspective(undistort,M,img_size,flags=cv2.INTER_LINEAR)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(warped)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
