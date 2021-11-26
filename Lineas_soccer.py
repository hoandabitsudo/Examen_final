# --------------------------------------------------------------------------------------------------------------------
# --------- Examen Final  --------------
# -------------------------------------------------------------------------------------------------------------------
# Juan David Venegas Sanabaria

# Segun lo expresado por correo, esta es la solucion del Examen final, teniendo en cuneta que no
# me funciono el correo al momento que usted envio el mensaje sobre la fecha del mismo.
# -------------------------------------------------------------------------------------------------------------------
# En este archivo se encuentra una parte de la solucion de la pregunta 3:

# 3) especificar una recta por mouse y generar una linea paralela a esta

# -------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
from enum import Enum

# Abrir imagen
image = cv2.imread('soccer_game.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_draw = np.copy(image)

# Harris
dst = cv2.cornerHarris(image_gray.astype(np.float32), 2, 3, 0.04)
dst = cv2.dilate(dst, None)
image_draw[dst > 0.01 * dst.max()] = [0, 0, 255]

# Shi-Tomasi
corners = cv2.goodFeaturesToTrack(image_gray, 100, 0.0001, 10)
corners = corners.astype(np.int)
for i in corners:
    x, y = i.ravel()
    cv2.circle(image_draw, (x, y), 3, [255, 0, 0], -1)

# sift and orb
sift = cv2.SIFT_create(nfeatures=100)
orb = cv2.ORB_create(nfeatures=100)

keypoints_sift, descriptors = sift.detectAndCompute(image_gray, None)
keypoints_orb, descriptors = orb.detectAndCompute(image_gray, None)
image_draw = cv2.drawKeypoints(image_gray, keypoints_orb, None)

cv2.imshow("Image", image_draw)
cv2.waitKey(0)

class Methods(Enum):
    SIFT = 1
    ORB = 2

image_1 = cv2.imread('soccer_game.png')
image_gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
image_draw_1 = np.copy(image_1)

image_2 = cv2.imread('soccer_game.png')
image_gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
image_draw_2 = np.copy(image_2)

# sift/orb interest points and descriptors
method = Methods.SIFT
if method == Methods.SIFT:
    sift = cv2.SIFT_create(nfeatures=100)   # shift invariant feature transform
    keypoints_1, descriptors_1 = sift.detectAndCompute(image_gray_1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_gray_2, None)
else:
    orb = cv2.ORB_create(nfeatures=100)     # oriented FAST and Rotated BRIEF
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_gray_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_gray_2, None)

image_draw_1 = cv2.drawKeypoints(image_gray_1, keypoints_1, None)
image_draw_2 = cv2.drawKeypoints(image_gray_2, keypoints_2, None)

# Interest points matching
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(descriptors_1, descriptors_2, k=1)
image_matching = cv2.drawMatchesKnn(image_1, keypoints_1, image_2, keypoints_2, matches, None)

# Retrieve matched points
points_1 = []
points_2 = []
for idx, match in enumerate(matches):
    idx2 = match[0].trainIdx
    points_1.append(keypoints_1[idx].pt)
    points_2.append(keypoints_2[idx2].pt)

# Compute homography and warp image_1
H, _ = cv2.findHomography(np.array(points_1), np.array(points_2), method=cv2.RANSAC)
image_warped = cv2.warpPerspective(image_1, H, (image_1.shape[1], image_1.shape[0]))

cv2.imshow("Image 1", image_1)
cv2.imshow("Image 2", image_2)
cv2.imshow("Image matching", image_matching)
cv2.imshow("Image warped", image_warped)
cv2.waitKey(0)