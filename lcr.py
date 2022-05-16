import cv2
import imutils
import numpy as np
import pytesseract
from edgedetection import *
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

img = cv2.imread('car.JPG',cv2.IMREAD_COLOR)
img = imutils.resize(img, width=300)
cv2.imshow('original image', img)
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow('grayed image', gray)
cv2.waitKey(0)

gray = cv2.bilateralFilter(gray, 11, 17, 17) 
cv2.imshow('smoothened image', gray)
cv2.waitKey(0)

gray = edgeDetection(gray)
edged = cv2.Canny(gray, 30, 200) 
cv2.imshow('image edges', edged)
cv2.waitKey(0)

cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1 = img.copy()
cv2.drawContours(img1, cnts, -1, (0,255, 0), 3)
cv2.imshow('contours', img1)
cv2.waitKey(0)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
screenCnt = None
img2 = img.copy()
cv2.drawContours(img2, cnts, -1, (0,255, 0), 3)
cv2.imshow('top 30 contours', img2)
cv2.waitKey(0)

i = 7
for c in cnts:
    perim = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perim, True)
    if len(approx) == 4:
        screenCnt = approx

    x, y, w, h = cv2.boundingRect(c)
    new_img = img[y:y+h,x:x+w]
    cv2.imwrite('./'+str(i)+'.png', new_img)
    i+=1
    break

cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow('image with detected license plaete', img)
cv2.waitKey(0)

Cropped_loc = './7.png'
cv2.imshow('cropped', cv2.imread(Cropped_loc))
plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
print('Number plate is: ', plate)
cv2.waitKey(0)
cv2.destroyAllWindows()