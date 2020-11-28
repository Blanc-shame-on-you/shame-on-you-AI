import os
import cv2 
import numpy 
import os 

def bitOperation(locations,img1):
  img2 = cv2.imread(os.getcwd().replace('\\','/')+'/models/virus.png')
  for x1,y1,x2,y2 in locations:
    if x2-x1 > y2-y1:
      #img2 = cv2.resize(img2, dsize=(x1-x2, x1-x2), interpolation=cv2.INTER_AREA)
      img2 = cv2.resize(img2, dsize=(x2-x1,x2-x1), interpolation=cv2.INTER_AREA)
    else:
      img2 = cv2.resize(img2, dsize=(y2-y1, y2-y1), interpolation=cv2.INTER_AREA)

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    rows, cols = img2.shape[:2] # 바이러스 이미지의 가로, 세로 

    vpos = y1
    hpos = x1    

    roi = img1[vpos:rows+vpos, hpos:cols+hpos]

    img1_bg = cv2.bitwise_and(roi,roi, mask=mask_inv)


    dst = cv2.add(img1_bg, img2_fg)
    img1[vpos:rows+vpos, hpos:cols+hpos] = dst    

  return img1
