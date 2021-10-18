# import all necessary packages
import numpy as np
import cv2

#Here We use OpenCV To Read the input image
img = cv2.imread('E:\\MLR\\bd.png')
#Coverts the input image into Grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 9)
_filter = cv2.bilateralFilter(blurred, 5, 75, 75)
adap_thresh = cv2.adaptiveThreshold(_filter,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21, 0)

element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
dilated = cv2.dilate(adap_thresh,element, iterations=1)

# blob detection by different parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False 
params.minThreshold = 75
params.maxThreshold = 100
params.blobColor = 0
params.minArea = 208
params.maxArea = 5000
params.filterByCircularity = True
params.filterByConvexity = False
params.minCircularity =.4
params.maxCircularity = 1

det = cv2.SimpleBlobDetector_create(params)
keypts = det.detect(dilated)


im_with_keypoints = cv2.drawKeypoints(dilated,keypts,np.array([]),(0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


res = cv2.drawKeypoints(img,keypts,np.array([]),(0, 0, 255 ),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

i = 0
for kp in keypts:
    print("(%f,%f)"%(kp.pt[0],kp.pt[1]))
    i+=1
    cv2.rectangle(res,(int(kp.pt[0]),int(kp.pt[1])),(int(kp.pt[0])+1,int(kp.pt[1])+1),(0,255,0),2)

# Get the count of the blobs are there in the image 
print("Number of blobs detected are :", len(keypts))

#Showing the Image Blob detected image
cv2.imshow("Blob Detection Result Image", res)
cv2.waitKey(0)
