#modified code from https://github.com/thecodacus/object-recognition-sift-surf
import cv2
import numpy as np
MIN_MATCH_COUNT=20

detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=cv2.imread("./ref_input/ref_test.jpg",0)
trainImg = cv2.resize(trainImg, (0,0), fx=1., fy=1.) 
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

import os, fnmatch

listOfFiles = os.listdir('./data/')  
pattern = "*.jpg"  
num_pic = 0
for entry in listOfFiles:  
    if fnmatch.fnmatch(entry, pattern):
        num_pic = num_pic + 1
        print "cropping ", num_pic, (entry)
        filename = os.path.splitext(entry)[0]

	image_screen = cv2.imread('./data/'+entry)
	image_screen = cv2.resize(image_screen, (0,0), fx=1., fy=1.) 

	QueryImgBGR = image_screen

	QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
	queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
	matches = flann.knnMatch(queryDesc, trainDesc, k=2)

	goodMatch = []
	for m, n in matches:
	    if(m.distance < 0.75 * n.distance):
		goodMatch.append(m)
	if(len(goodMatch) >= MIN_MATCH_COUNT):
	    tp = []
	    qp = []
	    for m in goodMatch:
		tp.append(trainKP[m.trainIdx].pt)
		qp.append(queryKP[m.queryIdx].pt)
	    tp, qp = np.float32((tp, qp))
	    H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
	    h, w = trainImg.shape
	    trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
	    queryBorder = cv2.perspectiveTransform(trainBorder, H)
	    # cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
	    bottomRight = (max(queryBorder[0][:, 0]), max(queryBorder[0][:, 1]))
	    topLeft = (min(queryBorder[0][:, 0]), min(queryBorder[0][:, 1]))
	    outputBorder = np.float32([[[topLeft[0], topLeft[1]], [bottomRight[0], topLeft[1]], [bottomRight[0], bottomRight[1]], [topLeft[0], bottomRight[1]], [topLeft[0], topLeft[1]]]])
		
	    #cv2.polylines(QueryImgBGR, np.int32([outputBorder]), 1, (255, 0, 0)) 
	    #crop result
            crop_img = image_screen[np.int32(topLeft[1]):np.int32(bottomRight[1]), np.int32(topLeft[0]):np.int32(bottomRight[0])]
            cv2.imwrite('./cropped/'+filename+'.jpg',crop_img)
            #cv2.imshow("cropped", crop_img)
	else:
	    print "Not Enough match found- %d/%d" % (len(goodMatch), MIN_MATCH_COUNT)

#read console information
cv2.waitKey(0) 
