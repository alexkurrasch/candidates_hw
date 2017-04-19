#!/usr/bin/python 
#
# Relativity Space - Problem 1 protoype code.  
# See problem1.py for final solution
#
# Alexander James Kurrasch 
# alex.kurrasch@gmail.com
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

# the frame is rotated and cropped to the region of interest.
frame = cv2.imread('./frames/00000100.png',0)
rotated_frame = imutils.rotate_bound(frame, 87.5)
cropped_frame  = rotated_frame[320:620, 338:398] 

IMAGE_WIDTH = cropped_frame.shape[1]
IMAGE_HEIGHT = cropped_frame.shape[0]

# approximate center of the weld.  This will be used to remove noise
WELD_CENTER = IMAGE_WIDTH / 2

# y values in the image less than this will be ignored 
# (upper portion of cropped image)
MOLTEN_ZONE = 100 

# group the images into regions around WELD_CENTER
REGIONS = 3
REGION_WIDTH = WELD_CENTER / REGIONS
REGION_OF_INTEREST = 1

# remove noise and find edges 
img = cv2.bilateralFilter(cropped_frame,9,75,75) 
edges = cv2.Canny(img,1,20)
contours,hierarchy  = cv2.findContours(edges.copy(),cv2.RETR_TREE,  
                                   cv2.CHAIN_APPROX_SIMPLE)

# remove contours which cross WELD_CENTER 
possible_contours = []
for contour in contours:
    if not ((contour[:,0,0].min() < WELD_CENTER) and 
            (contour[:,0,0].max() > WELD_CENTER)): 
        possible_contours.append(contour)
possible_contours = np.array(possible_contours)

# group all contours by symmetry around WELD_CENTER.  only contours in 
# REGION_OF_INTEREST will be used for the width computation
def x_region(xval):
    average_x = abs(xval - WELD_CENTER)
    for region in np.arange(0,REGIONS):
        lower = (region) * REGION_WIDTH 
        upper = (region+1) * REGION_WIDTH
        if ((lower <= average_x) and (average_x < upper)):
            return region
regions = {}
for r in range(REGIONS):
    regions[r] = []
for contour in possible_contours: 
    region = x_region(contour[:,0,0].mean()) 
    regions[region].append(contour)

# combine contours in REGION_OF_INTEREST 
#  to make 2 vectors for left and right side of weld 
left_edge  = np.zeros(IMAGE_HEIGHT)
right_edge = np.zeros(IMAGE_HEIGHT)
for contour in regions[REGION_OF_INTEREST]:
    min_y = np.vstack(contour)[:,1].min() 
    max_y = np.vstack(contour)[:,1].max() 
    x_vec = np.zeros(300)
    
    #right side 
    if (contour[:,0,0].mean() > WELD_CENTER):
        for point in np.vstack(contour):
            x_val = point[0]
            y_val = point[1]
            if (x_vec[y_val]==0):
                x_vec[y_val] = x_val
            else:
                x_vec[y_val] = (x_val + x_vec[y_val])  / 2.0
        # create continous vector from intervals in contour 
        # average if necessary
        for y in range(min_y,max_y+1):
            if (x_vec[y]==0):
                x_vec[y]=x_vec[y-1]
            if (right_edge[y]==0):
                right_edge[y] = x_vec[y]
            else:
                 right_edge[y] = (x_vec[y] + right_edge[y])  / 2.0
    # left side
    else:
        for point in np.vstack(contour):
            x_val = point[0]
            y_val = point[1]
            if (x_vec[y_val]==0):
                x_vec[y_val] = x_val
            else:
                x_vec[y_val] = (x_val + x_vec[y_val])  / 2.0
        for y in range(min_y,max_y+1):
            if (x_vec[y]==0):
                x_vec[y]=x_vec[y-1]
            if (left_edge[y]==0):
                left_edge[y] = x_vec[y]
            else:
                 left_edge[y] = (x_vec[y] + left_edge[y])  / 2.0

# width calculation.  only computed at points where both vectors are non-zero
# output is given as: 
# (mean width, variance, percentage of data points used in calculation)
def width(vec1, vec2, yvals):
    width = []
    for y in yvals:
        if ((vec1[y] != 0) and (vec2[y] != 0)):
            width.append(abs(vec1[y] - vec2[y]))
    width = np.array(width)
    return (width.mean(), width.var(), float(len(width))/float(len(yvals)))

print "---"
print "Region - (mean width, variance, percentage of data points used)"
print "Region 0 - " + str(width(right_edge, left_edge, range(0,100)))  
print "Region 1 - " + str(width(right_edge, left_edge, range(100,200)))  
print "Region 2 - " + str(width(right_edge, left_edge, range(200,300)))  

cv2.drawContours(cropped_frame, regions[REGION_OF_INTEREST], -1, (0, 255, 0), 1)

plt.subplot(121),plt.imshow(cropped_frame,cmap = 'gray')
plt.title('original with detected weld edges'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('all detected edges'), plt.xticks([]), plt.yticks([])

plt.figure()
plt.plot(right_edge,'go')
plt.plot(left_edge,'ro')
plt.plot(right_edge - left_edge,'bo')
plt.legend(('right edge','left edge','width'))

plt.show()
