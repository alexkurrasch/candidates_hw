#!/usr/bin/python 
#
# Relativity Space - Problem 1   
# Final solution
# see README.md for description 
#
# Alexander James Kurrasch 
# alex.kurrasch@gmail.com
import cv2
import imutils
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt

class WeldBeadWidth:
    """
    Measure weld bead width from image frame .
    this class is only designed for input from the provided test file
    """
    def __init__(self):
        self.frame_number = 0 

        self.weld_center = 30
        self.image_height = 300
        self.image_width = 60

        # these paramters are updated between frames to keep the weld 
        # bead centered in the image 
        self.rotation_angle = 87.5
        self.y_pos = 320
        self.x_pos = 338 
        
        # group the images into x-regions around self.weld_center
        self.num_xregions = 3
        self.xregion_width = self.weld_center / self.num_xregions 
        self.xregion_of_interest = 1

        # 3 y-regions in which the width of the bead is calculated 
        self.yregion_0 = range(0,100)
        self.yregion_1 = range(100,200)
        self.yregion_2 = range(200,300)

        self.left_edge  = np.zeros(self.image_height)
        self.right_edge = np.zeros(self.image_height)


    def __str__(self):
        return str("---" + 
                   "\nY Region   - (width, variance, percent of data points used)" +  
                   "\nY Region 0 - " + str(self.__width(self.right_edge, 
                                                        self.left_edge, 
                                                        self.yregion_0 )) + 
                   "\nY Region 1 - " + str(self.__width(self.right_edge, 
                                                        self.left_edge, 
                                                        self.yregion_1 )) +
                   "\nY Region 2 - " + str(self.__width(self.right_edge, 
                                                        self.left_edge, 
                                                        self.yregion_2 )))


    def __x_region(self, xval):
        """
        group all contours by symmetry around self.weld_center.  only contours in 
        self.region_of_interest will be used for the width computation        
        """
        average_x = abs(xval - self.weld_center)
        for region in np.arange(0,self.num_xregions):
            lower = (region) * self.xregion_width 
            upper = (region+1) * self.xregion_width 
            if ((lower <= average_x) and (average_x < upper)):
                return region


    def __width(self, vec1, vec2, yvals):
        """
        width calculation.  only computed at points where both vectors are non-zero
        output is given as: 
        (mean width, variance, percentage of data points used in calculation)
        """
        width = []
        for y in yvals:
            if ((vec1[y] != 0) and (vec2[y] != 0)):
                width.append(abs(vec1[y] - vec2[y]))
        width = np.array(width)
        return (width.mean(), width.var(), float(len(width))/float(len(yvals)))


    def __fit(self):
        """
        linear least squares fit to edges.  
        used to minimize slope and distance from center
        """

        # using y_region 1 and 2 to center data as there is less
        # noise away from the  welder head 
        yl, yr = [], []
        l, r = [], []
        for i in range(100, self.image_height):
            if (self.right_edge[i] != 0):
                yr.append(i)
                r.append(self.right_edge[i])
            if (self.left_edge[i] != 0):
                yl.append(i)
                l.append(self.left_edge[i])

        lfit = np.polyfit(yl,l,1)
        rfit = np.polyfit(yr,r,1)
        return [lfit, rfit]


    def __measure(self, frame, save_frame=False):

        self.left_edge  = np.zeros(self.image_height)
        self.right_edge = np.zeros(self.image_height)

        rotated_frame = imutils.rotate_bound(frame, self.rotation_angle)
        cropped_frame  = rotated_frame[self.y_pos:self.y_pos+self.image_height, 
                                       self.x_pos:self.x_pos+self.image_width] 

        # remove noise and find edges 
        img = cv2.bilateralFilter(cropped_frame,9,75,75) 
        edges = cv2.Canny(img,1,20)
        contours,hierarchy  = cv2.findContours(edges.copy(),cv2.RETR_TREE,  
                                               cv2.CHAIN_APPROX_SIMPLE)
        
        # classify contours into regions around weld center
        regions = {}
        for r in range(self.num_xregions):
            regions[r] = []
        for contour in contours: 
            region = self.__x_region(np.median(contour[:,0,0]))
            regions[region].append(contour)
 
        # combine contours in self.x_region_of_interest 
        #  to make 2 vectors for left and right side of weld 
        for contour in regions[self.xregion_of_interest]:
            min_y = np.vstack(contour)[:,1].min() 
            max_y = np.vstack(contour)[:,1].max() 
            x_vec = np.zeros(300)

            # remove portions of contour which are outside self.xregion_of_interest
            pruned_contour = []
            for point in np.vstack(contour):
                x_val = point[0]
                y_val = point[1]
                if (self.__x_region(x_val) == self.xregion_of_interest):
                    pruned_contour.append([x_val, y_val])
                

            #right side 
            if (contour[:,0,0].mean() > self.weld_center):
                for point in pruned_contour:
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
                    if (self.right_edge[y]==0):
                        self.right_edge[y] = x_vec[y]
                    else:
                         self.right_edge[y] = (x_vec[y] + self.right_edge[y])  / 2.0
            # left side
            else:
                for point in pruned_contour:
                    x_val = point[0]
                    y_val = point[1]
                    if (x_vec[y_val]==0):
                        x_vec[y_val] = x_val
                    else:
                        x_vec[y_val] = (x_val + x_vec[y_val])  / 2.0
                for y in range(min_y,max_y+1):
                    if (x_vec[y]==0):
                        x_vec[y]=x_vec[y-1]
                    if (self.left_edge[y]==0):
                        self.left_edge[y] = x_vec[y]
                    else:
                         self.left_edge[y] = (x_vec[y] + self.left_edge[y])  / 2.0
        
        # for development
        if (save_frame):
            cv2.drawContours(cropped_frame, regions[0], -1, (0, 255, 0), 1)
            cv2.drawContours(cropped_frame, regions[self.xregion_of_interest], -1, (255, 0, 0), 1)
            cv2.drawContours(cropped_frame, regions[2], -1, (0, 0, 255), 1)
            plt.subplot(121),plt.imshow(cropped_frame,cmap = 'gray')
            plt.title('original with detected weld edges'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(edges,cmap = 'gray')
            plt.title('all detected edges'), plt.xticks([]), plt.yticks([])
            plt.savefig("out-"+str(self.frame_number) + ".png")


    def get_yregion_0(self):        
        return self.__width(self.right_edge, self.left_edge, self.yregion_0)


    def get_yregion_1(self):        
        return self.__width(self.right_edge, self.left_edge, self.yregion_1)


    def get_yregion_2(self):        
        return self.__width(self.right_edge, self.left_edge, self.yregion_2)


    def run(self, frame):
        
        self.frame_number += 1

        self.__measure(frame)

        # update rotation and x position to center weld bead
        fit = self.__fit()
        ml, bl = fit[0]
        mr, br = fit[1]

        center_error = abs(bl- self.weld_center) - abs(br - self.weld_center)
        center_direction = np.sign((bl- self.weld_center) - (br - self.weld_center))
        
        rotation_error = np.arctan((ml + mr) / 2.0) * 360.0 / np.pi
        total_error = abs(center_error) + abs(rotation_error)        

        # simple but sltracking algorithm which tries to center the region of interest
        for i in range(10):        
            self.x_pos = self.x_pos + int( ((center_error / 2 )*center_direction) / 2.0)
            self.rotation_angle += np.sign(rotation_error) * 0.1  
            self.__measure(frame)
            fit = self.__fit()
            ml, bl = fit[0]
            mr, br = fit[1]
        
            center_error = abs(bl- self.weld_center) - abs(br - self.weld_center)
            center_direction = np.sign((bl- self.weld_center) - (br - self.weld_center))
            rotation_error = np.arctan((ml + mr) / 2.0)  * 360.0 / np.pi
            
            new_total_error = abs(center_error) + abs(rotation_error) 
        
            

            if (new_total_error > total_error):
                break
            total_error = new_total_error 

        #self.__measure(frame, save_frame=True)

     
################################################################################

wbw = WeldBeadWidth()
vid = cv2.VideoCapture('../videos/weld.mp4')

width = []
frame_count = 0
print "frame number: (width, variance, percent of data points used for calculation)"

while(vid.isOpened()):
    ret, frame = vid.read()
    if (type(frame) == type(None)):
        break

    wbw.run(frame)    
    frame_count += 1

    region_1 = wbw.get_yregion_1()
    print  str(frame_count) + ":\t" + str(region_1)
    width.append(region_1[0])

vid.release()
region_1 = np.array(region_1)

# frame rate is 29.97 Hz.  For 20 Hz output we need to 
# upsample 2000x then downsample 2997x using a polyphase filter
dt = 1.0 / 20.0
resampled_width = signal.resample_poly(width, 2000, 2997)
    
csv = open("weld_width.csv",'w')
t = 0
for point in resampled_width:
    csv.write(str(dt*t) + " , " + str(point) + "\n")
    t += 1
csv.close()






