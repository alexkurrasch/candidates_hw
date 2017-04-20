## Problem 1 - Alex Kurrasch
The solution to Problem 1 can be found in problem1.py

### Overview
The present solution achieves all 5 points in the problem description.  The unit of measurement is 'floating point pixels' which can be scaled to match physical units.

Several assumptions were made when solving the problem:
 * distance from camera to work is fixed
 * the image is orthographic (more or less)
 * the program only needs to function with the provided file 'weld.mp4'
  
The image is rotated and cropped to the region of interest.  This region is divided into 3 vertical (y) and horizontal (x) regions.  The width of the weld bead can be calculated in all 3 vertical regions, but only the 2nd region is output.  A simple algorithm is used to keep the weld bead centered from frame to frame.  

In addition to the width, a measure of the standard deviation and percentage of measurement points used in the calculation are provided.  This allows for one to weight the output data.  

### Potential improvements to solution 
  * From looking at the frames in the video it looks like the work advances in front of the camera at a rate of approximately 100 px   per frame.  If this is the case then 2 measurements can be taken for the same physical point using subsequent frames.  
 * The algorithm to track the weld bead functions, but leaves plenty of room for improvement.  A linear prediction filter could be an option.  Also it is very costly to recompute the contours for each iteration of the optimization.  
  * OpenCV was used for contour generation.  A more suitable contour generation algorithm could be developed specific for this problem which is constrained to linear features.  
  * There are edge effects at the beginning and end of the downsampled output file.  This is typical of such downsampling filters, but nonetheless undesirable.  
  * If the input file was larger, it would be desirable to write the code using a stream processing paradigm.
