Devin Bresser - ECE766 HW04 - Readme notes

All program designs are exclusively my own. I did discuss problem 1e conceptually with 
my classmates Keshav Sharan Pachipala and Nico Ranabhat, but no code was provided or copied.
All lines of code are written exclusively by me and I can explain every line.
ChatGPT was used to assist with Python syntax, NumPy operations, and debugging.

Challenge 1a:
No major comments, fairly straightforward implementation of homography using the formulas in the lecture videos.
Source and destination points were selected using pixspy.com.

Challenge 1b:
No major comments. Backward warp implementation based upon discussion of homogenous coordinates, applying homography/inverse homography, and dehomogenization in the lecture videos.
I used the manual corner points for the billboard from this piazza post:
https://piazza.com/class/lrqx4k5vfymnq/post/131

Challenge 1c:
No major comments. RANSAC algorithm implemented per lecture video.

Challenge 1d:
No major comments. Blend and overlay implemented per lecture video. 
Blend method uses distance_transform_edt from scipy.ndimage.

Challenge 1e:
This function was by far the most challenging part of this assignment to complete.
The hardest part was keeping track of the canvas size. I found the approach of pre-computing the maximum canvas size to work best. It is a slightly inefficient operation but it works.
Thanks to this piazza student response: https://piazza.com/class/lrqx4k5vfymnq/post/154
for helping me fine tune my method to make it work.

Challenge 1f:
I took three images of my desk (where I completed this homework) plus my sleeping cat. As per above, we can see that my homography, RANSAC, and warping are working, and my image stitching method works reasonably well on the real world images.


