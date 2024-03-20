This repository contains a full implementation of a Panorama stitching app in Python. 
The pipeline uses manual (NumPy) implementations of Homography matrix computation, backward warping, the RANSAC outlier removal mechanism, amd image blending & overlaying via distance transform to stitch together an arbitrary number of images of the same scene into a single panorama.

The final panorama stitching program first computes the dimensions of a canvas that will fit the center image and all of the warped images.
It then backward warps all of the images onto blank images with the dimensions of the final canvas, and systematically blends each component image them together using the distance transform method to achieve the result.

Example inputs:
![mountain_left](https://github.com/devinbresser/ece766-panorama-app/assets/66394890/80f9133c-f27c-4b93-a323-3e47ccdb26df)

![mountain_center](https://github.com/devinbresser/ece766-panorama-app/assets/66394890/3e6e3dab-ae5a-4d97-aee6-22d825c18476)

![mountain_right](https://github.com/devinbresser/ece766-panorama-app/assets/66394890/938b1224-36db-4b43-bf69-598f83a144a0)



Example output:
![stitched1e](https://github.com/devinbresser/ece766-panorama-app/assets/66394890/942302b9-e83e-4b94-8e67-8d1d99abd8cb)
