from PIL import Image
import numpy as np
from typing import Union, Tuple, List


def computeHomography(src_pts_nx2: np.ndarray, dest_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Compute the homography matrix.
    Arguments:
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    Returns:
        H_3x3: the homography matrix (3x3 numpy array).
    '''

    # Each equation pair is of the form:
    # [ xs(i) ys(i) 1 0 0 0 -xd(i)xs(i) -xd(i)ys(i) -xd(i) ]  * [h11 h12 h13 h21 h22 h23 h31 h32 h33]^T = [0 0]
    # [ 0 0 0 xs(i) ys(i) 1 -yd(i)xs(i) -yd(i)ys(i) -yd(i) ]

    n = src_pts_nx2.shape[0]
    A = np.zeros((2*n, 9))
                  
    for i in range(n):
        
        # Create the two equations for each point
        xs,ys = src_pts_nx2[i, 0], src_pts_nx2[i, 1]
        xd,yd = dest_pts_nx2[i, 0], dest_pts_nx2[i, 1]
        eqn1 = np.array([xs, ys, 1, 0, 0, 0, -xd*xs, -xd*ys, -xd])
        eqn2 = np.array([0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys, -yd])

        # Fill the two equations into A
        A[2*i] = eqn1
        A[2*i + 1] = eqn2

    # The eigenvector h associated with lmda_min of A^T A minimizes L(h)
    ev, evec = np.linalg.eig(np.dot(A.T, A))
    h = evec[:, np.argmin(ev)]
    H = h.reshape(3,3)
   
    return H
    
    

def applyHomography(H_3x3: np.ndarray, src_pts_nx2: np.ndarray) ->  np.ndarray:
    '''
    Apply the homography matrix to the source points.
    Arguments:
        H_3x3: the homography matrix (3x3 numpy array).
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
    Returns:
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    '''

    # Applying the homography is fairly straightforward:
    # Convert each point into a homogenous coordinate, apply homography, then dehomogenize the resulting points
    n = src_pts_nx2.shape[0]

    # Convert to homogenous coordinates by adding a 1 to every point
    src_pts_homog = np.hstack((src_pts_nx2, np.ones((n,1))))

    # Compute q = H*p
    dest_pts_homog = np.dot(H_3x3, src_pts_homog.T) # because src_pts_homog is nx3, need 3xn

    # Dehomogenize by dividing by last column and then removing it
    dest_pts_nx2 = dest_pts_homog[:2, :]/dest_pts_homog[2, :]

    return dest_pts_nx2.T



def showCorrespondence(img1: Image.Image, img2: Image.Image, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image.
        img2: the second image.
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        result: image depicting the correspondences.
    '''
    raise NotImplementedError

# function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H, dest_canvas_width_height)

def backwardWarpImg(src_img: Image.Image, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[Image.Image, Image.Image]:
    '''
    Backward warp the source image to the destination canvas based on the
    homography given by destToSrc_H. 
    Arguments:
        src_img: the source image.
        destToSrc_H: the homography that maps points from the destination
            canvas to the source image.
        canvas_shape: shape of the destination canvas (height, width).
    Returns:
        dest_img: the warped source image.
        dest_mask: a mask indicating sourced pixels. pixels within the
            source image are 1, pixels outside are 0.
    '''
    raise NotImplementedError


def blendImagePair(img1: List[Image.Image], mask1: List[Image.Image], img2: Image.Image, mask2: Image.Image, mode: str) -> Image.Image:
    '''
    Blend the warped images based on the masks.
    Arguments:
        img1: list of source images.
        mask1: list of source masks.
        img2: destination image.
        mask2: destination mask.
        mode: either 'overlay' or 'blend'
    Returns:
        out_img: blended image.
    '''
    raise NotImplementedError

def runRANSAC(src_pt: np.ndarray, dest_pt: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Run the RANSAC algorithm to find the inliers between the source and
    destination points.
    Arguments:
        src_pt: the coordinates of the source points (nx2 numpy array).
        dest_pt: the coordinates of the destination points (nx2 numpy array).
        ransac_n: the number of iterations to run RANSAC.
        eps: the threshold for considering a point to be an inlier.
    Returns:
        inliers_id: the indices of the inliers (kx1 numpy array).
        H: the homography matrix (3x3 numpy array).
    '''
    raise NotImplementedError

def stitchImg(*args: Image.Image) -> Image.Image:
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''
    raise NotImplementedError
