from PIL import Image, ImageDraw
import numpy as np
from typing import Union, Tuple, List
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt



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

    src_pts_nx2 = np.asarray(src_pts_nx2)
    dest_pts_nx2 = np.asarray(dest_pts_nx2)

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
        A[2*i+1] = eqn2

    # The eigenvector h associated with lmda_min of A^T*A minimizes L(h)
    ev, evec = np.linalg.eig(np.dot(A.T, A))
    h = evec[:, np.argmin(ev)]
    H = h.reshape(3,3)
   
    # Visualization for testing purposes
    # pil_img1 = Image.fromarray(img1)
    # pil_img2 = Image.fromarray(img2)
    # draw1 = ImageDraw.Draw(pil_img1)
    # draw2 = ImageDraw.Draw(pil_img2)

    # for (x, y) in src_pts_nx2:
    #     draw1.ellipse((x-5, y-5, x+5, y+5), fill=(255,0,0), outline=(255,0,0), width=2)

    # for (x, y) in dest_pts_nx2:
    #     draw2.ellipse((x-5, y-5, x+5, y+5), fill=(0,255,0), outline=(0,255,0), width=2)

    # Display both images
    # pil_img1.show()
    # pil_img2.show()


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



def showCorrespondence(img1: np.ndarray, img2: np.ndarray, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image (array).
        img2: the second image (array).
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        result: image depicting the correspondences.
    '''
    #print(f'Original img1 shape: {img1.shape}') 
    #print(f'Original img2 shape: {img2.shape}') 
    
    # Handling the case where one image is RGBA and one is just RGB
    if img1.shape[2] != img2.shape[2]:
        if img1.shape[2] == 4:  # img1 is RGBA, convert to RGB
            img1 = img1[:, :, :3]
        elif img2.shape[2] == 4:  # img2 is RGBA, convert to RGB
            img2 = img2[:, :, :3]

    # Concatenate images side by side
    combined = np.concatenate((img1, img2), axis=1)

    # Now the points for the second image need to be shifted horizontally by the width of the first image
    pts2_shift = np.copy(pts2_nx2)
    pts2_shift[:,0] += img1.shape[1]

    # Draw all of the lines and points on the image using PIL
    combined_img = Image.fromarray(combined)
    draw = ImageDraw.Draw(combined_img)

    for (x1, y1), (x2, y2) in zip(pts1_nx2, pts2_shift):
        draw.line((x1, y1, x2, y2), fill=(255,0,0), width=2)

    for x, y in pts1_nx2:
        draw.ellipse((x-4, y-4, x+4, y+4), fill=(0,0,255), outline=(0,0,255))

    for x, y in pts2_shift:
        draw.ellipse((x-4, y-4, x+4, y+4), fill=(0,0,255), outline=(0,0,255))

    return combined_img


# function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H, dest_canvas_width_height)

def backwardWarpImg(src_img: np.ndarray, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[Image.Image, Image.Image]:
    '''
    Backward warp the source image to the destination canvas based on the
    homography given by destToSrc_H. 
    Arguments:
        src_img: the source image.
        destToSrc_H: the homography that maps points from the destination
            canvas to the source image.
        canvas_shape: shape of the destination canvas (height, width).
    Returns:
        dest_img_array: the warped source image (array format).
        dest_mask: a mask indicating sourced pixels. pixels within the
            source image are 1, pixels outside are 0.
    '''
    # For each pixel on the billboard, we need to determine what part of the portrait should be there
    # We will accomplish this by applying the inverse Homography matrix to every point in the portrait
    # This is not the most efficient implementation but finding a "vectorized" method is a bit challenging.

    dest_img_array = np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=float)
    dest_mask = np.zeros((canvas_shape[0], canvas_shape[1]), dtype=bool)

    for x in range(canvas_shape[1]):
        for y in range(canvas_shape[0]):

            x_src, y_src, z_tilde = np.dot(destToSrc_H, [x, y, 1]) # Dot product of homogenized point with inverse Homography matrix
            x_src = int(x_src/z_tilde) # Dehomogenize x and y points (just using int() to round the pixel location)
            y_src = int(y_src/z_tilde)

            # Check if the newly mapped pixel should be within the mask
            if 0 <= x_src < src_img.shape[1] and 0 <= y_src < src_img.shape[0]:
                # If so, add its values (R,G,B) to the destination image and mark the mask with a 1
                dest_img_array[y, x, :] = src_img[y_src, x_src, :] 
                dest_mask[y, x] = 1

    return dest_img_array, dest_mask


def runRANSAC(src_points: np.ndarray, dest_points: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
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

    # Solution method: 
    # For ransac_n iterations:
    #   Take a random sample of 4 (src_pt, dest_pt) pairs
    #   Compute the homography matrix with that set of pairs
    #   Count & store inliers among all of the other points with that set
    # Conclude that the 4-pt sequence with the highest number of inliers is the best mapping.

    H = None
    inliers_id = []

    for _ in range(ransac_n):

        index = np.random.choice(len(src_points), 4, replace=False)
        src_sample = src_points[index]
        dest_sample = dest_points[index]
    
        candidate_H = computeHomography(src_sample, dest_sample) # Compute H using prior method
        transformed_src_points = applyHomography(candidate_H, src_points) # Apply homography using candidate H and source points
        candidate_distances = np.linalg.norm(dest_points - transformed_src_points, axis=1) # Compute the Euclidean distance using this H
        candidate_inliers = np.where(candidate_distances < eps)[0]

        if len(candidate_inliers) > len(inliers_id):
            print(f"Just found a new best H with {len(candidate_inliers)} inliers.")
            inliers_id = candidate_inliers
            H = candidate_H
    
    return inliers_id, H
        



def blendImagePair(img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, mask2: np.ndarray, mode: str) -> Image.Image:
    '''
    Blend the warped images based on the masks.
    Arguments:
        img1: source image (array).
        mask1: source mask (array).
        img2: destination image (array).
        mask2: destination mask (array).
        mode: either 'overlay' or 'blend'
    Returns:
        out_img: blended image.
    '''

    # print("Dimensions of img1:", img1.shape)
    # print("Dimensions of mask1:", mask1.shape)
    # print("Dimensions of img2:", img2.shape)
    # print("Dimensions of mask2:", mask2.shape)

    mask1_binary = (mask1 > 0).astype(int)
    mask2_binary = (mask2 > 0).astype(int)

    result = img1.copy()

    if mode not in ["overlay", "blend"]:
        raise ValueError("Mode must be 'overlay' or 'blend'")
    
    if mode == "overlay":
        # Overlay processing: copy img2 over img1 wherever mask2 = 1
        result[mask2_binary == 1] = img2[mask2_binary == 1]

    if mode == "blend":
        # Blend processing using distance_transform_edt

        dt1 = distance_transform_edt(mask1)
        dt2 = distance_transform_edt(mask2)
        blend_factor = dt2/(dt1 + dt2 + np.finfo(float).eps) # The last term is a very small value to prevent division by 0
        #print(f"blend_factor shape: {blend_factor.shape}")

        # Expand blend_factor from 1D to 3D to match RGB channels of the images
        if len(blend_factor.shape) != 3:
            blend_factor = np.repeat(blend_factor[:, :, np.newaxis], 3, axis=2)

        # Weighted combination of the input images with blend_factor
        result = ((1-blend_factor)*img1 + blend_factor*img2).astype(np.uint8)

    out_img = Image.fromarray(result)
    return out_img


def stitchImg(*imgs):
    """
    Stitches together all of the images onto a common canvas using a combination of the previous methods.
    Input: imgs: a list of images to be stitched. The center image must be first. Images in np array format.
    Output: result: a PIL Image object containing the resulting panorama.

    """

    from helpers import genSIFTMatches

    img_d = imgs[0]
    sources = list(imgs[1:])
    homographies = []
    x_coords = []
    x_translations = []
    y_coords = []
    y_translations = []

    for i, img in enumerate(sources):

        height = img_d.shape[1]
        width = img_d.shape[0]
        
        # Compute H
        xs, xd = genSIFTMatches(img, img_d)
        xs_flip = xs[:,[1,0]]
        xd_flip = xd[:,[1,0]]
        ransac_n, ransac_eps = 1000, 1
        _, H = runRANSAC(xs_flip, xd_flip, ransac_n, ransac_eps)

        
        # Apply the homography to the source image
        height, width = img_d.shape[:2]
        corners = np.array([[0, 0, 1], 
                            [width, 0, 1], 
                            [width, height, 1], 
                            [0, height, 1]]).T
        
    
        warped_corners = np.dot(H, corners)
        warped_corners = warped_corners[:2] / warped_corners[2]
        min_x, min_y = np.min(warped_corners[:2], axis=1)
        x_coords.extend(warped_corners[0])
        y_coords.extend(warped_corners[1])
        
        translation = np.array([[1, 0, -min_x], 
                                [0, 1, -min_y], 
                                [0, 0, 1]])

        x_translations.append(int(min_x))
        y_translations.append(int(min_y))
        
        # Apply the adjusted homography
        H_adj = np.dot(translation, H)
        homographies.append(H_adj)


    # Compute x_range
    furthest_negative_x = np.min(x_coords) if np.any(np.array(x_coords) < 0) else 0
    largest_positive_x = np.max(x_coords) if np.any(np.array(x_coords) > width) else width
    x_range = [furthest_negative_x, largest_positive_x]
    
    furthest_negative_y = np.min(y_coords) if np.any(np.array(y_coords) < 0) else 0
    largest_positive_y = np.max(y_coords) if np.any(np.array(y_coords) > height) else height
    y_range = [furthest_negative_y, largest_positive_y]

    #print(f"x_range: {x_range}, y_range: {y_range}")
    
    canvas_width = int(abs(x_range[1] - x_range[0]))
    canvas_height = int(abs(y_range[1] - y_range[0]))
    #print(f"canvas dimensions: {canvas_height}x{canvas_width} pixels")

    center_canvas = Image.new("RGB", (canvas_width, canvas_height), "black")
    warped_canvases = [Image.new("RGB", (canvas_width, canvas_height), "black") for _ in range(len(sources))]
    center_img_placement_x = int((canvas_width - img_d.shape[1]) // 2)
    center_img_placement_y = int((canvas_height - img_d.shape[0]) // 2)
    
    center_canvas.paste(Image.fromarray(img_d), (center_img_placement_x, center_img_placement_y))
    #main_canvas.show()

    
    # Warp the images onto the canvas
    for i, img in enumerate(sources):
        H = homographies[i]
        ### create temporary canvas to store that image ###
        warped_arr, _ = backwardWarpImg(img, np.linalg.inv(H), (canvas_height, canvas_width))
        warped_img = Image.fromarray(warped_arr.clip(0, 255).astype(np.uint8))

        paste_x = center_img_placement_x + x_translations[i]
        paste_y = center_img_placement_y + y_translations[i]
        
        warped_canvases[i].paste(warped_img, (paste_x, paste_y))
        #warped_canvases[i].show()
    
    def to_mask(array):
        return (array > 0).astype(np.uint8)*255


    warped_arrays = [np.array(img) for img in warped_canvases]
    masks = [to_mask(arr) for arr in warped_arrays]
    
    current_array = np.array(center_canvas)
    current_mask = to_mask(current_array)

        
    # Procedurally blend all of the images
    # 2 images -> 1 operation
    # 3 images -> 2 operations
    # 4 images -> 3 operations etc.

    # Now, blend the first two images, and then any more images into THAT result, if there are any, etc.
    for i in range(len(warped_arrays)):  # Start from the second item in warped_arrays
        next_image = warped_arrays[i]
        next_mask = masks[i]
        
        # Perform the blend
        result = blendImagePair(current_array, current_mask, next_image, next_mask, "blend")
        #intermediate.show()

        current_array = np.array(result)  # Update the current result array for the next blending
        current_mask = to_mask(current_array)  # Update the mask based on the newly blended image

    return result