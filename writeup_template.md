# **Advanced Lane Finding using OPENCV**

The goals / steps of this project are the following:

1. [x] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. [x] Apply a distortion correction to raw images.
3. [x] Use color transforms, gradients, etc., to create a thresholded binary image.
4. [x] Apply a perspective transform to rectify binary image ("birds-eye view").
5. [ ] Detect lane pixels and fit to find the lane boundary.
6. [ ] Determine the curvature of the lane and vehicle position with respect to center.
7. [ ] Warp the detected lane boundaries back onto the original image.
8. [ ] Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/binary_combined_thresholds.png "Binary image combined thresholds"
[image3]: ./output_images/perspective_transform.png "Perspective transform in straight lanes road"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## Steps description

### Here I will consider the steps individually and describe how I addressed each step in my implementation.  

---
### 1. Camera Calibration

The implementation of the camera calibration is shown in the second block of code of the jupyter notebook 'Advanced Lane Finding.ipynb'. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection after using `cv2.findChessboardCorners()`. 

The lists of points `objpoints` and `imgpoints`  are used to compute the camera calibration matrix and distortion coefficients, which are the outputs of the `cv2.calibrateCamera()` function.  

### 2. Distortion correction of raw images
As an example of the camera calibration step (see the respective section in the jupyter notebook), I applied this distortion correction to the `calibration5.jpg` image in the folder `camera_cal`. After applying the function `cv2.undistort()` I obtained the following result: 

![alt text][image1]


### 3. Color and gradient thresholds

This step is implemented in the jupyter notebook. The method considers thresholds over the gradient in the x direction of the lightness channel, which are specially useful to identify vertical or near vertical transitions in color a color channel (in this case the L channel of the HLS -hue, lightness, saturation- color space) and thresholds over the saturation channel of the HLS color space. The gradients in the x direction are achieved using `cv2.Sobel()` function over the (1,0) axis, which is the x axis in this case. After conditioning the values (taking the absolute value and scaling) a binary result is defined for those values that are within the specified gradient thresholds. A similar approach is carried out with the saturation channel, where the colors thresholds are used to define a binary image with values inside the minimum and maximum threshold values. Finally, both binary outputs are combined. Here is the results after applying and combining the color and thresholds the image `test5.jpg` from the folder `test_images`:

![alt text][image2]

### 4. Perspective transform

The perspective transform step (implemented in the jupyter notebook) uses the function `cv2.getPerspectiveTransform()` to obtain a transform matrix (M) to map source points to destination points. We can also obtain an inverse transform matrix (Minv) to map destination points to source points changing the source to destination and destination to source in the perspective transform function. The election of source and destination points are key in this step. The defined values are the following:

```python
    offset_left = 245 # offset in x direction at the left side of the warped image
    offset_right = 285 # offset in the x directions at the right side of the warped image

    #b. Define four source points src=np.float32([[,],[,],[,],[,]])
    src = np.float32([[232,img_size[1]-20], 
                     [582,460], 
                     [701,460], 
                     [1080,img_size[1]-20]])

    #c. Define four destination points dst=np.float32([[,],[,],[,],[,]])
    dst = np.float32([[offset_left,img_size[1]],
                     [offset_left,0],
                     [img_size[0]-offset_right,0],
                     [img_size[0]-offset_right,img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 232, 700      | 245, 720        | 
| 582, 460      | 245, 0      |
| 701, 460      | 995, 0      |
| 1075, 700     | 995, 720        |

Once a correct perspective transform matrix is obtained, we expect an image transformation where the lane lines appear parallel and well defined. The following image shows the perspective transform working as expected:

![alt text][image3]

### 5. Detect lane pixels and find the lane lines


## **Pipeline (single images)**

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
