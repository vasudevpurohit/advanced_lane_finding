import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

#calculating the distortion coeffecients and the camera matrix
img = glob.glob('camera_cal/calibration*')
nx = 9 #no. of corners along x
ny = 6 #no. of corners along y
obj_points = np.zeros((ny,nx,3),dtype='float32')                                
for i in range(ny):
    for j in range(nx):
        obj_points[i][j] = [j,i,0] #defining the object point array
obj_points = np.reshape(obj_points,(nx*ny,1,3)) #matching its shape to the i_array
o_array = [] #array to store all the object 3D points
i_array = [] #array to store all the image 2D points

for fname in img:                                                               
    image = plt.imread(fname)
    image_g = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    x_size = image_g.shape[0]
    y_size = image_g.shape[1]
    ret, corners = cv2.findChessboardCorners(image_g,(nx,ny),None)
    
    if ret == True:
        i_array.append(corners)
        o_array.append(obj_points)

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(o_array,i_array,(x_size,y_size),None,None)

#undistorting a real world image
test = plt.imread('test_images/test1.jpg')
test_udst = cv2.undistort(test,cameraMatrix,distCoeffs,None,cameraMatrix) #undistorting the pulled test image

#thresholding 'test' image based on color and gradient
test_g = cv2.cvtColor(test_udst,cv2.COLOR_BGR2GRAY)  #grayscale test image
k_size = 5  #kernel size for the Sobel operator

grad_x = cv2.Sobel(test_g,cv2.CV_64F,1,0,k_size)    #gradient computation
abs_gradx = np.abs(grad_x)  #absolute gradient values
scaled_absx = np.uint8(255*(abs_gradx/np.max(abs_gradx)))   #scaled gradients that lie between 0 & 255

min_g = 30  #minimum gradient threshold
max_g = 100 #maximum gradient threshold

bin_g = np.zeros((x_size,y_size),dtype='uint8') #empty array to store values for 'activated' pixels
bin_g[(scaled_absx >= min_g)&(scaled_absx <= max_g)] = 1

#thresholding the image based on color
test_c = cv2.cvtColor(test_udst,cv2.COLOR_BGR2HLS)
S = test_c[:,:,2]   #extracting the saturation channel

min_s = 170  #minimum saturation threshold
max_s = 255 #maximum saturation threshold
bin_s = np.zeros((x_size,y_size),dtype='uint8')
bin_s[(S > min_s) & (S <= max_s)] = 1

bin_comb = np.zeros((x_size,y_size),dtype='uint8')  #initialising the combined array with zeros
bin_comb[(bin_g == 1)|(bin_s == 1)] = 1

#perspective transformation
vertices = np.array([[(200,670),(1100,670),(750,450),(550,450)]])
mask = np.zeros((x_size,y_size),dtype='float32')
mask = cv2.fillPoly(mask,vertices,255)
mask = np.uint8(mask)
final_im = cv2.bitwise_and(bin_comb,mask)

m = 100 #margin
warped = np.array([[[80,700]],[[1200,700]],[[1200,50]],[[80,50]]],np.float32)
vertices = np.reshape(vertices,(4,1,2))
vertices = np.float32(vertices)
M_persp = cv2.getPerspectiveTransform(vertices,warped)  #transform matrix to carry out the perspective transformation
trans_im = np.zeros((x_size,y_size),dtype='uint8')      #initialising an empty array to store the transformed img
trans_im = cv2.warpPerspective(final_im,M_persp,(y_size,x_size),flags=cv2.INTER_LINEAR)

plt.imshow(trans_im,cmap='gray')

#finding polynomials for lane lines

