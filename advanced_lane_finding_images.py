import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

#CALIBRATION PROCESS
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
    x_size = image_g.shape[1]
    y_size = image_g.shape[0]
    ret, corners = cv2.findChessboardCorners(image_g,(nx,ny),None)
    
    if ret == True:
        i_array.append(corners)
        o_array.append(obj_points)

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(o_array,i_array,(y_size,x_size),None,None)

##DEFINING ALL THE HELPER FUNCTIONS

#image undistortion
def undstrtIm(img,cameraMatrix,distCoeffs):
    return cv2.undistort(img,cameraMatrix,distCoeffs,None,cameraMatrix)

#thresholding function
def threshMasking(img):
    min_g = 20  #minimum gradient threshold
    max_g = 110 #maximum gradient threshold
    min_s = 155  #minimum saturation threshold
    max_s = 255 #maximum saturation threshold
    min_c = 200  #minimum saturation threshold
    max_c = 255 #maximum saturation threshold
    
    #thresholding based on gradient
    img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(img_g,cv2.CV_64F,1,0)    #gradient computation
    abs_gradx = np.abs(grad_x)  #absolute gradient values
    scaled_absx = np.uint8(255*(abs_gradx/np.max(abs_gradx)))   #scaled gradients that lie between 0 & 255
    bin_g = np.zeros((y_size,x_size),dtype='uint8') #empty array to store values for 'activated' pixels
    bin_g[(scaled_absx >= min_g)&(scaled_absx <= max_g)] = 1
    
    #thresholding based on saturation
    img_s = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    S = img_s[:,:,2]   #extracting the saturation channel    
    bin_s = np.zeros((y_size,x_size),dtype='uint8')
    bin_s[(S > min_s) & (S <= max_s)] = 1
    
    #thresholding based on color
    bin_c = np.zeros((y_size,x_size),dtype='uint8')
    bin_c[(img_g > min_c) & (img_g <= max_c)] = 1
    
    #combining all the thresholds
    bin_comb = np.zeros((y_size,x_size),dtype='uint8')  #initialising the combined array with zeros
    bin_comb[(bin_g == 1)|(bin_s == 1)|(bin_c==1)] = 1

    #masking this image
    vertices = np.array([[(150,720),(590,450),(750,450),(1250,720)]])
    mask = np.zeros((y_size,x_size),dtype='float32')
    mask = cv2.fillPoly(mask,vertices,255)
    mask = np.uint8(mask)
    
    return cv2.bitwise_and(bin_comb,mask)

#perspective transformation function
def imageTransform(img):
    warped = np.array([[[200,720]],[[200,0]],[[1000,0]],[[800,720]]],np.float32)
    vertices = np.reshape(np.array([[(150,720),(590,450),(750,450),(1250,720)]],dtype='float32'),(4,1,2))
    M_persp = cv2.getPerspectiveTransform(vertices,warped)  #transform matrix to carry out the perspective transformation
    M_persp_inv = np.linalg.inv(M_persp)                    #inverse transform matrix to be used later
    return M_persp_inv, cv2.warpPerspective(img,M_persp,(x_size,y_size),flags=cv2.INTER_LINEAR)

#finding the lane lines
def laneLines(img1,M_persp_inv,img2):
    window_margin = 100
    window_no = 10  
    window_height = np.int(y_size/window_no)
    minpix = 50
    #determining the start of the search based on the previous frame
    sum = np.sum(img1[img1.shape[0]//2:,:],axis=0)   #finding peaks for the sliding window search to begin
    mid = np.int(x_size/2)
    left_start = np.argmax(sum[:mid])       #starting point - left lane
    right_start = np.argmax(sum[mid:])+mid  #starting point - right lane
    left_current = left_start
    right_current = right_start
    
    nonzeroxy = img1.nonzero()
    nonzerox = np.array(nonzeroxy[1])
    nonzeroy = np.array(nonzeroxy[0])
    co_l_indices = []
    co_r_indices = []
    
    upper_bound = y_size
    for i in range(window_no):
        xlower_l  = left_current - window_margin
        xlower_r  = right_current - window_margin
        xupper_l  = left_current + window_margin 
        xupper_r   = right_current + window_margin
        lower_bound = upper_bound - window_height
        indices_l = ((nonzerox >= xlower_l)&(nonzerox < xupper_l)&(nonzeroy >= lower_bound)&(nonzeroy < upper_bound)).nonzero()[0]
        indices_r = ((nonzerox >= xlower_r)&(nonzerox < xupper_r)&(nonzeroy >= lower_bound)&(nonzeroy < upper_bound)).nonzero()[0]
        if len(indices_l) >= minpix:
            left_current = np.int(np.mean(nonzerox[indices_l]))
        if len(indices_r) >= minpix:
            right_current = np.int(np.mean(nonzerox[indices_r]))
        co_l_indices.append(indices_l)
        co_r_indices.append(indices_r)
        upper_bound = lower_bound
    
    co_l_indices = np.concatenate(co_l_indices)
    co_r_indices = np.concatenate(co_r_indices)
    
    leftx = nonzerox[co_l_indices]
    lefty = nonzeroy[co_l_indices]
    rightx = nonzerox[co_r_indices]
    righty = nonzeroy[co_r_indices]
    
    poly_l = np.polyfit(lefty,leftx,2)
    poly_r = np.polyfit(righty,rightx,2)
    y_new = np.linspace(0,y_size-1,y_size)
    x_newl= np.polyval(poly_l,y_new)
    x_newr = np.polyval(poly_r,y_new)

    left_lane = np.transpose(np.array((x_newl,y_new)))
    right_lane = np.transpose(np.array((x_newr,y_new)))[::-1]
    lanes_both = np.vstack((left_lane,right_lane))
    final_im = np.zeros((y_size,x_size,3),dtype='uint8')
    final_im = cv2.fillPoly(final_im,np.int32([lanes_both]),(0,255,0))
    final_im_tr = cv2.warpPerspective(final_im,M_persp_inv,(x_size,y_size),flags=cv2.INTER_LINEAR)
    return poly_l, poly_r, cv2.addWeighted(final_im_tr,0.3,img2,1,0), sum


#curvature function
def curvatureRadius(poly,mx,my):
    return np.round(((np.power(1+np.power(((2*poly[0]*mx/(my*my))*(719*my))+(poly[1]*mx/my),2),1.5))/np.absolute(2*poly[0]*mx/(my*my))),2)

#offset function
def offset(poly_r,poly_l,x_size,mx):
    return (np.mean((np.polyval(poly_r,719),np.polyval(poly_l,719)))-x_size/2)*mx/2

#final output function
def finalOutput(R_l,R_r,offset_1,img):
    left_curvature = 'Left Lane Curvature: {}m'.format(R_l)
    right_curvature = 'Right Lane Curvature: {}m'.format(R_r)
    offset_string= 'Offset: {}m'.format(np.round(offset_1,2))
    cv2.putText(img,left_curvature,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,0),3)
    cv2.putText(img,right_curvature,(100,150),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,0),3)
    cv2.putText(img,offset_string,(100,200),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,0),3)
    return np.uint8(img)

##main code

test = plt.imread('test_images/straight_lines2.jpg')
filt_masked = threshMasking(test)
M_persp_inv, trans_im = imageTransform(filt_masked)
poly_l, poly_r, lanes, sum = laneLines(trans_im,M_persp_inv,test)
R_l = curvatureRadius(poly_l,mx=3.7/700,my=30/720)
R_r = curvatureRadius(poly_r,mx=3.7/700,my=30/720)
offset1 = offset(poly_l,poly_r,x_size,mx=3.7/700)
final_im = finalOutput(R_l,R_r,offset1,lanes)
plt.imsave('output_images/straight_lines2.jpg',final_im)
plt.imshow(final_im)

