import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import exposure
import warnings
warnings.filterwarnings('ignore')

video = cv2.VideoCapture('./video/Seoul_Traffic.mp4')

while(video.isOpened()):
    ret, frame = video.read()

    if not ret: 
        print('Error: Camera')
        break

    # 1. resizing 
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    height, width = frame.shape[:2]
    #print(frame.shape[:2])

    # 2. gradient combine 
    # Find lane lines with gradient information of Red channel
    temp = frame[220:height-12, :width, 2]
    temp=cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    cv2.imshow('rgb2bgr ', temp)
    
    # setting thresholds (hls, sobel)
    th_h, th_l, th_s = (120, 255), (50, 160), (0, 255)
    th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
    # 3. Sobel Filter
    def sobel_xy(img, orient='x', thresh=(20, 100)):

        if orient == 'x':
            # dx=1, dy=0이면 x 방향의 편미분
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        
        if orient == 'y':
            # dx=0, dy=1이면 y 방향의 편미분
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

        # Return the result
        return binary_output
        # sobel filter X 스케일러 미적용
    th_sobelx = (35, 100)
    sobel_x = sobel_xy(temp, 'x', th_sobelx)
    sobel_x=cv2.cvtColor(sobel_x, cv2.COLOR_BGR2RGB)
        
        # sobel filter X 스케일러 적용
    img = frame[220:height-12, :width, 2]
    abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sobel_x = np.zeros_like(scaled_sobel)

    th_sobelx = (35, 100)
    sobel_x[(scaled_sobel >= th_sobelx[0]) & (scaled_sobel <= th_sobelx[1])] = 255
    
    #cv2.imshow('sobel_x', sobel_x)
    #     # sobel filter Y 스케일러 미적용
    sobel_y = sobel_xy(th_sobely, 'y', th_sobely)
    sobel_y=cv2.cvtColor(sobel_y, cv2.COLOR_BGR2RGB)
    
    # sobel filter Y 스케일러 적용

    img = frame[220:height-12, :width, 2]
    abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    th_sobely = (30, 255)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sobel_y = np.zeros_like(scaled_sobel)
    sobel_y[(scaled_sobel >= th_sobely[0]) & (scaled_sobel <= th_sobely[1])] = 255

    
    #cv2.imshow('sobel_y', sobel_y)
    
        # gradient magnitude is used to measure how strong the change in image intensity is

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    th_mag = (30, 255)

    gradient_magnitude = np.zeros_like(gradmag)
    gradient_magnitude[(gradmag >= th_mag[0]) & (gradmag <= th_mag[1])] = 255
    #cv2.imshow('gradient_magnitude', gradient_magnitude)
    
        # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=15)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=15)

    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    gradient_direction = np.zeros_like(absgraddir)

    th_dir = (0.7, 1.3)
    gradient_direction[(absgraddir >= th_dir[0]) & (absgraddir <= th_dir[1])] = 255
    gradient_direction = gradient_direction.astype(np.uint8)
    
    # 그래디언트 컴바인
    grad_combine = np.zeros_like(gradient_direction).astype(np.uint8)
    grad_combine[((sobel_x > 1) & (gradient_magnitude > 1) & (gradient_direction > 1)) | ((sobel_x > 1) & (sobel_y > 1))] = 255
    #cv2.imshow('grad_combine', grad_combine)
    
    #HSL
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    height, width = frame.shape[:2]
    _, img = cv2.threshold(frame, 180, 255, cv2.THRESH_BINARY)

    H = hls[220:height - 12, 0:width, 0] # get Hue channel (색상)
    L = hls[220:height - 12, 0:width, 1] # get Light channel (밝기)
    S = hls[220:height - 12, 0:width, 2] # get Saturation channel (채도)

    h_img = np.zeros_like(H)
    h_img[(H > th_h[0]) & (H <= th_h[1])] = 255
    #cv2.imshow('h_img', h_img)
    l_img = np.zeros_like(L)
    l_img[(L > th_l[0]) & (L <= th_l[1])] = 255
    #cv2.imshow('l_img', l_img)
    s_img = np.zeros_like(S)
    s_img[(S > th_s[0]) & (S <= th_s[1])] = 255
    #cv2.imshow('s_img', s_img)
    
    hls_combine = np.zeros_like(s_img).astype(np.uint8)
    hls_combine[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1))] = 255 
    cv2.imshow('hls_combine', hls_combine)
    result = hls_combine

    #Warp image with Perspective Transform
    src = np.float32([[250, 220],
                    [350, 220],
                    [500, 320],
                    [100, 320]
                    ])
    dst = np.float32([[0, 0],
                    [width, 0],
                    [width, height],
                 [0, height]])
    height, width = frame.shape[:2]
    
    s_LTop2, s_RTop2 = [270, 40], [310, 40]
    s_LBot2, s_RBot2 = [100, height], [450, height]

    src = np.float32([s_LBot2, s_RBot2, s_RTop2, s_LTop2]) 
    dst = np.float32([(250, 0), (510, 0), (510, 720), (250, 720)]) 

    # Calculates a perspective transform from four pairs of the corresponding point
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(result, M, (720, 720), flags=cv2.INTER_LINEAR)
    #cv2.imshow('warp_img', warp_img)

    histogram = np.sum(warp_img[int(warp_img.shape[0] / 2):, :], axis=0)
    plt.plot(histogram)
    #plt.show()
    output = np.dstack((warp_img, warp_img, warp_img)) * 255

    cv2.imshow('output', output)



    # ESC를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if (key == 27): 
        break

video.release()
cv2.destroyAllWindows()
