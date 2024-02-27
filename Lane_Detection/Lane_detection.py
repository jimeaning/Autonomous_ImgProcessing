import cv2
import numpy as np

video = cv2.VideoCapture('./video/tracking_sample.mp4')

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

## 흰색선 검출
def color_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0,0,150])
    upper = np.array([255,255,255])

    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    
    return masked

# 2번째 ROI 영역
def region_of_interest(img,vertices):
	mask = np.zeros_like(img)

	if len(img.shape)>2 : 
		channel_count =img.shape[2]
		ignore_mask_color = (255,)*channel_count
	else :
		ignore_mask_color= 255

	cv2.fillPoly(mask, vertices, ignore_mask_color)

	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

# def draw_lines(img,lines, color = [255,0,0], thickness=5):
#     print(lines)
	# for line in lines:
	# 	for x1,y1,x2,y2 in line:
	# 		cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	lines_p = cv2.HoughLinesP(img, rho,theta, threshold,np.array([]),
		minLineLength=min_line_len,
		maxLineGap=max_line_gap)

	# line_img=np.zeros((img.shape[0], img.shape[1],3),dtype=np.uint8)
	# draw_lines(line_img, lines_p)
	# return line_img

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

while(video.isOpened()):
    ret, frame = video.read()

    if not ret: 
        print('에러: 카메라')
        break
    
    height,width=frame.shape[0:2]
    x=width//2
    y=(height*3)//4
    ##################################
    # 새로운 이미지를 생성하고 초기화합니다.
    drawing = frame
    
    # 2. gradient combine 
    # Find lane lines with gradient information of Red channel
    temp = frame[y:height+1, 0:width+1] 
    #temp=cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    
    '''
        # setting thresholds (hls, sobel)
        th_h, th_l, th_s = (120, 255), (50, 160), (0, 255)
        th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
        

        # sobel filter X 스케일러 미적용
        th_sobelx = (35, 100)
        sobel_x = sobel_xy(temp, 'x', th_sobelx)
        cv2.imshow('debugging_window', th_sobelx)
        sobel_x=cv2.cvtColor(sobel_x, cv2.COLOR_BGR2RGB)
        cv2.imshow('debugging_window', th_sobelx)
        
        
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

        H = hls[y:height+1, 0:width+1, 0] # get Hue channel (색상)
        L = hls[y:height+1, 0:width+1, 1] # get Light channel (밝기)
        S = hls[y:height+1, 0:width+1, 2] # get Saturation channel (채도)

        l_img = np.zeros_like(L)
        l_img[(L > th_l[0]) & (L <= th_l[1])] = 255
            
        #l_img_resize = l_img[y:height+1, 0:width+1]
        
        ##################################
        # ROI 영역
        cv2.imshow('l_img', l_img)

    '''
    
    # White Scale 영역만 나오게
    color_filter_roi = color_filter(temp)
    cv2.imshow('color_filter_roi', color_filter_roi)
    
    gray_roi = cv2.cvtColor(color_filter_roi, cv2.COLOR_BGR2GRAY)
    
    gray_img=cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    # Gaussian 필터 적용
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray_img, (kernel_size,kernel_size), 0)
    
    # Canny
    low_threshold = 200
    high_threshold = 255
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    cv2.imshow('edges', edges)

    # ROI .2
    imshape = edges.shape
    #print(imshape)
    vertices = np.array([[(30, imshape[0]),
		(450, 320),
		(550, 320),
		(imshape[1]-20, imshape[0])]], dtype= np.int32)

    mask = region_of_interest(edges, vertices)
    #cv2.imshow('mask', mask)

    # 선 그리기
    rho = 2
    theta = np.pi/180
    threshold = 90
    min_line_len = 120
    max_line_gap = 150

    lines = hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)
    
    # Contours 찾기
    # contours, hierarchy를 찾아냅니다.
    contours, hierarchy = cv2.findContours(gray_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # contours의 각 윤곽선에 대해 작업을 수행합니다.
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        # 중심점 추출
        center = rect[0]
        #center[0]..x 305
        #center[1]..y
        if area > 1500 and 200<center[0]<400:

            # 무작위 색상 생성
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

            # 윤곽선을 색으로 채워서 그립니다.
            box = cv2.boxPoints(rect).astype(np.int0)

            # 미네어리엑트 좌표를 이용하여 사각형 그리기
            #cv2.rectangle(drawing, tuple(box[1]), tuple(box[3]), (0, 255, 0), 2)
            #cv2.rectangle(drawing, (box[0]), (110, 110), (255, 255, 255), -1)
            box_x, box_y, box_w, box_h = cv2.boundingRect(contour)
            box_y += y
            #cv2.rectangle(drawing, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 0, 255), 2)
            #cv2.fillPoly(drawing, contour, (255, 255, 255))
            hull = cv2.convexHull(contour)+(0,y)
            #cv2.drawContours(drawing, [hull], -1, (0, 255, 0), 2)
            
            # 투명한 이미지 생성
            transparent_image = np.zeros((drawing.shape[0], drawing.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(transparent_image, [hull], -1, (0, 255, 0), thickness=cv2.FILLED)
            drawing = cv2.addWeighted(drawing, 1, transparent_image, 0.4, 0)

        # else

    # 결과 출력
    cv2.imshow('drawing', drawing)

    # ESC를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == 27: 
        break

video.release()
cv2.destroyAllWindows()