import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import sys, select ,os
import threading
if os.name == 'nt':
    import msvcrt,time
else:
    import tty, termios,time

import cv2
import numpy as np

def getKey():
    
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
                key = sys.stdin.read(1)
        else:
                key = ''
        return key
        
def moveThread():
        global  twist,pub,StopLineFlag, GreenLightFlag
        
        speed = 0
        current_speed =0 
        Start = 0
        
        while True:
                key = getKey()
                
                if key == 'q':
                        speed = -0.11
                        Start = 1
                        print("시작")
                elif key == 'w':
                        speed = 0
                        Start = 0
                        print("강제멈춤")
                elif key == 'e':
                        print("끝")
                        break
                
                
                if Start == 1:        
                        if GreenLightFlag == 1:
                                print("초록색")
                                speed = -0.11
                        elif GreenLightFlag == 0:
                                if StopLineFlag == 0:
                                        print("빨간색 그리고 정지선")
                                        time.sleep(1)
                                        speed = 0
                                else:
                                        print("빨간색")
                                        speed = -0.11
                                        
                if current_speed != speed:
                        print("속도" , speed)
                        current_speed = speed
                        
                twist.linear.x = speed        
                pub.publish(twist)
                        
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

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	lines_p = cv2.HoughLinesP(img, rho,theta, threshold,np.array([]),
		minLineLength=min_line_len,
		maxLineGap=max_line_gap)
 
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

def image_callback(ros_image_compressed):
        try:
                global tracker,StopLineFlag,GreenLightFlag
                global box_x, box_y, box_w, box_h
                np_arr = np.frombuffer(ros_image_compressed.data, np.uint8)
                video = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                video = cv2.resize(video,(640,480))
                video_h,video_w,video_c = video.shape 

                #video = cv2.rotate(video, cv2.ROTATE_90_COUNTERCLOCKWISE)
                StopVideo = video.copy()

                if tracker is None:
                        tracker = cv2.TrackerMIL_create()
                        x = 130
                        y = 130
                        w = 270
                        h = 50
                        
                        TrafficRoi = (x,y,w,h)
                        
                        tracker.init(video, TrafficRoi)                       

                # 이미지 테스트, 분류 
                
                # 트랙커 설정
                ret, TrafficROi = tracker.update(video)
                cv2.rectangle(video, TrafficROi[:4], (0,0,255), 2)
                

                
                ROI_x = TrafficROi[0]
                ROI_y = TrafficROi[1]
                ROI_w = TrafficROi[2]
                ROI_h = TrafficROi[3]

                TrafficImage = video[ROI_y:ROI_y + ROI_h, 
                                        ROI_x:ROI_x + ROI_w]
                                        
                TrafficHSV = cv2.cvtColor(TrafficImage,cv2.COLOR_BGR2HSV)

                # 정지선 처리 
                
                height,width=StopVideo.shape[0:2]
                x=width//2
                y=(height*3)//6

                # 새로운 이미지를 생성하고 초기화합니다.
                drawing = StopVideo
                
                # 2. gradient combine 
                # Find lane lines with gradient information of Red channel
                temp = StopVideo[y:height+1, 0:width+1] 

                 # White Scale 영역만 나오게
                color_filter_roi = color_filter(temp)
                cv2.imshow('color_filter_roi', color_filter_roi)
                
                gray_roi = cv2.cvtColor(color_filter_roi, cv2.COLOR_BGR2GRAY)
                gray_img=cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

                # Gaussian 필터 적용
                kernel_size = 5
                blur_gray = cv2.GaussianBlur(gray_img, (kernel_size,kernel_size), 0)
                
                # Canny
                low_threshold = 50
                high_threshold = 255
                edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
                
                cv2.imshow('edges', edges)
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                close_filter = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
                cv2.imshow('close', close_filter)
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
                
                for i, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        # print(area)
                        rect = cv2.minAreaRect(contour)
                        # 중심점 추출
                        center = rect[0]
                        #center[0]..x 305
                        #center[1]..y
                        if area > 1200 and 200<center[0]<400:

                        # 무작위 색상 생성
                                color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                                # 윤곽선을 색으로 채워서 그립니다.
                                box = cv2.boxPoints(rect).astype(np.int0)

                                # 미네어리엑트 좌표를 이용하여 사각형 그리기
                                box_x, box_y, box_w, box_h = cv2.boundingRect(contour)
                                box_y += y
                                hull = cv2.convexHull(contour)+(0,y)
                        
                                # 투명한 이미지 생성
                                transparent_image = np.zeros((drawing.shape[0], drawing.shape[1], 3), dtype=np.uint8)
                                cv2.drawContours(transparent_image, [hull], -1, (0, 255, 0), thickness=cv2.FILLED)
                                drawing = cv2.addWeighted(drawing, 1, transparent_image, 0.4, 0)

                                if box_w * box_h < 2000:
                                        StopLineFlag = 0 
                                        print(box_w * box_h)
                             
                cv2.imshow('stopline', drawing)
                
                # 색 검출 과정 
                # 초록색
                lower_green = np.array([36,30,30])
                higher_green = np.array([83,255,255])
                # 빨간색
                lower_red = np.array([15,100,0])
                higher_red = np.array([173,255,255])
                # 마스크 과정 
                mask_green = cv2.inRange(TrafficHSV,lower_green, higher_green)
                mask_red = cv2.inRange(TrafficHSV, lower_red, higher_red)

                res_red = cv2.bitwise_and(TrafficHSV, TrafficHSV, mask=mask_red)
                res_green = cv2.bitwise_and(TrafficHSV, TrafficHSV, mask=mask_green)
                
                # 색오픈 확장
                kernelSz = 3
                shape = cv2.MORPH_RECT
                sz = (2 * kernelSz + 1, 2 * kernelSz + 1)
                SE = cv2.getStructuringElement(shape, sz)

                src_Red_1st_open = cv2.morphologyEx(res_red, cv2.MORPH_OPEN, SE)
                src_Red_2nd_dilate = cv2.morphologyEx(src_Red_1st_open, cv2.MORPH_DILATE, SE)
                
                src_Green_1st_open = cv2.morphologyEx(res_green, cv2.MORPH_OPEN, SE)
                src_Green_2nd_dilate = cv2.morphologyEx(src_Green_1st_open, cv2.MORPH_DILATE, SE)
                                
                # 면적
                src_Red_2nd_dilate_gray = cv2.cvtColor(src_Red_2nd_dilate, cv2.COLOR_BGR2GRAY)
                src_Green_2nd_dilate_gray = cv2.cvtColor(src_Green_2nd_dilate, cv2.COLOR_BGR2GRAY)
                
                _, src_Red_2nd_dilate_binary = cv2.threshold(src_Red_2nd_dilate_gray, 1, 255, cv2.THRESH_BINARY)
                _, src_Green_2nd_dilate_binary = cv2.threshold(src_Green_2nd_dilate_gray, 1, 255, cv2.THRESH_BINARY)

                contours_red, _ = cv2.findContours(src_Red_2nd_dilate_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_green, _ = cv2.findContours(src_Green_2nd_dilate_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                area_r = 0
                area_g = 0

                for i, contour in enumerate(contours_red):
                        area_r = cv2.contourArea(contour)
                        GreenLightFlag = 0
                        
                
                for i, contour in enumerate(contours_green):
                        area_g = cv2.contourArea(contour)
                        GreenLightFlag = 1
                        
                cv2.imshow('traffic_light', video)
                         
                # ESC를 누르면 종료
                key = cv2.waitKey(1) & 0xFF
                if (key == 27): 
                        exit()
                
        except CvBridgeError as e:
                print("Error")

## MAIN
if __name__ == '__main__':
        
        tracker = None
        StopLineFlag = 1        # 정지선 포착
        box_x= box_y= box_w= box_h = 0
        # 로봇 제어 전역변수 
        GreenLightFlag = 1
        current_speed =0
 
        # 카메라 처리 
        rospy.init_node('autonomous_move')
        image_topic = "/raspicam_node/image/compressed"
        rospy.Subscriber(image_topic, CompressedImage, image_callback)
        print("Subscribe start")

        # 로봇 움직임 처리 
        pub = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
        
        twist = Twist()
        twist.linear.x = twist.linear.y = twist.linear.z = 0
        twist.angular.x = twist.angular.y = twist.angular.z = 0
        
        pub.publish(twist)
        
         # 로봇 제어 쓰레드 실행
        moveth = threading.Thread(target=moveThread)
        moveth.start()
        
        # 카메라 계속 찍게 하는거 라고 생각하면됨
        rospy.spin()
