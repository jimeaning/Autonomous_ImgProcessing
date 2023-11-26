import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import sys, select ,os, threading
import cv2
import numpy as np
import threading
import datetime

import rospkg

rospack = rospkg.RosPack()
package_path = rospack.get_path('move')



if os.name == 'nt':
    import msvcrt,time
else:
    import tty, termios,time




def getKey():
    
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    return key



        
def moveThread():
        global start_flag, current_speed,twist,pub
        speed = 0
        while True:
                key = getKey()
                
                if key == 'q':
                        speed = -0.11
                        print("시작")
                elif key == 'w':
                        speed = 0
                        print("멈춤")
                elif key == 'e':
                        print("끝")
                        break
                
                if current_speed != speed:
                        print("속도" , speed)
                        current_speed = speed
                        
                twist.linear.x = speed
                pub.publish(twist)
                
        

def image_callback(ros_image_compressed):
        try:
                global tracker
                np_arr = np.frombuffer(ros_image_compressed.data, np.uint8)
                video = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                video = cv2.resize(video,(640,480))

                if tracker is None:
                        tracker = cv2.TrackerMIL_create()
                        x = 260
                        y = 230
                        w = 70
                        h = 50
                        
                        TrafficRoi = (x,y,w,h)
                        
                        tracker.init(video, TrafficRoi)

                # 이미지 테스트, 분류 

                
                # 트랙커 설정
                
                ret, TrafficROI = tracker.update(video)
                
                cv2.rectangle(video, TrafficROI[:4], (0,0,255), 2)
                cv2.imshow('frame', video)

                ROI_x = TrafficROI[0]
                ROI_y = TrafficROI[1]
                ROI_w = TrafficROI[2]
                ROI_h = TrafficROI[3]


                TrafficImage = video[ROI_y:ROI_y + ROI_h, 
                                        ROI_x:ROI_x + ROI_w]
                
                
                TrafficHSV = cv2.cvtColor(TrafficImage,cv2.COLOR_BGR2HSV)
                #TrafficHSV = cv2.cvtColor(video,cv2.COLOR_RGB2HSV)
                #TrafficGray = cv2.cvtColor(TrafficHSV, cv2.COLOR_BGR2GRAY)
                #TrafficBin = cv2.cvtColor(TrafficGray, cv2.THRESH_BINARY)
                
                #cv2.imshow('bin', TrafficBin)
                
                # 색 검출 과정 
                # 초록색
                lower_green = np.array([36,30,30])
                higher_green = np.array([83,255,255])
                # 빨간색
                lower_red = np.array([15,100,0])
                higher_red = np.array([173,255,255])
                # # 노란색
                # lower_yellow = np.array([8,100,80])
                # higher_yellow = np.array([20,255,255])

                mask_green = cv2.inRange(TrafficHSV,lower_green, higher_green)
                mask_red = cv2.inRange(TrafficHSV, lower_red, higher_red)
                # mask_yellow = cv2.inRange(TrafficHSV, lower_yellow, higher_yellow)

                res_red = cv2.bitwise_and(TrafficHSV, TrafficHSV, mask=mask_red)
                res_green = cv2.bitwise_and(TrafficHSV, TrafficHSV, mask=mask_green)
                # res_yellow = cv2.bitwise_and(TrafficHSV, TrafficHSV, mask=mask_yellow)
                
                # 색오픈 확장
                kernelSz = 3
                shape = cv2.MORPH_RECT
                sz = (2 * kernelSz + 1, 2 * kernelSz + 1)
                SE = cv2.getStructuringElement(shape, sz)

                src_Red_1st_open = cv2.morphologyEx(res_red, cv2.MORPH_OPEN, SE)
                src_Red_2nd_dilate = cv2.morphologyEx(src_Red_1st_open, cv2.MORPH_DILATE, SE)

                #src_Yellow_1st_open = cv2.morphologyEx(res_yellow, cv2.MORPH_OPEN, SE)
                #src_Yellow_2nd_dilate = cv2.morphologyEx(src_Yellow_1st_open, cv2.MORPH_DILATE, SE)
                
                src_Green_1st_open = cv2.morphologyEx(res_green, cv2.MORPH_OPEN, SE)
                src_Green_2nd_dilate = cv2.morphologyEx(src_Green_1st_open, cv2.MORPH_DILATE, SE)
                                
                
                #cv2.imshow('Yellow', src_Yellow_2nd_dilate)
                
                # 면적
                src_Red_2nd_dilate_gray = cv2.cvtColor(src_Red_2nd_dilate, cv2.COLOR_BGR2GRAY)
                src_Green_2nd_dilate_gray = cv2.cvtColor(src_Green_2nd_dilate, cv2.COLOR_BGR2GRAY)
                
                _, src_Red_2nd_dilate_binary = cv2.threshold(src_Red_2nd_dilate_gray, 1, 255, cv2.THRESH_BINARY)
                _, src_Green_2nd_dilate_binary = cv2.threshold(src_Green_2nd_dilate_gray, 1, 255, cv2.THRESH_BINARY)

                contours_red, _ = cv2.findContours(src_Red_2nd_dilate_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_green, _ = cv2.findContours(src_Green_2nd_dilate_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                area_r = 0
                area_g = 0

                for i, contour in enumerate(contours_red):
                        area_r = cv2.contourArea(contour)
                        area_R = f"Red area = {area_r:.1f}"
                        print(area_R)
                
                for i, contour in enumerate(contours_green):
                        area_g = cv2.contourArea(contour)
                        area_G = f"green area = {area_g:.1f}"
                        print(area_G)
                # if int(area_r) > 1300:
                #     print('stop')
                
                # if int(area_g) > 1000:
                #     print('go')
                

                # ESC를 누르면 종료
                key = cv2.waitKey(1) & 0xFF
                if (key == 27): 
                        exit()
                
        except CvBridgeError as e:
                print("Error")




if __name__ == '__main__':
        
        tracker = None
        
        # 로봇 제어 전역변수 
        start_flag = 1
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
