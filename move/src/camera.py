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
                global x, y, classes, layer_names,output_layers,CONF_THR,prev_time,FPS
                global start_flag 
                np_arr = np.frombuffer(ros_image_compressed.data, np.uint8)
                video = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                video = cv2.resize(video,(640,480))
                
                start = datetime.datetime.now()
                
                #if ret and (current_time > 1./FPS):
    # 이미지 테스트, 분류 
                blob = cv2.dnn.blobFromImage(video, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                model.setInput(blob)
                output = model.forward(output_layers)

                h, w = video.shape[0:2]
                img = cv2.resize(video, dsize=(int(video.shape[1]), int(video.shape[0])))
                ih = int(h / 2)
                iw = int(w / 2)

                class_ids = []
                confidences = []
                boxes = []
                for out in output:
                        for detection in out:
                                scores = detection[5:]
                                class_id = np.argmax(scores)
                                conf = scores[class_id]

                                if class_id < 12 and conf > CONF_THR:     # 임계치 0.5
                                        center_x = int(detection[0] * iw)
                                        center_y = int(detection[1] * ih)
                                        w = int(detection[2] * iw)
                                        h = int(detection[3] * ih)
                                        # 사각형 좌표
                                        x = int(center_x - w / 2)
                                        y = int(center_y - h / 2)

                                        boxes.append([x, y, w, h])
                                        confidences.append(float(conf))
                                        class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # 노이즈 제거

                font = cv2.FONT_HERSHEY_PLAIN
                for i in range(len(boxes)):
                        if i in indexes:
                                x, y, w, h = boxes[i]
                                label = str(classes[class_ids[i]])
                                color = colors[i]
                                cv2.line(img, (x, y), (x, y), color, 10)
                                cv2.line(img, (x + w, y + h), (x + w, y + h), color, 10)
                                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                                cv2.putText(img, label, (x, y - 10), font, 3, color, 2)

                end = datetime.datetime.now()

                total = (end - start).total_seconds()

                fps = f'FPS: {1 / total:.2f}'
                cv2.imshow('color', img)
                ##HSV
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # Roi 영역 설정
                roi = hsv[int(100):int(251), int(600):int(751)]

                cv2.imshow('roi', roi)

                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                _, roi_thre = cv2.threshold(roi_gray, 30, 255, cv2.THRESH_BINARY)
                cv2.imshow('roi_thre', roi_thre)
                

                # 초록색
                lower_green = np.array([30,240,30])
                higher_green = np.array([120,255,205])
                # 빨간색
                lower_red = np.array([-10,30,50])
                higher_red = np.array([10,255,255])
                # 노란색
                lower_yellow = np.array([8,100,80])
                higher_yellow = np.array([20,255,255])

                mask_green = cv2.inRange(roi,lower_green, higher_green)
                mask_red = cv2.inRange(roi, lower_red, higher_red)
                mask_yellow = cv2.inRange(roi, lower_yellow, higher_yellow)

                res_red = cv2.bitwise_and(roi, roi, mask=mask_red)
                res_green = cv2.bitwise_and(roi, roi, mask=mask_green)
                res_yellow = cv2.bitwise_and(roi, roi, mask=mask_yellow)
                
                # 색 오픈, 확장
                kernelSz = 3
                shape = cv2.MORPH_RECT
                sz = (2 * kernelSz + 1, 2 * kernelSz + 1)
                SE = cv2.getStructuringElement(shape, sz)

                src_Red_1st_open = cv2.morphologyEx(res_red, cv2.MORPH_OPEN, SE)
                src_Red_2nd_dilate = cv2.morphologyEx(src_Red_1st_open, cv2.MORPH_DILATE, SE)

                src_Yellow_1st_open = cv2.morphologyEx(res_yellow, cv2.MORPH_OPEN, SE)
                src_Yellow_2nd_dilate = cv2.morphologyEx(src_Yellow_1st_open, cv2.MORPH_DILATE, SE)
                
                src_Green_1st_open = cv2.morphologyEx(res_green, cv2.MORPH_OPEN, SE)
                src_Green_2nd_dilate = cv2.morphologyEx(src_Green_1st_open, cv2.MORPH_DILATE, SE)
                                
                cv2.imshow('Red',src_Red_2nd_dilate)
                cv2.imshow('Green',src_Green_2nd_dilate)
                # cv2.imshow('Yellow', src_Yellow_2nd_dilate)
                
                # 면적
                src_Red_2nd_dilate_gray = cv2.cvtColor(src_Red_2nd_dilate, cv2.COLOR_BGR2GRAY)
                src_Green_2nd_dilate_gray = cv2.cvtColor(src_Green_2nd_dilate, cv2.COLOR_BGR2GRAY)
                
                _, src_Red_2nd_dilate_binary = cv2.threshold(src_Red_2nd_dilate_gray, 1, 255, cv2.THRESH_BINARY)
                _, src_Green_2nd_dilate_binary = cv2.threshold(src_Green_2nd_dilate_gray, 1, 255, cv2.THRESH_BINARY)
                cv2.imshow('src_Red_2nd_dilate_binary',src_Red_2nd_dilate_binary)
                cv2.imshow('src_Green_2nd_dilate_gray',src_Green_2nd_dilate_gray)

                contours_red, _ = cv2.findContours(src_Red_2nd_dilate_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_green, _ = cv2.findContours(src_Green_2nd_dilate_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                area_r = 0
                area_g = 0

                for i, contour in enumerate(contours_red):
                        area_r = cv2.contourArea(contour)
                        area_R = f"Red area = {area_r:.1f}"
                
                for i, contour in enumerate(contours_green):
                        area_g = cv2.contourArea(contour)
                        area_G = f"green area = {area_g:.1f}"
                        # print(area_R)
                        # print(area_G)
                if int(area_r) > 1300:
                        print('stop')
                        start_flag = 0
                if int(area_g) > 150:
                        print('go')
                        start_flag = 1
                cv2.imshow('Seoul Traffic Video', img)

                key = cv2.waitKey(1) & 0xFF
                if (key == 27): 
                        exit
                
        except CvBridgeError as e:
                print("Error")




    

if __name__ == '__main__':
        classes = []
        with open(package_path + "/../../../Yolo/coco.names", "rt", encoding="UTF8") as f:
                classes = [line.strip() for line in f.readlines()]
                colors = np.random.uniform(0, 255, size=(len(classes), 3))
                
        # 학습모델과 라벨 설정 
        model = cv2.dnn.readNet(package_path + "/../../../Yolo/yolov3.weights", package_path + "/../../../Yolo/yolov3.cfg")
        layer_names = model.getLayerNames()
        output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
        CONF_THR = 0.6
        prev_time = 0
        FPS = 10
        video = None

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
