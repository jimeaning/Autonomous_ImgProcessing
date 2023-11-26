import cv2
import numpy as np

video = cv2.VideoCapture('./video/tracking_sample.mp4')
#video = cv2.VideoCapture(0)


if not video.isOpened():
    print('Video open failed!')
    exit

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = video.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)


# 초기 트래픽 ROI 설정 
tracker = cv2.TrackerMIL_create()

x = 260
y = 240
w = 70
h = 50

TrafficRoi = (x,y,w,h)

ret, frame = video.read()
if not ret:
    print("프레임을 읽어오지 못했다")
    exit()

tracker.init(frame, TrafficRoi)

while(video.isOpened()):
    
    ret, frame = video.read()

    if not ret: 
        print('Error: Camera')
        break

    
    # 트랙커 설정
    
    ret, TrafficROI = tracker.update(frame)
    
    cv2.rectangle(frame, TrafficROI[:4], (0,0,255), 2)
    cv2.imshow('frame', frame)

    ROI_x = TrafficROI[0]
    ROI_y = TrafficROI[1]
    ROI_w = TrafficROI[2]
    ROI_h = TrafficROI[3]


    TrafficImage = frame[ROI_y:ROI_y + ROI_h, 
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
                
    cv2.imshow('src_Red_2nd_dilate',src_Red_2nd_dilate)
    cv2.imshow('src_Green_2nd_dilate',src_Green_2nd_dilate)
    
    #cv2.imshow('Yellow', src_Yellow_2nd_dilate)
    cv2.imshow('res_red',res_red)
    cv2.imshow('res_green',res_green)
    
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
        break
    
    
    
video.release()
cv2.destroyAllWindows()