import cv2
import numpy as np
import datetime

classes = []
with open("./dnn/coco.names", "rt", encoding="UTF8") as f:
    classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 학습모델과 라벨 설정 
model = cv2.dnn.readNet("./dnn/yolov3.weights", "./dnn/yolov3.cfg")
layer_names = model.getLayerNames()
output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
#output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

video = cv2.VideoCapture('./video/test3.mp4')
# video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_POS_MSEC, 300)

print( video.get(cv2.CAP_PROP_FPS))
CONF_THR = 0.4
prev_time = 0
FPS = 10

boxes = []


while(video.isOpened()):
    start = datetime.datetime.now()

    ret, frame = video.read()

    if not ret: 
        print('Error: Camera')
        break

    # ROI 횡단보도 영역
    roi_ppl = frame[int(300):int(451), int(400):int(801)]
    cv2.imshow('roi_ppl', roi_ppl)

    #if ret and (current_time > 1./FPS):
    # 이미지 테스트, 분류 
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    output = model.forward(output_layers)

    h, w = frame.shape[0:2]
    img = cv2.resize(frame, dsize=(int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
    ih = int(h / 2)
    iw = int(w / 2)

    class_ids = []
    confidences = []
    
    x, y = 0, 0
    for out in output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if class_id == 0 and conf > CONF_THR:
                print('Detected Person : Stop')
                
            # Yolo : 신호등 객체 탐지
            if class_id == 9 and conf > CONF_THR:
                center_x = int(detection[0] * iw)
                center_y = int(detection[1] * ih)
                w = int(detection[2] * iw)
                h = int(detection[3] * ih)
                # 사각형 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # boxes.append([x, y, w, h])
                boxes = [x, y, w, h]
                confidences.append(float(conf))
                class_ids.append(class_id)
                
    # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # 노이즈 제거

    # font = cv2.FONT_HERSHEY_PLAIN
    # for i in range(len(boxes)):
    #     if i in indexes:
    #         x, y, w, h = boxes[i]
    #         label = str(classes[class_ids[i]])
    #         color = colors[i]
    #         cv2.line(img, (x, y), (x, y), color, 10)
    #         cv2.line(img, (x + w, y + h), (x + w, y + h), color, 10)
    #         cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    #         cv2.putText(img, label, (x, y - 10), font, 3, color, 2)

    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    fps = f'FPS: {1 / total:.2f}'
    print(fps)

    ##HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # ROI 신호등 영역 설정 x,y - 왼쪽 위x : x + w // y:y+h
    roi = hsv.copy()
    
    roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0
    if boxes:
        roi_x = boxes[0]
        roi_y = boxes[1]
        roi_w = boxes[2]
        roi_h = boxes[3]
        
        roi = hsv[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        cv2.imshow('roi', roi)
    
    print(boxes)
    

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
    cv2.imshow('Yellow', src_Yellow_2nd_dilate)
    
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
        #print(area_R)
    
    for i, contour in enumerate(contours_green):
        area_g = cv2.contourArea(contour)
        area_G = f"green area = {area_g:.1f}"
        #print(area_G)
    # if int(area_r) > 1300:
    #     print('stop')
    
    # if int(area_g) > 1000:
    #     print('go')

    cv2.imshow('Seoul Traffic Video', img)

    # ESC를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if (key == 27): 
        break

video.release()
cv2.destroyAllWindows()
