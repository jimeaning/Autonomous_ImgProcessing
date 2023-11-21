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
#output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]

# video = cv2.VideoCapture('./video/Seoul_Traffic.mp4')
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONF_THR = 0.4
prev_time = 0
FPS = 10

while(video.isOpened()):
    start = datetime.datetime.now()

    ret, frame = video.read()

    if not ret: 
        print('Error: Camera')
        break

    #if ret and (current_time > 1./FPS):
    # 이미지 테스트, 분류 
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    output = model.forward(output_layers)

    h, w = frame.shape[0:2]
    img = cv2.resize(frame, dsize=(int(frame.shape[1]), int(frame.shape[0])))
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

    ## 신호등 구분
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 초록색
    lower_green = np.array([50,50,80])
    higher_green = np.array([90,255,255])
    # 빨간색
    lower_red = np.array([-10,30,50])
    higher_red = np.array([10,255,255])
    # 노란색
    lower_yellow = np.array([11,100,80])
    higher_yellow = np.array([30,255,255])

    mask_green = cv2.inRange(hsv,lower_green, higher_green)
    mask_red = cv2.inRange(hsv, lower_red, higher_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, higher_yellow)

    res_red = cv2.bitwise_and(img,img, mask=mask_red)
    res_green = cv2.bitwise_and(img, img, mask=mask_green)
    res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)

    cv2.imshow('Green',res_green)
    cv2.imshow('Red',res_red)
    cv2.imshow('Yellow', res_yellow)

    cv2.imshow('Seoul Traffic Video', img)

    # ESC를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if (key == 27): 
        break

video.release()
cv2.destroyAllWindows()