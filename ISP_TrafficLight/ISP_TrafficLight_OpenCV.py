import cv2
import time

video = cv2.VideoCapture('./video/Seoul_Traffic.mp4')

prev_time = 0
FPS = 10

while(video.isOpened()):
    ret, frame = video.read()

    current_time = time.time() - prev_time  # 시간 지연 예방

    if ret and (current_time > 1./FPS):
        cv2.imshow('Seoul Traffic Video', frame)

    # ESC를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if (key == 27): 
        break

video.release()
cv2.destroyAllWindows()