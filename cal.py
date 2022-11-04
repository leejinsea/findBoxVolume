# 카메라 촬영 촬영시 c0 부터 이름이 저장되어 c1, c2 ... 식으로 저장
# key x : 촬영
# key q : 종료

import cv2
import numpy as np

cap = cv2.VideoCapture(1)  # 노트북 웹캠을 카메라로 사용
cap.set(3, 1280)  # 너비
cap.set(4, 720)  # 높이qq

cnt = 0

# 사진 저장
while cap.isOpened:
    ret, frame = cap.read()

    if ret:
        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            file_name = f'c{cnt}.jpg'
            cv2.imwrite(file_name, frame)
            cnt += 1
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    else:
        break
