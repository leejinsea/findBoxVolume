# 최종 완성 본
# key x : 현재 상태의 사진을 찍고 박스 크기[화면에서 상자가 차지하는 비율] 구하기
# key q : 종료

import cv2
import numpy as np

cap = cv2.VideoCapture(1)  # 노트북 웹캠을 카메라로 사용
cap.set(3, 1280)  # 너비
cap.set(4, 720)  # 높이

ret = 0.5724375663071016
mtx = np.array([[496.9596652 ,   0.        , 605.87821196],
       [  0.        , 499.37732365, 355.0369373 ],
       [  0.        ,   0.        ,   1.        ]])
dist = np.array([[-0.23633821,  0.07374086,  0.0005789,
                  -0.0008329 , -0.01166726]])

while cap.isOpened:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1,
                                                      (w, h))
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    # roi = 66, 127, 775, 380
    dst = dst[y+10:y + h-160, x+100:x + w-120] # 값을 더하고 빼고는 일일이 찾은 값
    dst = cv2.resize(dst, (1280, 720))
    if ret:
        cv2.imshow("video", dst)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            img = dst

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_img[:, :, 2] = 0

            # Morphology 닫힘 연산 + 외각선 검출 + 허프 변환
            ## canny
            canny_img = cv2.Canny(img, 0, 50)
            ## Morphology 닫힘 연산
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            close_img = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, k)
            ## 닫힘 이미지에서 외각선 검출
            cont_img = close_img.copy()
            black_img = np.zeros((720, 1280), dtype=np.uint8)
            contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            ## 외각선 중 가장 많은 영역을 차지하는 외각선만 검출
            box_pixel = 0
            main_contour = None
            for contour in contours:
                tree = black_img.copy()
                tree = cv2.drawContours(tree, [contour], -1, (255, 0, 0), -1)
                tree_1ch = np.reshape(tree, (-1))
                _, cnt_pixel = np.unique(tree_1ch, return_counts=True)
                main_contour = contour if box_pixel < cnt_pixel[
                    1] else main_contour
                box_pixel = cnt_pixel[1] if box_pixel < cnt_pixel[
                    1] else box_pixel

            black_img = cv2.drawContours(black_img, [main_contour], -1,
                                         (255, 0, 0), -1)
            if tree_1ch is not None:
                box_size = box_pixel / tree_1ch.size
            else:
                box_size = 0
            cv2.imwrite('dst.jpg', dst)
            print(box_size)
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    else:
        break
