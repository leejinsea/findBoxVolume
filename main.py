# 상자 크기 측정 원본
# 이미지 화질에 따라 cany thread값 변경

import cv2
import numpy as np


img = cv2.imread('./dst.jpg')
img = cv2.resize(img, (1280, 720))

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
contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

## 외각선 중 가장 많은 영역을 차지하는 외각선만 검출
box_pixel = 0
main_contour = None
for contour in contours:
    tree = black_img.copy()
    tree = cv2.drawContours(tree, [contour], -1, (255, 0, 0), -1)
    tree_1ch = np.reshape(tree, (-1))
    _, cnt_pixel = np.unique(tree_1ch, return_counts=True)
    main_contour = contour if box_pixel < cnt_pixel[1] else main_contour
    box_pixel = cnt_pixel[1] if box_pixel < cnt_pixel[1] else box_pixel


black_img = cv2.drawContours(black_img, [main_contour], -1, (255, 0, 0), -1)
box_size = box_pixel / tree_1ch.size

cv2.imshow('dst', black_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
