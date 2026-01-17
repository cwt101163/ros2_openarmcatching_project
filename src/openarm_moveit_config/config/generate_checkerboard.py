# generate_checkerboard.py
import numpy as np
import cv2

cols = 4   # 横向方格数
rows = 4   # 纵向方格数
pixel_per_square = 100  # 每格 100 像素
width = cols * pixel_per_square   # 400
height = rows * pixel_per_square  # 400

checker = np.ones((height, width, 3), dtype=np.uint8) * 255

for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            x1, x2 = j * pixel_per_square, (j + 1) * pixel_per_square
            y1, y2 = i * pixel_per_square, (i + 1) * pixel_per_square
            checker[y1:y2, x1:x2] = [0, 0, 0]

cv2.imwrite("checkerboard_4x4.png", checker)
print("已生成 4×4 棋盘格图像（400×400 像素）")