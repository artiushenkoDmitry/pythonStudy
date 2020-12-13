import cv2
import numpy as np

glassImage = cv2.imread("../data/images/sunglass.png", cv2.IMREAD_UNCHANGED)
pantherImage = cv2.imread("../data/images/panther.png", cv2.IMREAD_UNCHANGED)

# normalize alpha channels from 0-255 to 0-1
pantherImage = np.float32(pantherImage) / 255
glassImage = np.float32(glassImage) / 255
alpha_background = pantherImage[:, :, 3]
alpha_foreground = glassImage[:, :, 3]

# set adjusted colors
for color in range(0, 3):
    pantherImage[:, :, color] = alpha_foreground * glassImage[:, :, color] + \
                                alpha_background * pantherImage[:, :, color] * (1 - alpha_foreground)

# set adjusted alpha and denormalize back to 0-255
pantherImage[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

# display the image
cv2.imshow("Composited image", pantherImage)
cv2.waitKey(0)
