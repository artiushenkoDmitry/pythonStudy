import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

capture = cv2.VideoCapture("videos/chaplin.mp4")

frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
frame_index = 1
if not capture.isOpened():
    print("Error")

while capture.isOpened():
    ret, frame = capture.read()

    if ret:
        cv2.imshow("input frame", frame)

        if cv2.waitKey(10) & 0xFF == ord('m'):
            break

    else:
        break
capture.release()
cv2.destroyAllWindows()


# capture = cv2.VideoCapture("data/videos/chaplin.mp4")
# frame_width = capture.get(3)
# frame_height = capture.get(4)

# Show video in separate window, and exit when pressed "m"
# if not capture.isOpened():
#     print("Error")
#
# while capture.isOpened():
#     ret, frame = capture.read()
#
#     if ret is True:
#         cv2.imshow("input frame", frame)
#         grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow("gray frame", grayFrame)
#
#         if cv2.waitKey(10) & 0xFF == ord('m'):
#             break
#     else:
#         break
# capture.release()
# cv2.destroyAllWindows()

# Save video to file system
# out = cv2.VideoWriter('outputChaplin.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (int(frame_width), int(frame_height)))
#
#
# while capture.isOpened():
#     ret, frame = capture.read()
#
#     if ret:
#         out.write(frame)
#         cv2.waitKey(25)
#     else:
#         break
# capture.release()
# out.release()
# cv2.destroyAllWindows()


# capture = cv2.VideoCapture(0)
#
# if not capture.isOpened():
#     print("Error")
#
# while capture.isOpened():
#     ret, frame = capture.read()
#
#     if ret is True:
#         cv2.imshow("input frame", frame)
#
#         if cv2.waitKey(10) & 0xFF == ord('m'):
#             break
#     else:
#         break
# capture.release()
# cv2.destroyAllWindows()


# def draw_circle(action, x, y, flags, userdata):
#     global center, circumference
#
#     if action == cv2.EVENT_LBUTTONDOWN:
#         center = [(x, y)]
#         cv2.circle(source, center[0], 1, (255, 255, 0), 2, cv2.LINE_AA)
#
#     elif action == cv2.EVENT_LBUTTONUP:
#         circumference = [(x, y)]
#         radius = math.sqrt(math.pow(center[0][0] - circumference[0][0], 2) +
#                            math.pow(center[0][1] - circumference[0][1], 2))
#         cv2.circle(source, center[0], int(radius), (0, 255, 0), 2,
#                    cv2.LINE_AA)
#         cv2.imshow("Window", source)
#
#
# # Lists to store the points
# center = []
# circumference = []
#
# source = cv2.imread("data/images/sample.jpg")
# cv2.namedWindow("Window")
# cv2.setMouseCallback("Window", draw_circle)
# k = 0
# while k != 27:
#     cv2.imshow("Window", source)
#     cv2.putText(source, '''Choose center, and drag,
#                           Press ESC to exit and c to clear''',
#                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (255, 255, 255), 2)
#     k = cv2.waitKey(20) & 0xFF
# cv2.destroyAllWindows()


