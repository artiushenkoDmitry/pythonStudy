import cv2
import matplotlib.pyplot as plt

boyImage = cv2.imread("week1/data/images/boy.jpg", cv2.IMREAD_COLOR)
# draw line
# cv2.line(boyImage, (200, 80), (280, 80), (0, 255, 0), thickness=3, lineType=cv2.LINE_AA);

# draw a circle
# cv2.circle(boyImage, (250, 125), 100, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA);
# нарисовать закрашенный круг
# cv2.circle(boyImage, (250, 125), 100, (0, 0, 255), thickness=-5, lineType=cv2.LINE_AA);

# Draw an ellipse
# Note: Ellipse Centers and Axis lengths must be integers
# cv2.ellipse(boyImage, (250, 125), (100, 50), 0, 0, 360, (255, 0, 0), thickness=3, lineType=cv2.LINE_AA);
# cv2.ellipse(boyImage, (250, 125), (100, 50), 90, 0, 360, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA);

# Draw an ellipse
# Note: Ellipse Centers and Axis lengths must be integers
# Incomplete/Open ellipse
# cv2.ellipse(boyImage, (250, 125), (100, 50), 0, 180, 360, (255, 0, 0), thickness=3, lineType=cv2.LINE_AA);
# Filled ellipse
# cv2.ellipse(boyImage, (250, 125), (100, 50), 0, 0, 180, (0, 0, 255), thickness=-2, lineType=cv2.LINE_AA);

# Draw a rectangle (thickness is a positive integer)
# cv2.rectangle(boyImage, (170, 50), (300, 200), (255, 0, 255), thickness=5, lineType=cv2.LINE_8);

# Put text into image
text = "I am studying"
fontScale = 1.5
fontFace = cv2.FONT_HERSHEY_COMPLEX
fontColor = (250, 10, 10)
fontThickness = 2
cv2.putText(boyImage, text, (20, 350), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA)

plt.imshow(boyImage[:, :, ::-1])
plt.show()
