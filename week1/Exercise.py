import cv2
import matplotlib.pyplot as plt
from week1.Script_file.dataPath import DATA_PATH
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)

imagePath = DATA_PATH + "/images/IDCard-Satya.png"
qr_code_image = cv2.imread(imagePath, cv2.IMREAD_COLOR)

qrCodeDetector = cv2.QRCodeDetector()
opencvData, bbox, rectifiedImage = qrCodeDetector.detectAndDecode(qr_code_image)
n = len(bbox)

cv2.rectangle(qr_code_image, (10, 70), (140, 200), (255, 0, 0), thickness=5, lineType=cv2.LINE_8)
# cv2.imshow("main", qr_code_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.subplot(131); plt.imshow(qr_code_image[:,:,::-1]);plt.title("original image")
plt.show()
cv2.imwrite("test.png", qr_code_image)
