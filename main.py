import cv2
import matplotlib.pyplot as plt
import numpy as np
from Script_file.dataPath import DATA_PATH

imagePath = DATA_PATH + "images/number_zero.jpg"
testImage = cv2.imread(imagePath, cv2.IMREAD_COLOR)
muskImage = cv2.imread("./data/images/musk.jpg")
pantherImage = cv2.imread("./data/images/panther.png", cv2.IMREAD_UNCHANGED)
boyImage = cv2.imread("./data/images/boy.jpg")

# image is a matrix
# testImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
# # так определяется тип данных
# print("Data type = {}\n".format(testImage.dtype))
# # так определяется тип
# print("Data type = {}\n".format(type(testImage)))
# # так определяется размер
# print("Data type = {}\n".format(testImage.shape))

# manipulating pixels
# testImage[y, x] - not testImage[x, y]
# print(testImage[0, 0])
# testImage[0, 0] = 200
# print(testImage)

# Manipulating group of pixels ROI (region of interest)
# test_roi = testImage[0:2, 0:4]
# print("Original matrix\n{}\n".format(testImage))
# print("Selected region\n{}\n".format(test_roi))
# testImage[0:2, 0:4] = 111
# print("Modified image\n{}\n".format(testImage))

# Displaying the image
# plt.imshow(testImage)
# plt.colorbar()
# plt.show()
# cv2.imshow("main", testImage)
# cv2.waitKey(0)
# cv2.destroyWindow(testImage)
# cv2.destroyAllWindows()

# Additional Display Utility Functions
# window = cv2.namedWindow("100500", cv2.WINDOW_NORMAL)
# cv2.waitKey(0)
# cv2.destroyWindow(window)
# cv2.destroyAllWindows()

# Saving an image
# cv2.imwrite("100500.jpg", testImage)

# Color images
# print("image dimension = {}".format(muskImage.shape))
# plt.imshow(muskImage)
# plt.imshow(muskImage[:,:,::-1])
# plt.title("some title")
# plt.imshow(muskImage[:,:,1])
# plt.show()

# Splitting and merging images
# b, g, r = cv2.split(muskImage)
# plt.imshow(g)
# imgMerged = cv2.merge((b, g, r))
# plt.imshow(imgMerged[:, :, ::-1])
# plt.show()

# Manipulating color pixels
# print(testImage[0,0])
# testImage[0,0] = [0, 255, 255]
# testImage[1,1]=[255,255,0]
# testImage[2,2]=[255,0,255]
# plt.imshow(testImage[:,:,::-1])
# plt.show()

# Modify region of interest
# testImage[0:3, 0:3] = [0, 255, 255]
# testImage[3:6, 0:3] = [0, 255, 0]
# testImage[6:9, 0:3] = [0, 0, 255]
# plt.imshow(testImage[:,:,::-1])
# plt.show()

# Images with alpha chanel
# print("image Dimension = {}".format(pantherImage.shape))
# imgBGR = pantherImage[:, :, 0:3]
# imgMask = pantherImage[:, :, 3]
# plt.imshow(imgBGR[:,:,::-1])
# plt.imshow(imgMask, cmap='gray')
# plt.show()

# Basic image operations
# Create new image
# plt.imshow(boyImage[:, :, ::-1])
# plt.show()
# Copy image
# imageCopy = boyImage.copy()
# plt.imshow(boyImage[:, :, ::-1])
# plt.show()
# emptyMatrix = np.zeros((100,200,3),dtype='uint8') # uint8 - unsigned integer 8 bit standart image format
# plt.imshow(emptyMatrix)
# plt.show()
# emptyMatrix = 100*np.ones((100,200,3),dtype='uint8') - создает матрицу заполненную 1 их можно умножить на какое-нить число
# plt.imshow(emptyMatrix)
# plt.show()
# emptyMatrix = 100*np.ones_like(boyImage) #- создает матрицу размером таким-же как существующая картинка
# plt.imshow(emptyMatrix)
# plt.show()

# Crop image
# crop = boyImage[40:200, 170:320]
# plt.imshow(crop[:, :, ::-1])
# plt.show()

# Copying a Region to another
# copyRoi=boyImage[40:200, 170:320]
# # находим высоту и ширину ROI
# roiHeight, roiWidth = copyRoi.shape[:2]
# copiedImage = boyImage.copy()
# # добавляем копию слева
# copiedImage[40:40+roiHeight, 10:10+roiWidth] = copyRoi
# # добавляем копию справа
# copiedImage[40:40+roiHeight, 330:330+roiWidth] = copyRoi
# plt.imshow(copiedImage[...,::-1])
# plt.show()

# Resizing an Image
# Method1 - Specify width and height
# resizeDownWidth = 300
# resizeDownHeight = 200
# resizedDown = cv2.resize(boyImage, (resizeDownWidth, resizeDownHeight), interpolation=cv2.INTER_LINEAR)
#
# resizeUpWidth = 600
# resizeUpHeight = 900
# resizedUp = cv2.resize(boyImage, (resizeUpWidth, resizeUpHeight), interpolation=cv2.INTER_LINEAR)
#
# plt.figure(figsize=[15,15])
# plt.subplot(131); plt.imshow(boyImage[:,:,::-1]);plt.title("original image")
# plt.subplot(132); plt.imshow(resizedUp[:,:,::-1]);plt.title("resized up")
# plt.subplot(133); plt.imshow(resizedDown[:,:,::-1]);plt.title("resized down")
# plt.show()

# Method2 - Specify scaling factor
# scaleUpX = 1.5
# scaleUpY = 1.5
#
# scaleDown = 0.6
#
# scaleDown = cv2.resize(boyImage, None, fx= scaleDown, fy= scaleDown, interpolation=cv2.INTER_LINEAR)
# scaleUp = cv2.resize(boyImage, None, fx= scaleUpX, fy= scaleUpY, interpolation=cv2.INTER_LINEAR)
#
# plt.figure(figsize=[15,15])
# plt.subplot(121); plt.imshow(scaleDown[...,::-1]);plt.title("Scaled down image, size = {}".format(scaleDown.shape[:2]))
# plt.subplot(122); plt.imshow(scaleUp[...,::-1]);plt.title("Scaled up image, size = {}".format(scaleDown.shape[:2]))
# plt.show()

# Creating an Image Mask
# Create a mask using coordinates
# mask1 = np.zeros_like(boyImage)
# mask1[50:200,170:320] = 255
# plt.figure(figsize=[15,15])
# plt.subplot(121); plt.imshow(boyImage[...,::-1]);plt.title("Original image")
# plt.subplot(122); plt.imshow(mask1[...,::-1]);plt.title("Mask")
# print(mask1.dtype)
# plt.show()

# Create a mask using coordinates
# mask2 = cv2.inRange(boyImage, (0,0,150), (100,100,250))
# plt.subplot(121); plt.imshow(boyImage[...,::-1]);plt.title("Original image")
# plt.subplot(122); plt.imshow(mask2, cmap='gray');plt.title("Mask")
# print(mask2.dtype)
# plt.show()

# Datatype Conversion
# scalingFactor = 1/255.0
# # Convert unsigned  int to float
# image = np.float32(boyImage)
# # Scale the values so that thay lie between [0,1]
# image = image * scalingFactor
# # Convert back to unsigned int
# image = image * (1/scalingFactor)
# image = np.uint8(image)
# plt.imshow(image[...,::-1])
# plt.show()

# Contrast Enhancement
# contrastPercentage = 30.0
# # contrastHigh = boyImage * (1+contrastPercentage/100)
# # plt.figure(figsize=[20,20])
# # plt.subplot(121);plt.imshow(boyImage[...,::-1]);plt.title("original image")
# # plt.subplot(122);plt.imshow(contrastHigh[...,::-1]);plt.title("High contrast")
# # plt.show()
# # Clip values to [0,255] and change it back to uint8 for display
# contrastImage = boyImage * (1+contrastPercentage/100)
# clippedContrastImage = np.clip(contrastImage, 0, 255)
# contrastHighClippedUint8 = np.uint8(clippedContrastImage)
# # Convert the range to [0,1] and keep it in the float format
# contrastHighNormalized = (boyImage * 1+contrastPercentage/100)/255
# contrastHighNormalizedClipped = np.clip(contrastHighNormalized, 0, 1)
# plt.figure(figsize=[20,20])
# plt.subplot(131);plt.imshow(boyImage[...,::-1]);plt.title("original image")
# plt.subplot(132);plt.imshow(contrastHighClippedUint8[...,::-1]);plt.title("Converted back to uint8")
# plt.subplot(133);plt.imshow(contrastHighNormalizedClipped[...,::-1]);plt.title("Normalized float to [0,1]")
# plt.show()

# Brightness Enhancement
# brightnessOffset = 50
# # Add the offset for increasing brightness
# brightHighOpenCV = cv2.add(boyImage, np.ones(boyImage.shape, dtype='uint8')*brightnessOffset)
# brightHighInt32 = np.int32(boyImage) + brightnessOffset
# brightHighInt32Clipped = np.clip(brightHighInt32, 0, 255)
# #Display the outputs
# plt.figure(figsize=[20,20])
# plt.subplot(131);plt.imshow(boyImage[...,::-1]);plt.title("Original image")
# plt.subplot(132);plt.imshow(brightHighOpenCV[...,::-1]);plt.title("Using cv2 and function")
# plt.subplot(133);plt.imshow(brightHighInt32Clipped[...,::-1]);plt.title("Using numpy and clipped")
# plt.show()
# Add the offset for increasing brightness
# brightHighFloat32 = np.float32(boyImage)+brightnessOffset
# brightHighFloat32NormalizedClipped = np.clip(brightHighFloat32/255,0,1)
# brightHighFloat32NormalizedClippedUint8 = np.uint8(brightHighFloat32NormalizedClipped*255)
# # Display the outputs
# plt.figure(figsize=[20,20])
# plt.subplot(131);plt.imshow(boyImage[...,::-1]);plt.title("Original image")
# plt.subplot(132);plt.imshow(brightHighFloat32NormalizedClipped[...,::-1]);plt.title("Using np.float32 and clipping")
# plt.subplot(133);plt.imshow(brightHighFloat32NormalizedClippedUint8[...,::-1]);plt.title("Using int->float->int and clipping")
# plt.show()