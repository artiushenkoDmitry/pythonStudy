import cv2

#Сохранение видео файла
cap = cv2.VideoCapture('MyInputVid.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
success, frame = cap.read()
while success:
    videoWriter.write(frame)
    success, frame = cap.read()
# Проигрывание видео файла
# if(cap.isOpened()==False):
#     print('Error opening video stream or file')
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         cv2.imshow('Frame', frame)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()