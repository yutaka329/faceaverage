import cv2

img = cv2.imread("Images/jimmy-carter.jpg")
for i in range(1000):
    cv2.imwrite("Images/{}.jpg".format(i), img)