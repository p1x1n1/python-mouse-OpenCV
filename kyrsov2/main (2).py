# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



import cv2
import numpy as np
#import mathplotlib as plt
def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#задаем координаты для нашей прямой
def make_coordinates(image, line_parameters):
    # Y = MX + B
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

#усреднение линий
def average_slope_intercept(image, lines):

    left_fit = []
    right_fit = []

    while lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)

        left_line = make_coordinates(image, left_fit_average)
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)

        return np.array([left_line, right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 100, 150)
    return canny

#отображение самих линий
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 15)
    return line_image

"""#наложение маски на изображение
#Дорога для водителя выглядит как трапеция
def mask(image):
    height = image.shape[0]
    polygons = np.array([(200, height), (1100, height), (1000, 200),(0,200)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.array([polygons], dtype=np.int64), 1024)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image"""
"""
#Для обработки видео
video=cv2.VideoCapture("путь к видео")#0 значит с веб камеры
if not video.isOpened():
    print("error")

cv2.waitKey(1)#задержка в ms

while video.isOpened():
    frame=video.read()

    viewImage(frame)
    
"""

def empty(a):
    pass
#def hsv_mass(image):




"""image = cv2.imread("rat.mp4")"""
#Для обработки видео
video=cv2.VideoCapture("rat.mp4")#0 значит с веб камеры
if not video.isOpened():
    print("error")

cv2.waitKey(1)#задержка в ms
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat min","TrackBars",0,255,empty)
cv2.createTrackbar("Sat max","TrackBars",255,255,empty)
cv2.createTrackbar("Val min","TrackBars",0,255,empty)
cv2.createTrackbar("Val max","TrackBars",255,255,empty)

while video.isOpened():
    _,image=video.read()
    #viewImage(image)
    imageHSV=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    while True:
        h_mn = cv2.getTrackbarPos("Hue min","TrackBars")
        h_mx = cv2.getTrackbarPos("Hue max","TrackBars")
        s_mn = cv2.getTrackbarPos("Sat min","TrackBars")
        s_mx = cv2.getTrackbarPos("Sat max","TrackBars")
        v_mn = cv2.getTrackbarPos("Val min","TrackBars")
        v_mx = cv2.getTrackbarPos("Val max","TrackBars")
        print(h_mn,s_mn,v_mn,h_mx, s_mx, v_mx)
        lover=np.array([h_mn,s_mn,v_mn])
        upper = np.array([h_mx, s_mx, v_mx])
        mask = cv2.inRange(imageHSV, lover, upper)
        cv2.namedWindow("гауссаво размытие ", cv2.WINDOW_NORMAL)
        cv2.imshow( "гауссаво размытие ",mask)
        cv2.waitKey(1)#задержка в ms"""
    """    lover = np.array([0, 137, 0])
    upper = np.array([255, 160, 240])
    mask = cv2.inRange(imageHSV, lover, upper)
    cv2.namedWindow("гауссаво размытие ", cv2.WINDOW_NORMAL)
    cv2.imshow("гауссаво размытие ", mask)
    cv2.waitKey(0)  # задержка в ms"""

    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #0 137 0 255 105 162
    #0 137 0 255 160 240
#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#ret, threshold_image = cv2.threshold(gray_image, 127, 255, 0)
#viewImage(gray_image, " в градациях серого")
#viewImage(threshold_image, "белый ")
#
#image=mask(image)
    blur=cv2.GaussianBlur(gray,(7,7),2)
#viewImage(blur, "гауссаво размытие ")
    canny_img=cv2.Canny(blur,30,70)#1 k 3 or 1k 2
    canny_img = cv2.GaussianBlur(canny_img, (3, 3), 0)
    _,tresh=cv2.threshold(canny_img,200,215,cv2.THRESH_BINARY)
    viewImage(tresh, "гауссаво размытие ")
#linesP=cv2.HoughLinesP(canny_img,1,np.pi/180,100,np.array([()]),minLineLength=10,maxLineGap=3)
    linesP=cv2.HoughLinesP(canny_img,1,np.pi/180,50,None,60,5)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(rgb_image, (l[0], l[1]), (l[2], l[3]), (0, 120, 255), 3, cv2.LINE_AA)
    viewImage(rgb_image,"линии")
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    break
"""lines=cv2.HoughLines(canny_img,np.pi/180,150,None,0,0)
if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)"""
#viewImage(image,"линии")
#aver_lines=average_slope_intercept(canny_img,lines)
#line_image=display_lines(rgb_image,aver_lines)
#combo=cv2.addWeighted(rgb_image,0.8,line_image,0.5,1)
#viewImage(combo, "гауссаво размытие ")