# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import math
#import mathplotlib as plt
def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def peresech(mass,obl):#определяет пересекает ли область линии
    k=-1
    for i in range(0,len(mass)):
        #if mass[i][0]>=obl[1][0] and mass[i][1]>=obl[1][1] and mass[i][2]<=obl[0][0] and mass[i][3]<=obl[0][1]:
         if obl[0][0]>mass[i][0] and obl[1][0]<mass[i][2]:#по крайним границам линии по х
             if obl[0][1] < mass[i][3] and obl[2][1] > mass[i][1]:  # по крайним границам линии по у надо доделать
                k=i
                break
    #print(k)
    return k

#Для обработки видео
video=cv2.VideoCapture("rat.mp4")#0 значит с веб камеры
if not video.isOpened():
    print("error")
while video.isOpened():
    _,image=video.read()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur=cv2.GaussianBlur(gray,(7,7),2)

    canny_img=cv2.Canny(blur,30,70)#1 k 3 or 1k 2
    canny_img = cv2.GaussianBlur(canny_img, (3, 3), 0)
    canny_img1=canny_img.copy()
    _,tresh=cv2.threshold(canny_img,90,200,cv2.THRESH_BINARY)
    #viewImage(tresh, "linetresh ")
    linesP=cv2.HoughLinesP(canny_img,1,np.pi/180,50,np.array([()]),60,10)
    j=0
    lenese=[]
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            if (l[1]>400) and(l[1]<1000):
                #print(l[0], l[1], l[2], l[3])
                lenese.append([0]*4)
                lenese[j][0]=l[0]
                lenese[j][1] = l[1]
                lenese[j][2]=l[2]
                lenese[j][3]=l[3]
                j+=1
    lenese=sorted(lenese,key=lambda x:x[1])
    lenese1 = sorted(lenese, key=lambda x: x[3])
    j=1
    l=lenese[0]
    cv2.line(rgb_image, (l[0], l[1]), (l[2], l[3]), (120, 60, 60), 3, cv2.LINE_AA)
    str(j)
    #cv2.putText(rgb_image, str(j), (l[0], l[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #lback1=l
    lback = l
    """kofficient=[]
    kofficient.append([0] * 4)
    kofficient[j-1][0]=l[3]-l[1]
    kofficient[j - 1][1] = -(l[2] - l[0])
    kofficient[j - 1][2] = l[0]*l[3] - l[1]*l[2]"""
    linline=[]
    linline.append([0] * 5)
    linline[j - 1][0] = l[0]
    linline[j - 1][1] = l[1]
    linline[j - 1][2] = l[2]
    linline[j - 1][3] = l[3]
    for i in range(1, len(lenese)):
        l=lenese[i]
        l1=lenese1[i]
        if abs(lback[0]-l[0])>100 or abs(lback[2]-l[2])>100 and abs(lback[1]-l[1])>10:
            j+=1
            cv2.line(rgb_image, (l[0], l[1]), (l[2], l[3]), (120, 60, 60), 3, cv2.LINE_AA)
            str(j)
            #cv2.putText(rgb_image, str(j), (l[0], l[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            linline.append([0]*5)
            linline[j-1][0] = l[0]
            linline[j-1][1] = l[1]
            linline[j-1][2] = l[2]
            linline[j-1][3] = l[3]
            """ kofficient.append([0] * 4)
            kofficient[j - 1][0] = l[3] - l[1]
            kofficient[j - 1][1] = -(l[2] - l[0])
            kofficient[j - 1][2] = l[0] * l[3] - l[1] * l[2]"""
            #lback1=l1
        #print(l,j)
    viewImage(rgb_image,"линии")
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    break
vpr=-1
print(linline)
while video.isOpened():
    _,img=video.read()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    tresh=cv2.GaussianBlur(gray,(5,5),0)
    tresh=cv2.Canny(tresh,50,150)
    _,tresh=cv2.threshold(gray,95,255,cv2.THRESH_BINARY_INV)
    contours0, hierarchy0 = cv2.findContours( tresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours0:
        rect = cv2.minAreaRect(cnt) # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
        box = np.intp(box) # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        if area>450 and area<4500:
            #print(box)
            cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
            v=peresech(linline,box)
            if v!=-1 and vpr!=v:
                linline[v][4]+=1
                vpr=v
                print(v,linline[v][4])
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.imshow('video',img)
    if cv2.waitKey(40) & 0xFF==ord('q'):
        video.release()
        cv2.destroyAllWindows()

cv2.destroyAllWindows()
video.release()