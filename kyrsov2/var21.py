# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np



def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
[33, 853, 141, 795, 0, -0.537] 6
[[ 25 822]
 [ 47 800]
 [ 72 825]
 [ 50 847]]
[[ 27 823]
 [ 45 799]
 [ 73 820]
 [ 55 844]] 6

 [526, 580, 645, 514, 0, -0.555] 5
 [[611 507]
 [628 496]
 [645 523]
 [627 534]] 5


 [[ 387 1024]
 [ 460  994]
 [ 471 1024]
 [ 399 1053]]
Количество пересечений линии  12 : 1
[[ 398 1028]
 [ 464  992]
 [ 481 1023]
 [ 415 1060]]
Количество пересечений линии  13 : 1
[[ 413 1023]
 [ 480  962]
 [ 504  987]
 [ 437 1049]]
Количество пересечений линии  12 : 2

[385, 746, 534, 1026, 0, 1.879] 12
[381, 748, 529, 1028, 0, 1.892] 13
"""


def peresech(lin, obl):  # определяет пересекает ли область линии
    k = -1
    # print(obl)
    for i in range(0, len(lin)):
        # if mass[i][0]>=obl[1][0] and mass[i][1]>=obl[1][1] and mass[i][2]<=obl[0][0] and mass[i][3]<=obl[0][1]:
        if (obl[0][0] >= lin[i][0] or obl[3][0] >= lin[i][0]) and (obl[1][0]) <= lin[i][
            2]:  # по крайним границам линии по х
            if (obl[0][1] <= lin[i][3] and (obl[2][1]) >= lin[i][1]) or (
                    obl[0][1] >= lin[i][3] and (obl[2][1]) <= lin[i][
                1]):  # по крайним границам линии по у надо доделать
                k = i
                break
    # print(k)
    return k


# Для обработки видео
video = cv2.VideoCapture("rat.mp4")  # 0 значит с веб камеры
if not video.isOpened():
    print("error")
while video.isOpened():
    _, image = video.read()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # viewImage(rgb_image, "rgb ")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # viewImage(gray, "linetresh ")
    blur = cv2.GaussianBlur(gray, (7, 7), 2)
    # viewImage(gray, "linetresh ")
    canny_img = cv2.Canny(blur, 30, 70)  # 1 k 3 or 1k 2
    # viewImage(canny_img, "сфттн ")
    canny_img = cv2.GaussianBlur(canny_img, (3, 3), 0)
    canny_img1 = canny_img.copy()
    _, tresh = cv2.threshold(canny_img, 80, 200, cv2.THRESH_BINARY)
    # viewImage(tresh, "linetresh ")

    # lines = cv2.HoughLines(tresh, 1, np.pi / 180, 100, np.array([]))
    '''for line in lines:
        # for i in range(0, len(lines)):
        rho, theta = line[0]

        a = math.cos(theta)
        b = math.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (1))
        if (y1 < 400) and (y1 > 1000):
            cv2.line(rgb_image1, (x1, y1), (x2, y2), (0, 0, 255), 2)
    viewImage(rgb_image1, "линии")'''
    # houghlinesP
    linesP = cv2.HoughLinesP(tresh, 1, np.pi / 180, 50, np.array([()]), 80, 10)
    j = 0
    lenese = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            if (l[1] > 400) and (l[1] < 1000):
                # print(l[0], l[1], l[2], l[3])
                lenese.append([0] * 5)
                lenese[j][0] = l[0]
                lenese[j][1] = l[1]
                lenese[j][2] = l[2]
                lenese[j][3] = l[3]
                lenese[j][4] = round((l[3] - l[1]) / (l[2] - l[0]), 3)
                j += 1

    lenese = sorted(lenese, key=lambda x: x[4])
    j = 0
    for l in lenese:
        j += 1
        cv2.putText(rgb_image2, str(j), (l[0], l[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.line(rgb_image2, (l[0], l[1]), (l[2], l[3]), (142, 107, 255), 2, cv2.LINE_AA)
        print(l, j)
    viewImage(rgb_image2, "+++")
    l = lenese[0]
    cv2.line(rgb_image, (l[0], l[1]), (l[2], l[3]), (142, 107, 255), 2, cv2.LINE_AA)
    lback = l
    linline = []
    linline.append([0] * 6)
    linline[0][0] = l[0]
    linline[0][1] = l[1]
    linline[0][2] = l[2]
    linline[0][3] = l[3]
    linline[0][5] = l[4]
    k=0
    for i in range(1, len(lenese)):#сортировка линий по угловому коэффициенту
        t = 0
        l = lenese[i]
        for j in range(0, len(linline)):
            ll = linline[j]
            if abs(ll[5] - l[4]) <= 0.06 and (l[0] != ll[0] or l[2] != ll[2]) and (
                    abs(l[0] - ll[0]) < 100 or abs(l[2] - ll[2]) < 100) and t == 0:
                if (l[0] <= ll[0]) and (l[2] >= ll[2]):
                    ll[0] = l[0]
                    ll[1] = l[1]
                    ll[2] = l[2]
                    ll[3] = l[3]
                    ll[5] = l[4]
                    t = 1
                t = 1
        if t == 0:
            k+=1
            linline.append([0] * 6)
            linline[k][0] = l[0]
            linline[k][1] = l[1]
            linline[k][2] = l[2]
            linline[k][3] = l[3]
            linline[k][5] = l[4]
    j = 1
    for l in linline:
        j += 1
        print(l, j)
    """lenese = sorted(lenese, key=lambda x: x[4])
    j=1
    l=lenese[0]
    cv2.line(rgb_image, (l[0], l[1]), (l[2], l[3]), (142, 107, 255), 2, cv2.LINE_AA)
    str(j)
    #cv2.putText(rgb_image, str(j), (l[0], l[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #lback1=l
    #kofficient=[]
    #kofficient.append([0] * 4)

    #kofficient[j - 1][1] = -(l[2] - l[0])
    #kofficient[j-1][0]=(l[3]-l[1])/(l[2] - l[0])#при х
    #kofficient[j - 1][2] = l[0]*l[3] - l[1]*l[2]
    lback = l
    linline=[]
    linline.append([0] * 6)
    linline[j - 1][0] = l[0]
    linline[j - 1][1] = l[1]
    linline[j - 1][2] = l[2]
    linline[j - 1][3] = l[3]
    linline[j - 1][5] = l[4]
    for l in lenese:
        # l1=lenese1[i]
        if abs(lback[0] - l[0]) > 100 and abs(lback[2] - l[2]) > 100 and abs(lback[1] - l[1]) > 10:
            lback = l
        else:
            lenese.remove(l)"""
    k = 1
    i = 1
    k = 0
    while i < len(linline):#отбрасывание линий находящися за границами круга
        l = linline[i]
        if l[5] > 2:
            linline.remove(l)
        i += 1
    for i in range(0, len(linline)):
        l = linline[i]
        cv2.line(rgb_image, (l[0], l[1]), (l[2], l[3]), (142, 107, 255), 3, cv2.LINE_AA)
        cv2.putText(rgb_image, str(i + 1), (l[0], l[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    viewImage(rgb_image, "линии")
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    break
vpr = -1
print(linline)
k = 0
for l in linline:
    k += 1
    print(l, k)
k = 0
for l in linline:
    k += 1
    print(l, k)
"""[352, 485, 355, 403, 0, -27.333] 1
[384, 613, 512, 408, 0, -1.602] 2
[225, 869, 303, 744, 0, -1.603] 3
[526, 580, 645, 514, 0, -0.555] 4
[32, 852, 162, 783, 0, -0.531] 5
[11, 670, 97, 671, 0, 0.012] 6
[614, 682, 695, 683, 0, 0.012] 7
[531, 789, 670, 872, 0, 0.597] 8
[96, 529, 180, 580, 0, 0.607] 9
[204, 404, 315, 612, 0, 1.874] 10
[385, 746, 534, 1026, 0, 1.879] 11"""
#радиусы эллипсов
dia1=linline[6][3]-linline[5][0]#внешний эллипс по горизонтали
dia2=3*(dia1//5)# #средний эллипс
dia3=dia1//5#внутренний эллипс
print(dia1,dia2,dia3)
for i in range(0, len(linline)):
    l = linline[i]
    t = 0
    for lback in linline:
        if abs(lback[5] - l[5]) < 0.03 and abs(lback[5] - l[5]) != 0:
            t = 1
            break
    if t == 0:
        lin = [0] * 6
        lin[5] = l[5]
        linb = int((-l[0] * l[3] + l[0] * l[1]) / (l[2] - l[0]) + l[2])
        if l[0] == 352:
            lin[0] = l[0]
            lin[1] = int(lin[5] * l[0] + linb)
            lin[2] = l[2]
            lin[3] = int(lin[5] * l[2] + linb)
            #cv2.line(rgb_image, (lin[0], lin[1]), (lin[2], lin[3]), (142, 107, 255), 3, cv2.LINE_AA)
            #k+=1
            #cv2.putText(rgb_image, str(k), (lin[0], lin[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #linline.append(lin)
            #print(lin)
        pass

while video.isOpened():
    _, img = video.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tresh = cv2.GaussianBlur(gray, (5, 5), 0)
    tresh = cv2.Canny(tresh, 50, 150)
    _, tresh = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
    contours0, hierarchy0 = cv2.findContours(tresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours0:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.intp(box)  # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        if area > 500 and area < 3500:
            be = 0
            j = 0
            k = 0
            for l in linline:
                # j+=1
                # if j != 2 and j != 12 and j != 8 and j != 18 and j != 10 and j != 15 and j != 19 and j != 5 and j != 11:
                k += 1
                cv2.line(img, (l[0], l[1]), (l[2], l[3]), (142, 107, 255), 3, cv2.LINE_AA)
                cv2.putText(img, str(k), (l[0], l[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if box[0][0] > 0:
                # print(box)
                cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
                v = peresech(linline, box)
                if v != -1 and vpr != v:
                    linline[v][4] += 1
                    vpr = v
                    print(box)
                    print("Количество пересечений линии ", v + 1, ":", linline[v][4])

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.imshow('video', img)
    # viewImage(img,'video')
    if cv2.waitKey(40) & 0xFF == ord('q'):
        video.release()
        cv2.destroyAllWindows()
cv2.destroyAllWindows()
video.release()

