import cv2
import numpy as np
#import matplotlib as plt


frameWidth = 640
frameHeight = 480
cam = cv2.VideoCapture(0)
cam.set(3, frameWidth)
cam.set(4, frameHeight)

def empty(a):
    pass

## Parameters window:
cv2.namedWindow("Parametros")
cv2.resizeWindow("Parametros", 640,240)
cv2.createTrackbar("Umbral","Parametros",93,255,empty)
cv2.createTrackbar("Distancia","Parametros",20,100,empty)
cv2.createTrackbar("Area","Parametros",500,10000,empty)

#Refernce dictionary
references = {}

## Get reference images
square = cv2.imread("cuadrado.jpg")
gray_square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
binary_square = cv2.threshold(gray_square,50,255,cv2.THRESH_BINARY_INV)[1]
sq_contours, sq_hierachy = cv2.findContours(binary_square, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
references["square"] = sq_contours[0]

triangle = cv2.imread("triangulo.png")
gray_triangle = cv2.cvtColor(triangle, cv2.COLOR_BGR2GRAY)
binary_triangle = cv2.threshold(gray_triangle,50,255,cv2.THRESH_BINARY_INV)[1]
tr_contours, tr_hierachy = cv2.findContours(binary_triangle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
references["triangle"] = tr_contours[0]

circle = cv2.imread("circulo.jpg")
gray_circle = cv2.cvtColor(circle, cv2.COLOR_BGR2GRAY)
binary_circle = cv2.threshold(gray_circle,50,255,cv2.THRESH_BINARY_INV)[1]
cir_contours, cir_hierachy = cv2.findContours(binary_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
references["circle"] = cir_contours[0]

##Funcion para apilar las ventanas:
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def get_contours(img,img_contour,param_area):

    contours, hierachy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    

    for i, cont in enumerate(contours):
        #Para que NO marque el primer contorno (toda la imagen)
        if i == 0:
            continue

        #Reduzco el ruido mediante el area (elimino contornos pequeños)
        area = cv2.contourArea(cont)
        print(f'Contorno {i} con area {area}. Limite en {param_area}')
        if area >= param_area:
            perimeter = cv2.arcLength(cont, True)
            aprox = cv2.approxPolyDP(cont, 0.02*perimeter, True)
            cv2.drawContours(img_contour, contours,-1,(0,255,0),7)

            x,y,w,h = cv2.boundingRect(aprox)
            x_mid = int(x + w/3)
            y_mid = int(y + h/1.5)

            coords = (x_mid,y_mid)
            colour = (0,0,0)

def process_img(img, threshold1, match_dist, min_area):
    #La convierto a gris
    img_blur = cv2.GaussianBlur(img, (7,7), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    #La convierto a binaria
    _, img_binary = cv2.threshold(img_gray, threshold1,255,cv2.THRESH_BINARY)

    # Operación morfológica
    kernel = np.ones((5, 5), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

    #Contornos
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Para cada contorno detectado
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        best_match = None
        best_dist = float("inf")

        for name, cont in references.items():
            dist = cv2.matchShapes(c, cont, cv2.CONTOURS_MATCH_I1, 0.0)
            if dist < best_dist:
                best_dist = dist
                best_match = name

        color = (0, 0, 255)  # rojo por defecto (desconocido)
        label = "Desconocido"

        if best_dist < match_dist:
            color = (0, 255, 0)  # verde si es conocido
            label = best_match

        cv2.drawContours(img, [c], -1, color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, img_binary

while True:
    #Obtener parametros
    threshold1 = cv2.getTrackbarPos("Umbral","Parametros")
    match_dist = cv2.getTrackbarPos("Distancia","Parametros") / 100.0
    min_area = cv2.getTrackbarPos("Area","Parametros")

    #Obtengo imagen de webcam
    success, img = cam.read()
    img_contour = img.copy()

    
    #Proceso el frame
    result, img_binary = process_img(img.copy(),threshold1,match_dist,min_area)


    #img_canny = cv2.Canny(img_gray,threshold1,threshold2)



    #get_contours(img_dil,img_contour,area)
    stack = stackImages(0.8,([img,img_binary,result]))
    cv2.imshow("Resultado",stack)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break