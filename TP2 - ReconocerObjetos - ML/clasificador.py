import cv2
import numpy as np
#from joblib import load
#import sklearn as sk

GREEN = (0,255,0)

"""
----TAGS----
1. Cuadrado
2. Triangulo
3. Circulo

"""
# load model
#clasificator = load('shapes_model.joblib') 


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
references = {1: 'Cuadrado',
              2: 'Circulo',
              3: 'Triangulo'}

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

def process_img(img, threshold1, match_dist, min_area):
    #La convierto a gris
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #La convierto a binaria
    _, img_binary = cv2.threshold(img_gray, threshold1,255,cv2.THRESH_BINARY_INV)

    # Operación morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #kernel = np.ones((5, 5), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

    #Hallo ontornos
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Valido los contornos respecto al area
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            valid_contours.append(c)

    #Dibujo contornos
    cv2.drawContours(img, valid_contours, -1, GREEN, 2)

    #Para cada contorno detectado
    for c in valid_contours:
        #Obtener Hu Moments
        moments = cv2.moments(c)
        hu_moments = cv2.HuMoments(moments)
        
        #Predecir
        #predicted_tag = clasificator.predict(hu_moments.flatten().reshape(1, -1))
        #label = references[predicted_tag[0]]
        label = "Cuadrado"

        #Escribir prediccion
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
    return img, img_binary

while True:
    #Obtener parametros
    threshold1 = cv2.getTrackbarPos("Umbral","Parametros")
    match_dist = cv2.getTrackbarPos("Distancia","Parametros") / 100.0
    min_area = cv2.getTrackbarPos("Area","Parametros")

    #Obtengo imagen de webcam
    success, img = cam.read()

    #Proceso el frame
    result, img_binary = process_img(img.copy(),threshold1,match_dist,min_area)

    #Muestro las ventanas juntas
    stack = stackImages(0.8,([img,img_binary,result]))
    cv2.imshow("Resultado",stack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break