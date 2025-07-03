import cv2
import mediapipe as mp
import numpy as np
import math
from pynput.keyboard import Controller, Key
from time import sleep
import ctypes

import mouse



user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


#Comment when using mouse
# frameWidth = 1280
# frameHeight = 720

#Uncomment to use mouse
frameWidth = 1920
frameHeight = 1080

frame_reduction = 100
smoothening = 7



SHOW_KEYBOARD_BTN_X = 650
SHOW_KEYBOARD_BTN_Y = 5
PAD_X = 700
PAD_Y = 450
PAD_WIDTH = frameWidth-700
PAD_HEIGHT = frameHeight-450

pTime = 0
PLOCX = 0
PLOCY = 0


cam = cv2.VideoCapture(0)
cam.set(3, frameWidth)
cam.set(4, frameHeight)


# screenWidth, screenHeight = autopy.screen.size()

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P","<-"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";","->"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/","(X)"]]
finalText = ""

keyboard = Controller()


def cornerRect(img, bbox, l=30, t=5, rt=1,
               colorR=(0, 0, 0), colorC=(0, 255, 0)):
    """
    :param img: Image to draw on.
    :param bbox: Bounding box [x, y, w, h]
    :param l: length of the corner line
    :param t: thickness of the corner line
    :param rt: thickness of the rectangle
    :param colorR: Color of the Rectangle
    :param colorC: Color of the Corners
    :return:
    """
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    if rt != 0:
        cv2.rectangle(img, bbox, colorR, rt)
    # Top Left  x,y
    cv2.line(img, (x, y), (x + l, y), colorC, t)
    cv2.line(img, (x, y), (x, y + l), colorC, t)
    # Top Right  x1,y
    cv2.line(img, (x1, y), (x1 - l, y), colorC, t)
    cv2.line(img, (x1, y), (x1, y + l), colorC, t)
    # Bottom Left  x,y1
    cv2.line(img, (x, y1), (x + l, y1), colorC, t)
    cv2.line(img, (x, y1), (x, y1 - l), colorC, t)
    # Bottom Right  x1,y1
    cv2.line(img, (x1, y1), (x1 - l, y1), colorC, t)
    cv2.line(img, (x1, y1), (x1, y1 - l), colorC, t)

    return img


def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]),
                      (30, 144, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 20, y + 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
        
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out
        
def drawNoKeyboard(img):
    imgNew = np.zeros_like(img, np.uint8)
    x = SHOW_KEYBOARD_BTN_X
    y = SHOW_KEYBOARD_BTN_Y
    cornerRect(imgNew, (x, y, 65, 65),20, rt=0)
    cv2.rectangle(imgNew, (x,y), (x+65, y+65),(30, 144, 255), cv2.FILLED)
    cv2.putText(imgNew, "(X)", (x + 20, y + 40),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
    
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

def drawPad(img):
    imgNew = np.zeros_like(img, np.uint8)
    x = 0
    y = frameHeight
    cornerRect(imgNew, (PAD_X, PAD_Y, PAD_WIDTH, PAD_HEIGHT),20, rt=0)
    cv2.rectangle(imgNew, (PAD_X,PAD_Y), (PAD_X + 1500, PAD_Y + 1500),(200, 200, 200), cv2.FILLED)


    
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

def checkMouse(index_x_pos,index_y_pos,middle_x_pos,middle_y_pos,plocX,plocY):
        
        x3 = np.interp(middle_x_pos, (100, 640 - 100), (0, screensize[0]))
        y3 = np.interp(middle_y_pos, (100, 480 - 100), (0, screensize[1]))
        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
    
        # 7. Move Mouse
        # autopy.mouse.move(wScr - clocX, clocY)
        mouse.move(screensize[0]-clocX,clocY)

        plocX = clocX
        plocY = clocY

        distance = math.hypot((index_x_pos-middle_x_pos), (index_y_pos-middle_y_pos))
        if(distance < 40):
             mouse.click("left")


        return plocX,plocY


class Key():
    def __init__(self, pos, text, size=[65, 65]):
        self.pos = pos
        self.size = size
        self.text = text


# -------------------> MAIN <----------------------


keyList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        keyList.append(Key([100 * j + 80, 100 * i + 80], key))

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
     static_image_mode=False,
     max_num_hands=2,
     min_detection_confidence=0.5)


show_keyboard = True
while True:
    #Obtengo imagen de webcam
    success, frame = cam.read()
    frame = cv2.flip(frame,1)

    #Draw interface
    if(show_keyboard == True):
        frame = drawAll(frame,keyList)
    else:
        frame = drawNoKeyboard(frame)

    # frame = drawPad(frame)

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


    #Get finger tracking
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            # index_x_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 1280
            # index_y_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 720
            # middle_x_pos = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * 1280
            # middle_y_pos = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * 720

            index_x_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frameWidth
            index_y_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frameHeight
            middle_x_pos = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frameWidth
            middle_y_pos = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frameHeight



            if(show_keyboard == False):
                PLOCX,PLOCY = checkMouse(index_x_pos,index_y_pos,middle_x_pos,middle_y_pos,PLOCX,PLOCY)
                if(SHOW_KEYBOARD_BTN_X < middle_x_pos  and middle_x_pos < SHOW_KEYBOARD_BTN_X+65
                   and SHOW_KEYBOARD_BTN_Y < middle_y_pos  and middle_y_pos < SHOW_KEYBOARD_BTN_Y+65):
                            cv2.rectangle(frame, (SHOW_KEYBOARD_BTN_X - SHOW_KEYBOARD_BTN_Y, SHOW_KEYBOARD_BTN_Y - 5), (SHOW_KEYBOARD_BTN_X + 65 + 5, SHOW_KEYBOARD_BTN_Y + 65 + 5), (30, 144, 255), cv2.FILLED)
                            cv2.putText(frame, "(X)", (SHOW_KEYBOARD_BTN_X, SHOW_KEYBOARD_BTN_Y+40),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4)
                            show_keyboard = not show_keyboard
                            cv2.rectangle(frame, key.pos, (SHOW_KEYBOARD_BTN_X + 65, SHOW_KEYBOARD_BTN_Y + 65), (30, 144, 255), cv2.FILLED)
                            cv2.putText(frame, key.text, (SHOW_KEYBOARD_BTN_X + 20, SHOW_KEYBOARD_BTN_Y + 65),
                                        cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                            sleep(0.50)
            else:
                for key in keyList:
                    x,y = key.pos
                    w,h = key.size


                    distance = math.hypot((middle_x_pos-index_x_pos), (middle_y_pos-index_y_pos))
                    if(x < middle_x_pos  and middle_x_pos < x+w
                    and y < middle_y_pos  and middle_y_pos < y+h):
                            cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (30, 144, 255), cv2.FILLED)
                            if(key.text in ("<-","->","(X)")):
                                cv2.putText(frame, key.text, (x, y+40),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4)
                            else:
                                cv2.putText(frame, key.text, (x + 15, y + 50),cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                            
                            if(distance < 40):
                                if(key.text == "<-"):
                                    keyboard.press(keyboard._Key.backspace)
                                    finalText = finalText[:-1]
                                elif(key.text == "->"):
                                    keyboard.press(keyboard._Key.enter)
                                    finalText = ""
                                elif(key.text == "(X)"):

                                    #Uncomment to use mouse
                                    show_keyboard = not show_keyboard

                                    #Comment when use mouse
                                    # cam.release()
                                    # cv2.destroyAllWindows()
                                else:
                                    keyboard.press(key.text)
                                    finalText += key.text
                                cv2.rectangle(frame, key.pos, (x + w, y + h), (30, 144, 255), cv2.FILLED)
                                cv2.putText(frame, key.text, (x + 20, y + 65),
                                            cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                                sleep(0.50)

                #Show written text
                cv2.rectangle(frame, (50, 350), (700, 450), (30, 144, 255), cv2.FILLED)
                cv2.putText(frame, finalText, (60, 430),
                            cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5)

    cv2.imshow("Resultado",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()