import cv2
import mediapipe as mp
import numpy as np
import math
from pynput.keyboard import Controller, Key
# import keyboard
from time import sleep



frameWidth = 1280
frameHeight = 720 
cam = cv2.VideoCapture(0)
cam.set(3, frameWidth)
cam.set(4, frameHeight)

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P","DEL"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
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
        cv2.putText(imgNew, button.text, (x + 40, y + 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    # print(mask.shape)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out


class Key():
    def __init__(self, pos, text, size=[65, 65]):
        self.pos = pos
        self.size = size
        self.text = text


# -------------------> MAIN <----------------------
keyboard = Controller()

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

while True:
    #Obtengo imagen de webcam
    success, frame = cam.read()
    frame = cv2.flip(frame,1)

    frame = drawAll(frame,keyList)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)


    if results.multi_hand_landmarks is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            for key in keyList:
                x,y = key.pos
                w,h = key.size

                index_x_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 1280
                index_y_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 720
                middle_x_pos = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * 1280
                middle_y_pos = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * 720

                distance = math.hypot((middle_x_pos-index_x_pos), (middle_y_pos-index_y_pos))

                if(x < index_x_pos  and index_x_pos < x+w
                   and y < index_y_pos  and index_y_pos < y+h):
                    cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (30, 144, 255), cv2.FILLED)
                    cv2.putText(frame, key.text, (x + 15, y + 50),cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                    if(distance < 30):
                        if(key.text == "DEL"):
                            keyboard.press(keyboard._Key.backspace)
                        else:
                            keyboard.press(key.text)
                        cv2.rectangle(frame, key.pos, (x + w, y + h), (30, 144, 255), cv2.FILLED)
                        cv2.putText(frame, key.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                        finalText += key.text
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