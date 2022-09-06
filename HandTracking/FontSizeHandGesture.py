import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

##############################################################################
# PROBLEM WITH COMTypes IN MAC
# CHANGE OF PLANS: Volume change with gesture --> Font size change, of a number on the screen, with gesture

# import pycaw
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
##############################################################################

############################
wCam, hCam = 1280, 720
############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.handDetector(detectionConf=0.7)

##############################################################################
# PROBLEM WITH COMTypes IN MAC
# CHANGE OF PLANS: Volume change with gesture --> Font size change, of a number on the screen, with gesture

# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
# volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
##############################################################################

minFont = 1
maxFont = 70
numberOnScreen = 7
font = 0
fontBar = 300

while True:
    success, img = cap.read()

    # cv2.putText(img, str(numberOnScreen), (120, 400), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 3)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList) # total of 21 values
        # print(lmList[4], lmList[8]) # position of landmark number 4 and 8 (tips of thumb and index finger)

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2 # find the middle point of the line

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # create a line to connect the two tips
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) # draw a circle on the middle of the line

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # Font range 1 - 100
        font = np.interp(length, [30, 300], [minFont, maxFont])
        fontBar = np.interp(length, [30, 300], [300, 50])
        # print(int(length), font)

        cv2.putText(img, str(numberOnScreen), (120, 715), cv2.FONT_HERSHEY_PLAIN, font, (255, 0, 0), 6)

        # Hand range 30 - 300
        if length < 30:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 50), (85, 300), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(fontBar)), (85, 300), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'Font size: {int(font)}', (48, 350), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 0), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (1000, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)