import cv2
import time
import numpy as np
import FaceMeshModule as fmm
import math

############################
wCam, hCam = 1700, 1000
############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

angelFar = cv2.imread('angel_far.jpg')
angelFar = cv2.resize(angelFar, (wCam, hCam), interpolation = cv2.INTER_AREA)
angelClose = cv2.imread('angel_close.jpg')
menu = cv2.imread('menu.jpg')

detector = fmm.FaceMeshDetector(maxFaces=1)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
initRatio = -1
blinkCounter = 0

while True:
    imgMenu = cv2.resize(menu, (wCam, hCam), interpolation=cv2.INTER_AREA)
    cv2.imshow("Menu", imgMenu)
    cv2.waitKey(0)
    cv2.destroyWindow("Menu")
    break

while True:
    # That's for short videos that we want to play on repeat without closing
    # if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if len(faces)!=0:
        face = faces[0]
        # for id in idList:
        #     cv2.circle(img, (face[id][1], face[id][2]), 3, (255, 0, 255), cv2.FILLED)

        leftUp = [face[159][1], face[159][2]]
        leftDown = [face[23][1], face[23][2]]
        # print(leftUp, leftDown)
        lengthVer,_ = detector.findDistance(leftUp, leftDown)
        # cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        # print(lengthVer)

        leftLeft = [face[130][1], face[130][2]]
        leftRight = [face[243][1], face[243][2]]
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)
        # cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)
        # print(lengthHor)

        ratio = (lengthVer/lengthHor)*100
        # print(ratio)
        if initRatio == -1:
            initRatio = ratio
            print(initRatio)

        img = cv2.resize(img, (wCam, hCam), interpolation=cv2.INTER_AREA)
        cv2.imshow("DontBlink", np.concatenate((cv2.flip(img, 1), angelFar), axis=0))

        if ratio < 0.9*initRatio:
            blinkCounter += 1
            if blinkCounter != 0:
                cv2.destroyWindow("DontBlink")
                cv2.imshow("You are dead!", angelClose)
                cv2.waitKey(0)
                break

        cv2.waitKey(1)
