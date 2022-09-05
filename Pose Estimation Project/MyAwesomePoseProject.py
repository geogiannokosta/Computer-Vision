import cv2
import time
import PoseModule as pm

# cap = cv2.VideoCapture('PoseVideos/4.mp4')
cap = cv2.VideoCapture(0)
pTime = 0

detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)

    # check if the list has at least one value
    # if len(lmList) != 0:
    #     print(lmList)

    # Get location only of id=14, draw bigger dot on id=14
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 25, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)