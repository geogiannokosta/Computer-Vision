import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Videos/2.mp4')

pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results.detections)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score) #score concerning whether the element is a face or not
            # print(detection.location_data.relative_bounding_box)

            bBoxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bBox = int(bBoxC.xmin*w), int(bBoxC.ymin*h), int(bBoxC.width*w), int(bBoxC.height*h)
            # we draw on our own, without using the default draw method (keep clean rectangle)
            cv2.rectangle(img, bBox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bBox[0], bBox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1) # reduce fps by increasing the number inside waitKey