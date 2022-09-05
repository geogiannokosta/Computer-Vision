import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionConf=0.5):
        self.minDetectionConf = minDetectionConf

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionConf)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(results.detections)
        bBoxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bBoxC = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bBox = int(bBoxC.xmin*w), int(bBoxC.ymin*h), int(bBoxC.width*w), int(bBoxC.height*h)
                bBoxs.append([id, bBox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bBox)
                    # we draw on our own, without using the default draw method (keep clean rectangle)
                    # cv2.rectangle(img, bBox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bBox[0], bBox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img, bBoxs

    def fancyDraw(self, img, bBox, len = 30, thic = 5, recThic = 1):
        x, y, w, h = bBox
        x1, y1 = x + w, y + h # bottom right corner spot

        cv2.rectangle(img, bBox, (255, 0, 255), recThic)

        # Top Left x,y
        cv2.line(img, (x, y), (x + len, y), (255, 0, 255), thic)
        cv2.line(img, (x, y), (x, y + len), (255, 0, 255), thic)

        # Top Right x1,y
        cv2.line(img, (x1, y), (x1 - len, y), (255, 0, 255), thic)
        cv2.line(img, (x1, y), (x1, y + len), (255, 0, 255), thic)

        # Bottom Left x,y1
        cv2.line(img, (x, y1), (x + len, y1), (255, 0, 255), thic)
        cv2.line(img, (x, y1), (x, y1 - len), (255, 0, 255), thic)

        # Bottom Right x1,y1
        cv2.line(img, (x1, y1), (x1 - len, y1), (255, 0, 255), thic)
        cv2.line(img, (x1, y1), (x1, y1 - len), (255, 0, 255), thic)

        return img

def main():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('Videos/1.mp4')
    pTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bBoxs = detector.findFaces(img)
        # print(bBoxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()