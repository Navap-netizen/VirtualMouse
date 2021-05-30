import cv2
import numpy as np
import time
import autopy
import handTrackingmodule2 as htm

wCam, hCam = 640, 480
frameR = 70  # frame reduction
smoothening = 7
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
d = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)
# cTime = 0

while True:
    # 1.Find hand Landmarks
    success, img = cap.read()
    img = d.findHands(img)
    lmList, bbox = d.findPosition(img)
    # 2.Get the tip of the index & middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # 3.Check which fingers are up
        fingers = d.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        # 4.Only index finger up:Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5.Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6.Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # 7.Move mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        # 8.Both Index & middle fingers up: clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9.Find distance between fingers
            length, img, lineInfo = d.findDistance(8, 12, img)
            # print(length)
            # 10.Click the mouse if the distance is short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    cv2.imshow("Image", img)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
