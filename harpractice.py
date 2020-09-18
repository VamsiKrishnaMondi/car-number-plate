import cv2
import numpy as np
import os
import DetectChars
import DetectPlates
import PossiblePlate
from datetime import datetime
frameWidth = 640
franeHeight = 480
plateCascade = cv2.CascadeClassifier("vamsi.xml")
minArea = 500
cap =cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,franeHeight)
cap.set(10,150)
count = 0
def mark(name):
    with open('vamsi.csv','r+')as f:
        mydata=f.readlines()
        namelist=[]
        for line in mydata:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dtstr=now.strftime('%H:%M:%S')
        f.writelines(f'\n{dtstr},{name}')
while True:
    success , img  = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img,"Plate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgRoi)
    cv2.imshow("Result",img)
    if cv2.waitKey(1) & 0xFF ==ord('s'):
        cv2.imwrite("Cr"+".jpg",imgRoi)
        cv2.rectangle(img,(0,100),(300,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Saved",(15,265),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        cv2.imshow("Result",img)
        SCALAR_BLACK = (0.0, 0.0, 0.0)
        SCALAR_WHITE = (255.0, 255.0, 255.0)
        SCALAR_YELLOW = (0.0, 255.0, 255.0)
        SCALAR_GREEN = (0.0, 255.0, 0.0)
        SCALAR_RED = (0.0, 0.0, 255.0)

        showSteps = True
        def main():
            blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()
            if blnKNNTrainingSuccessful == False:
                print("\nerror: KNN traning was not successful\n")
                return
            imgOriginalScene = cv2.imread("Cr.jpg")
            if imgOriginalScene is None:
                print("\nerror: image not read from file \n\n")
                os.system("pause")
                return
            listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
            listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)
            cv2.imshow("imgOriginalScene", imgOriginalScene)
            if len(listOfPossiblePlates) == 0:
                print("\nno license plates were detected\n")
            else:
                listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
                licPlate = listOfPossiblePlates[0]

                cv2.imshow("imgPlate", licPlate.imgPlate)
                cv2.imshow("imgThresh", licPlate.imgThresh)

                if len(licPlate.strChars) == 0:
                    print("\nno characters were detected\n\n")
                    return
                drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
                print(
                    "\nlicense plate read from image = " + licPlate.strChars + "\n")
                print("------------")
                mark(licPlate.strChars)
                writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
                cv2.imshow("imgOriginalScene", imgOriginalScene)
                cv2.imwrite("imgOriginalScene.png", imgOriginalScene)
            cv2.waitKey(0)
            return
        def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
            p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
            cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
            cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
            cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
            cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
        def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
            ptCenterOfTextAreaX = 0
            ptCenterOfTextAreaY = 0
            ptLowerLeftTextOriginX = 0
            ptLowerLeftTextOriginY = 0
            sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
            plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape
            intFontFace = cv2.FONT_HERSHEY_SIMPLEX
            fltFontScale = float(plateHeight) / 30.0
            intFontThickness = int(round(fltFontScale * 1.5))
            textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                                 intFontThickness)
            ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
             fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene
            intPlateCenterX = int(intPlateCenterX)
            intPlateCenterY = int(intPlateCenterY)
            ptCenterOfTextAreaX = int(
                intPlateCenterX)
            if intPlateCenterY < (sceneHeight * 0.75):
                ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
                    round(plateHeight * 1.6))
            else:
                ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
                    round(plateHeight * 1.6))
            textSizeWidth, textSizeHeight = textSize

            ptLowerLeftTextOriginX = int(
                ptCenterOfTextAreaX - (textSizeWidth / 2))
            ptLowerLeftTextOriginY = int(
                ptCenterOfTextAreaY + (textSizeHeight / 2))
            cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY),
                        intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)

        if __name__ == "__main__":
            main()
        cv2.waitKey(500)
        count+=1