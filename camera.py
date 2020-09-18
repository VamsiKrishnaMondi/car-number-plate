import cv2
import pytesseract
import os
from datetime import datetime
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        _,image= self.video.read()
        pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"
        plateCascade = cv2.CascadeClassifier("haarcascade_licence_plate_rus_16stages.xml")
        minArea = 500

        def mark(name):
            with open('vamsi.csv', 'r+')as f:
                mydata = f.readlines()
                namelist = []
                for line in mydata:
                    entry = line.split(',')
                    namelist.append(entry[0])
                if name not in namelist:
                    now = datetime.now()
                    dtstr = now.strftime('%H:%M:%S')
                f.writelines(f'\n{dtstr},{name}')
        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)
        for (x, y, w, h) in numberPlates:
            area = w*h
            if area > minArea:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image, "Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                imgRoi = image[y:y + h,x:x + w]
                cv2.imshow("ROI", imgRoi)
                text = pytesseract.image_to_string(imgRoi, lang='eng')
                mark(text)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    cv2.imwrite("Cr" + ".jpg", imgRoi)
                    cv2.rectangle(image, (0, 100), (300, 300), (0, 255, 0), cv2.FILLED)
                    cv2.putText(image, "Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                    cv2.imshow("Result", image)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()