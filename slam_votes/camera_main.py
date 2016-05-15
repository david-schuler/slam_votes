import cv2
import os
from scanner import getVotesFromCV2Img 

def main():

    try:
        cam = cv2.VideoCapture(0)
    except:
        pass
    ret, frame = cam.read()
    while ret:
        ret, frame = cam.read()
        cv2.imshow('Live', frame)
        k = cv2.waitKey(33)
#        print(k)
        if k == 32: #32 on mac #1048608:   #Space
            print('space')

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = getVotesFromCV2Img(img)
            print(result)
        if k == 27: #27 on mac 1048603:   #ESC
            cam.release()
            break


if __name__ == '__main__':
    main()
