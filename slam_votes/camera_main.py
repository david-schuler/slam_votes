import sys
import cv2
import os
from scanner import getVotesFromCV2Img, setCameraDebugValues
import operator
import datetime

def main():
    totalResult = {}
    voteCount = 0

    now = datetime.datetime.now()
    timeStr = now.strftime("%y_%m_%d-%H_%M_%S")
    filename = "votes_" + timeStr + ".txt"
    f = open(filename,'w')
    for i in range(1,13):
        totalResult[i] = 0
    #=[0] *12
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
            print('___TAKING PICTURE___')

            #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            result = getVotesFromCV2Img(frame)
            voteCount += 1
            processResult(result, totalResult, voteCount, f)
        if k == 27: #27 on mac 1048603:   #ESC
            cam.release()
            break
    f.close()


def processResult(result, totalResult, voteCount, f):
    print(result)
    for c, r in enumerate(result):
        totalResult[c+1] =  totalResult[c+1] + r
    f.write(' '.join(str(x) for x in result) + '\n')
    sorted_x = sorted(totalResult.items(), key=operator.itemgetter(1))
    print("Vote Count: {} ".format(voteCount))
    for c, r in sorted_x:
        print("{:>3}  : {:>4} ".format(c,r))

if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) >= 2:
        print(sys.argv[1])
        if sys.argv[1] == 'debug':
            setCameraDebugValues()
    main()
