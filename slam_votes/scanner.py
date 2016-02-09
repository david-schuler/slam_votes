from __future__ import division
from __future__ import print_function
import cv2
import numpy
import os.path
import sys
import math

DEBUG = False
VISUAL_DEBUG = True 


def displayLine(img, line):
    x1, y1, x2, y2 = line
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

 
def getGradient(line):
    x1, y1, x2, y2 = line
    m = 0
    if x1 != x2:
        m = (y2 - y1) / (x2 - x1) 
    return m

def getVotesFromImage(imageName):
    # load the image and convert it to grayscale
    if not os.path.exists(imageName):
        raise RuntimeError("File not found: {}".format(imageName))
    img = cv2.imread(imageName, cv2.CV_8UC1)

    if VISUAL_DEBUG:
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


    # blur image
    # http://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html#gsc.tab=0
    # getGaussianKernel
    # blur = cv2.GaussianBlur(img, (5, 5), 10)

    # get black white image
    working_img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 41, 20)
    #invert image
    working_img = (255 - working_img)



    # detect lines in image 
    # edges = cv2.Canny(working_img, 50, 200, apertureSize = 3)
    # edges = cv2.Canny(color_img, 50, 200, apertureSize = 3)
    edges = cv2.Canny(img, 50, 100, apertureSize = 3)
    lines = cv2.HoughLinesP(edges, 1, numpy.pi/180, 100,  minLineLength = 50, maxLineGap=6)
    if DEBUG and (lines is not None):
        print("Deteted {} lines in image", format(len(lines)))
    x_min = sys.maxsize
    x_max = 0
    y_min = sys.maxsize 
    y_max = 0
    x_minline = None
    x_maxline = None
    y_minline = None
    y_maxline = None
    for i in lines:
        count = 0
        for x1,y1,x2,y2 in i:
            #TODO if isVertical(line)
            if x1 > x_max or x2 > x_max:
                x_max = max(x1, x2)
                x_maxline = i[0]
            if x1 < x_min or x2 < x_min:
                x_min = min(x1, x2)
                x_minline = i[0]

            if y1 > y_max or y2 > y_max:
                y_max = max(y1, y2)
                y_maxline = i[0]
            if y1 < y_min or y2 < y_min:
                y_min = min(y1, y2)
                y_minline = i[0]
            # cv2.line(display_img,(x1,y1),(x2,y2),(0,0,255),2)

    if VISUAL_DEBUG:
        displayLine(display_img, x_minline)
        displayLine(display_img, x_maxline)
        displayLine(display_img, y_minline)
        displayLine(display_img, y_maxline)

    m_vert_1 = getGradient(x_minline)
    m_vert_2 = getGradient(x_maxline)

    m_hor_1 = getGradient(y_minline)
    m_hor_2 = getGradient(y_maxline)

    shiftx = 0 - x_minline[0]
    shifty = 0 - y_minline[1]
    M = numpy.float32([[1,0, shiftx],[0,1, shifty]])
    rows,cols = img.shape
    dst = cv2.warpAffine(img, M, (cols,rows))
    working_img = cv2.warpAffine(working_img, M, (cols,rows))
    display_img = cv2.warpAffine(display_img, M, (cols,rows))

    x_range = x_maxline[0] - x_minline[0]
    y_range = y_maxline[1] - y_minline[1]

    print("vertical gradient 1: {} angle: {} ".format(m_vert_1, math.atan(m_vert_1)))
    print("vertical gradient 2: {} angle: {} ".format(m_vert_2, math.atan(m_vert_2)))
    print("horizontal gradient 1: {} angle: {} ".format(m_hor_1, math.atan(m_hor_1)))
    print("horizontal gradient 2: {} angle: {} ".format(m_hor_2, math.atan(m_hor_2)))
    # TODO rotation


    x_offset = int(x_range * 0.5)
    y_offset = int(y_range * 0.235)
    xdiff = int((x_range / 1141) * 60)
    ydiff = int((y_range / 1690) * 105)
    # box size
    xsize = int((x_range / 1141) * 50) 
    ysize = int((y_range / 1690) * 50)  
    maxValues = {}
    print("xrange: {} yrange {} ".format(x_range, y_range))
    for row in range(0, 12):
        maxValues[row] = (-1, 0)
        for i in range(0, 9):
            xstart = x_offset + (i * xdiff)
            ystart = y_offset + (row * ydiff)
            xend = xstart + xsize
            yend = ystart + ysize
            # sumatrix has different indexing than drawing rectangles
            ar = working_img[ystart:yend, xstart:xend]
            rectSum = numpy.sum(ar)
            if(rectSum > maxValues[row][0]):
                maxValues[row] = (rectSum, i+1)
            if DEBUG:
                print(" {}  {}".format(i+1, rectSum))
            if VISUAL_DEBUG:
                cv2.rectangle(working_img, (xstart, ystart),
                              (xend, yend), (255, 255, 255), 1)
                cv2.rectangle(display_img, (xstart, ystart),
                              (xend, yend), (0, 255, 0), 1)
    votes = [v[1] for v in maxValues.values()]

    if DEBUG:
        print(maxValues)
        print(votes)
        for key, value in maxValues.iteritems():
            print("{} : {}".format(key, value[1]))

    if VISUAL_DEBUG:
        # resize image
        # the ratio of the new image to the old image
        newWith = 500
        r = newWith / img.shape[1]
        dim = (newWith, int(img.shape[0] * r))
        # perform the actual resizing of the image and show it
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resizedResult = cv2.resize(
                working_img, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("Original", resized)
        cv2.imshow("Result", resizedResult)
        cv2.waitKey(3000)

    return votes


if __name__ == "__main__":
    VISUAL_DEBUG = True
    DEBUG = True
    print(getVotesFromImage("../testimages/card__859724459716.jpg"))

