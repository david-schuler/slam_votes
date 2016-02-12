from __future__ import division
from __future__ import print_function
import cv2
import numpy
import os.path
import sys
import math

# Debug switches
DEBUG = False
VISUAL_DEBUG = True
VISUAL_DEBUG_SHOW_BLURED_AND_THRESHOLDED_IMG = False
VISUAL_DEBUG_SHOW_ALL_LINES = False
VISUAL_DEBUG_SHOW_BOUNDING_EDGES = False


# draws a line with given coordinate and color in an image
def displayLine(img, line, color=(0, 0, 255)):
    x1, y1, x2, y2 = line
    cv2.line(img, (x1, y1), (x2, y2), color, 2)


def getGradient(line):
    x1, y1, x2, y2 = line
    m = 0
    if x1 != x2:
        m = (y2 - y1) / (x2 - x1)
    return m


def resizeAndDisplay(img, name, wait):
    # resize image
    new_with = 500
    # the ratio of the new image to the old image
    r = new_with / img.shape[1]
    dim = (new_with, int(img.shape[0] * r))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(name, resized)
    cv2.waitKey(wait)


def getVotesFromImage(imageName):
    # load the image and convert it to grayscale
    if not os.path.exists(imageName):
        raise RuntimeError("File not found: {}".format(imageName))
    img = cv2.imread(imageName, cv2.CV_8UC1)

    if VISUAL_DEBUG:
        display_img = cv2.imread(imageName, cv2.IMREAD_COLOR)

    # blur image
    working_img = cv2.GaussianBlur(img, (15, 15), 0)
    # get black white image
    tmp, working_img = cv2.threshold(
            working_img, 100, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if VISUAL_DEBUG and VISUAL_DEBUG_SHOW_BLURED_AND_THRESHOLDED_IMG:
        resizeAndDisplay(working_img, "Thresh", 1)
        resizeAndDisplay(display_img, "comp", 0)

    # invert image
    working_img = (255 - working_img)
    cv2.imshow('kacka', working_img)
    cv2.waitKey()



    # detect lines in image
    edges = cv2.Canny(working_img, 100, 200, apertureSize=3)
    lines = cv2.HoughLinesP(
            edges, 1, numpy.pi/180, 40,
            minLineLength=10, maxLineGap=6)
    if DEBUG and (lines is not None):
        print("Deteted {} lines in image".format(lines.shape[1]))
    x_min = sys.maxsize
    x_max = 0
    y_min = sys.maxsize
    y_max = 0
    x_minline = None
    x_maxline = None
    y_minline = None
    y_maxline = None
    for i in lines[0]:
        line = i
        x1, y1, x2, y2 = line
        # TODO if isVertical(line)
        if x1 > x_max or x2 > x_max:
            x_max = max(x1, x2)
            x_maxline = line
        if x1 < x_min or x2 < x_min:
            x_min = min(x1, x2)
            x_minline = line
        if y1 > y_max or y2 > y_max:
            y_max = max(y1, y2)
            y_maxline = line
        if y1 < y_min or y2 < y_min:
            y_min = min(y1, y2)
            y_minline = line
        if VISUAL_DEBUG and VISUAL_DEBUG_SHOW_ALL_LINES:
            displayLine(display_img, line, (255, 0, 0))

    if VISUAL_DEBUG:
        displayLine(display_img, x_minline)
        displayLine(display_img, x_maxline)
        displayLine(display_img, y_minline)
        displayLine(display_img, y_maxline)

    if VISUAL_DEBUG and VISUAL_DEBUG_SHOW_BOUNDING_EDGES:
        resizeAndDisplay(display_img, "Bounding edges", 1)
        displayLine(working_img, x_minline)
        displayLine(working_img, x_maxline)
        displayLine(working_img, y_minline)
        displayLine(working_img, y_maxline)
        resizeAndDisplay(working_img, "Bounding edges in working_img", 0)

#    # TODO rotation
#    # get gradients for the bounding box
#    m_vert_1 = getGradient(x_minline)
#    m_vert_2 = getGradient(x_maxline)
#    m_hor_1 = getGradient(y_minline)
#    m_hor_2 = getGradient(y_maxline)
#    print("vertical gradient 1: {} angle: {} ".format(
#        m_vert_1, math.atan(m_vert_1)))
#    print("vertical gradient 2: {} angle: {} ".format(
#        m_vert_2, math.atan(m_vert_2)))
#    print("horizontal gradient 1: {} angle: {} ".format(
#        m_hor_1, math.atan(m_hor_1)))
#    print("horizontal gradient 2: {} angle: {} ".format(
#        m_hor_2, math.atan(m_hor_2)))
#
    # translate images

    min_x = int(round((x_minline[0] + x_minline[2]) / 2))
    min_y = int(round((y_minline[1] + y_minline[3]) / 2))
    M = numpy.float32([[1, 0, -1 * min_x], [0, 1, -1 * min_y]])
    rows, cols = img.shape
    dst = cv2.warpAffine(img, M, (cols, rows))
    working_img = cv2.warpAffine(working_img, M, (cols, rows))
    display_img = cv2.warpAffine(display_img, M, (cols, rows))


    max_x = int(round((x_maxline[0] + x_maxline[2]) / 2))
    max_y = int(round((y_maxline[1] + y_maxline[3]) / 2))
    x_range = max_x - min_x
    y_range = max_y - min_y

    # offset for first (first column in first row) box
    x_offset = x_range * 0.505
    y_offset = y_range * 0.235
    # box size
    xsize = x_range * 0.05
    ysize = y_range * 0.04
    # distance between to boxes
    xdiff = xsize
    ydiff = y_range * 0.0625

    maxValues = {}
    if DEBUG:
        print("xrange: {} yrange {} ".format(x_range, y_range))

    # get non blured bw image
    tmp, non_blured_bw_img = cv2.threshold(
            img, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # translate this image
    non_blured_bw_img = cv2.warpAffine(non_blured_bw_img, M, (cols, rows))

    for row in range(0, 12):
        maxValues[row] = (-1, 0)
        for i in range(0, 9):
            xstart_f = x_offset + (i * xdiff)
            xstart = int(round(xstart_f))
            ystart_f = y_offset + (row * ydiff)
            ystart = int(round(ystart_f))
            xend = int(round(xstart_f + xsize))
            yend = int(round(ystart_f + ysize))

            # sumatrix has different indexing than drawing rectangles
            ar = non_blured_bw_img[ystart:yend, xstart:xend]
            rectSum = numpy.sum(ar)
            if(rectSum > maxValues[row][0]):
                maxValues[row] = (rectSum, i+1)
            if DEBUG:
                print("Sum for box {} in row {}: {}".format(i+1, row, rectSum))
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
        resizeAndDisplay(working_img, "Result", 10)
        resizeAndDisplay(display_img, "Display Image", 3000)
    return votes


if __name__ == "__main__":
    VISUAL_DEBUG = True
    DEBUG = True
    imageName = "../testimages/card_webcam__898656717575.jpg"
    #  imageName = "../testimages/card_webcam__898656717575.jpg"
    print(getVotesFromImage(imageName))
    print(imageName)
