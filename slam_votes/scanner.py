from __future__ import division
from __future__ import print_function

import cv2
import numpy
import os.path
import sys
import math
#import configtest

#cfg = configtest.getConfig("NONE")

# Debug switches

DEBUG = False 
VISUAL_DEBUG = False
VISUAL_DEBUG_SHOW_NONBLURED_AND_THRESHOLDED_IMG = False
VISUAL_DEBUG_SHOW_ALL_LINES = False
VISUAL_DEBUG_SHOW_BOUNDING_EDGES = False
VISUAL_DEBUG_SHOW_VOTE_IMG = False
VISUAL_DEBUG_IMAGE_WIDTH = 500 
VISUAL_DEBUG_WAIT_FOR_KEYPRESS = False
USE_ADAPTIVE_THRESHOLDING = False

THRESHOLD_BOUNDARY_EDGES = 0
THRESHOLD_VOTES = 0

def setCameraDebugValues():
    #TODO configuration object
    global     DEBUG 
    DEBUG = True 
    global     VISUAL_DEBUG 
    VISUAL_DEBUG = True
    global     VISUAL_DEBUG_SHOW_NONBLURED_AND_THRESHOLDED_IMG 
    VISUAL_DEBUG_SHOW_NONBLURED_AND_THRESHOLDED_IMG = True
    global     VISUAL_DEBUG_SHOW_ALL_LINES 
    VISUAL_DEBUG_SHOW_ALL_LINES = True
    global     VISUAL_DEBUG_SHOW_BOUNDING_EDGES 
    VISUAL_DEBUG_SHOW_BOUNDING_EDGES = False
    global     VISUAL_DEBUG_SHOW_VOTE_IMG 
    VISUAL_DEBUG_SHOW_VOTE_IMG = False
    global     VISUAL_DEBUG_IMAGE_WIDTH 
    VISUAL_DEBUG_IMAGE_WIDTH = 500 
    global     USE_ADAPTIVE_THRESHOLDING 
    USE_ADAPTIVE_THRESHOLDING = False
    global     THRESHOLD_BOUNDARY_EDGES 
    THRESHOLD_BOUNDARY_EDGES = 0
    global     THRESHOLD_VOTES 
    THRESHOLD_VOTES = 0
    global     VISUAL_DEBUG_WAIT_FOR_KEYPRESS 
    VISUAL_DEBUG_WAIT_FOR_KEYPRESS = False

# def initGlobals(cfg):
#     global DEBUG
#     DEBUG = cfg.getBoolean('show_messages')
#     global VISUAL_DEBUG
#     VISUAL_DEBUG = cfg.getBoolean('show_images')
#     global VISUAL_DEBUG_SHOW_NONBLURED_AND_THRESHOLDED_IMG
#     VISUAL_DEBUG_SHOW_NONBLURED_AND_THRESHOLDED_IMG = cfg.getBoolean('show_bounding_edges') 
#     global VISUAL_DEBUG_SHOW_ALL_LINES
#     VISUAL_DEBUG_SHOW_ALL_LINES = cfg.getBoolean('show_all_lines')
#     global VISUAL_DEBUG_SHOW_BOUNDING_EDGES
#     VISUAL_DEBUG_SHOW_BOUNDING_EDGES = cfg.getBoolean('show_bounding_edges')
#     global VISUAL_DEBUG_SHOW_VOTE_IMG
#     VISUAL_DEBUG_SHOW_VOTE_IMG = cfg.getBoolean('show_vote_img')
#     global VISUAL_DEBUG_IMAGE_WIDTH
#     VISUAL_DEBUG_IMAGE_WIDTH = cfg.getInt('image_width')
# 
#     global USE_ADAPTIVE_THRESHOLDING
#     USE_ADAPTIVE_THRESHOLDING = cfg.getBoolean('use_adaptive_thresholding')
# 
#     global THRESHOLD_BOUNDARY_EDGES
#     THRESHOLD_BOUNDARY_EDGES = cfg.get('threshold_boundary_edges')
#     global THRESHOLD_VOTES
#     THRESHOLD_VOTES = cfg.get('threshold_votes')


# draws a line with given coordinate and color in an image
def displayLine(img, line, color=(0, 0, 255)):
    x1, y1, x2, y2 = line
    cv2.line(img, (x1, y1), (x2, y2), color, 2)


def getGradient(line):
    x1, y1, x2, y2 = line
    m = 0
    if x1 != x2:
        m = (y2 - y1) / (x2 - x1)
    else:
        m = sys.maxsize 
    return m


def resizeAndDisplay(img, name, wait, translate=0):
    # resize image
    new_with = VISUAL_DEBUG_IMAGE_WIDTH
    # the ratio of the new image to the old image
    r = new_with / img.shape[1]
    dim = (new_with, int(img.shape[0] * r))
    print("New dim" , dim)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(name, resized)
    if translate != 0:
        cv2.moveWindow(name, translate, 0)
    if VISUAL_DEBUG_WAIT_FOR_KEYPRESS:
        cv2.waitKey(wait)

# maximum angle of pi/4 (equals gradient=1) is considered as vertical
def isVertical(line):
    grad = getGradient(line)
    return abs(grad) > 1


def isHorizontal(line):
    return not isVertical(line)


def getVotesFromImage(imageName):
    if not os.path.exists(imageName):
        raise RuntimeError("File not found: {}".format(imageName))

    # load the image and convert it to grayscale
    img = cv2.imread(imageName, cv2.CV_8UC1)

    if VISUAL_DEBUG:
        display_img = cv2.imread(imageName, cv2.IMREAD_COLOR)
    return getVotesFromCV2Img(img)

def getVotesFromCV2Img(img):

    if VISUAL_DEBUG:
        display_img = img.copy()

    # blur image
    working_img = cv2.GaussianBlur(img, (3, 3), 0)
    # get black white image
    threshold_method = cv2.THRESH_BINARY
    if THRESHOLD_BOUNDARY_EDGES == 0:
        threshold_method = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    tmp, working_img = cv2.threshold(
            working_img, THRESHOLD_BOUNDARY_EDGES, 255,
            threshold_method)

    if VISUAL_DEBUG and VISUAL_DEBUG_SHOW_NONBLURED_AND_THRESHOLDED_IMG:
        resizeAndDisplay(
                working_img, "Threshshold and binary", 1,
                VISUAL_DEBUG_IMAGE_WIDTH)
        resizeAndDisplay(
                display_img, "Comparison", 0,
                VISUAL_DEBUG_IMAGE_WIDTH *2)

    # invert image
    working_img = (255 - working_img)

    # detect lines in image
    edges = cv2.Canny(working_img, 100, 200, apertureSize=3)
    if DEBUG:
        print("Edges: {}".format(len(edges)))
    lines = cv2.HoughLinesP(
            edges, 1, numpy.pi/180, 40,
            minLineLength=10, maxLineGap=6)
    if DEBUG and (lines is not None):
        print("Deteted {} lines in image".format(len(lines)))
    x_min = sys.maxsize
    x_max = 0
    y_min = sys.maxsize
    y_max = 0
    x_minline = None
    x_maxline = None
    y_minline = None
    y_maxline = None
    # Values for test on Mac:
    # (145, 1, 4) (22, 1, 4) (29, 1, 4) (10, 1, 4) (7, 1, 4)
    #for i in lines:
    for i in lines[0]:
        # for some reason i is an array with 1 line as item
        line = i
        #line = i[0]
        x1, y1, x2, y2 = line
        if (x1 > x_max or x2 > x_max) and isVertical(line):
            x_max = max(x1, x2)
            x_maxline = line
        if (x1 < x_min or x2 < x_min) and isVertical(line):
            x_min = min(x1, x2)
            x_minline = line
        if (y1 > y_max or y2 > y_max) and isHorizontal(line):
            y_max = max(y1, y2)
            y_maxline = line
        if (y1 < y_min or y2 < y_min) and isHorizontal(line):
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
        resizeAndDisplay(working_img, "Bounding edges in working_img", 1)

#    # TODO rotation
#    # get gradients for the bounding box
#    m_vert_1 = getGradient(x_minline)
#    m_vert_2 = getGradient(x_maxline)
#    m_hor_1 = getGradient(y_minline)
#    m_hor_2 = getGradient(y_maxline)
#    print("vertical gradient 1: {} angle: {} ".format(
#        m_vert_1, math.atan(m_vert_1)/ math.pi))
#    print("vertical gradient 2: {} angle: {} ".format(
#        m_vert_2, math.atan(m_vert_2)/ math.pi))
#    print("horizontal gradient 1: {} angle: {} ".format(
#        m_hor_1, math.atan(m_hor_1/ math.pi)))
#    print("horizontal gradient 2: {} angle: {} ".format(
#        m_hor_2, math.atan(m_hor_2)/ math.pi))

    # translate images
    min_x = int(round((x_minline[0] + x_minline[2]) / 2))
    min_y = int(round((y_minline[1] + y_minline[3]) / 2))
    M = numpy.float32([[1, 0, -1 * min_x], [0, 1, -1 * min_y]])
    rows, cols = img.shape
    dst = cv2.warpAffine(img, M, (cols, rows))
    working_img = cv2.warpAffine(working_img, M, (cols, rows))
    if VISUAL_DEBUG:
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

    # adaptive thresholding can be activated for testing puproses
    if USE_ADAPTIVE_THRESHOLDING:
        # calculate window size relative to size of the vote box
        window_size = int(xsize / 10) * 2 + 1
        window_size = max(window_size, 3)
        non_blured_bw_img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, window_size, 5)
    else:
        # get non blured bw image
        threshold_method = cv2.THRESH_BINARY_INV 
        if THRESHOLD_VOTES == 0:
            threshold_method = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        tmp, non_blured_bw_img = cv2.threshold(
                img, THRESHOLD_VOTES, 255,
                threshold_method)
    # translate this image
    non_blured_bw_img = cv2.warpAffine(non_blured_bw_img, M, (cols, rows))
    if VISUAL_DEBUG and VISUAL_DEBUG_SHOW_VOTE_IMG:
        resizeAndDisplay(non_blured_bw_img, "Nonblured bw Votes", 1,
                         VISUAL_DEBUG_IMAGE_WIDTH)
    # make this image the new working image,
    # the old one is only needed for the boundaries.
    working_img = non_blured_bw_img
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

    if VISUAL_DEBUG and VISUAL_DEBUG_SHOW_VOTE_IMG:
        resizeAndDisplay(working_img, "Vote with boxes", 100,
                VISUAL_DEBUG_IMAGE_WIDTH *2)

    if DEBUG:
        print(maxValues)
        print(votes)
        for key, value in maxValues.iteritems():
            print("{} : {}".format(key, value[1]))

    if VISUAL_DEBUG:
        resizeAndDisplay(display_img, "Display Image", 3000)
    if DEBUG:
        print("Expected 1: {} ".format(votes == [2,6,5,6,5,9,7,5,9,7,3,5]))
        print("Expected 2: {} ".format(votes == [8,5,9,7,2,4,4,5,9,7,1,6]))
        print("Expected 3: {} ".format(votes == [8,9,8,6,5,6,7,1,7,5,7,5]))

    return votes


if __name__ == "__main__":
    VISUAL_DEBUG = True
    DEBUG = True
    imageName = "../testimages/card__859724459716.jpg"
    #  imageName = "../testimages/card_webcam__898656717575.jpg"
    print(getVotesFromImage(imageName))
    print(imageName)
