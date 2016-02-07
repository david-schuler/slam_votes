from __future__ import division
from __future__ import print_function
import cv2
import numpy


DEBUG = False
VISUAL_DEBUG = False 


def getVotesFromImage(imageName):
    # load the image and convert it to grayscale
    #TODO check if file exists
    img = cv2.imread(imageName, cv2.CV_8UC1)

    # blur image
    # http://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html#gsc.tab=0
    # getGaussianKernel
    blur = cv2.GaussianBlur(img, (5, 5), 10)
    threshold = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 41, 20)
    threshold = (255 - threshold)
    xdiff = 60
    ydiff = 105
    maxValues = {}
    for row in range(0, 12):
        maxValues[row] = (-1, 0)
        for i in range(0, 9):
            xstart = (8 * 100) - 15 + (i * xdiff)
            ystart = (7 * 100) - 25 + (row * ydiff)
            xsize = 50
            ysize = 50
            xend = xstart + xsize
            yend = ystart + ysize
            # sumatrix has different indexing than drawing rectangles
            ar = threshold[ystart:yend, xstart:xend]
            rectSum = numpy.sum(ar)
            if(rectSum > maxValues[row][0]):
                maxValues[row] = (rectSum, i+1)
            if DEBUG:
                print(" {}  {}".format(i+1, rectSum))
            if VISUAL_DEBUG:
                cv2.rectangle(threshold, (xstart, ystart),
                              (xend, yend), (255, 255, 255), 1)

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
                threshold, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("Original", resized)
        cv2.imshow("Result", resizedResult)
        cv2.waitKey(3000)

    return votes


if __name__ == "__main__":
    print(getVotesFromImage("test.jpg"))

