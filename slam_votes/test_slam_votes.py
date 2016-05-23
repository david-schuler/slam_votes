import unittest
import os
import cv2
import scanner
from scanner import getVotesFromImage



class TestSlamVotes(unittest.TestCase):
 
    def setUp(self):
        self.longMessage = True
        pass


    def check_all_images(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") and "__" in filename: 
                self.checkImage(directory, filename)



    def checkImage(self, directory, filename):
        print("Testing image {} ".format(filename))
        name = os.path.splitext(filename)[0]
        expectedStr = name.split("__")[1]
        expectedList = [int(x) for x in expectedStr]
        fullFilename = os.path.join(directory, filename) 
        scanner.setCameraDebugValues()
        actualResult = scanner.getVotesFromImage(fullFilename)
        if(actualResult != expectedList):
            cv2.waitKey(10001)
        self.assertEqual(actualResult, expectedList, msg="\nFailure for test image: {}".format(fullFilename))
        print("Success for" + filename)


    def x_test_card_1(self):
        l = getVotesFromImage("../testimages/card__859724459716.jpg")
        expected = [8, 5, 9, 7, 2, 4, 4, 5, 9, 7, 1, 6]
        self.assertEqual(l, expected)

    def test_debug_cards_testimages(self):
        #self.checkImage("../testimages/", "card_webcam_box2__896676796899.jpg")
        #self.checkImage("../testimages/", "card_webcam_box__785783566369.jpg")
        #self.checkImage("../testimages/", "card_blocks__265659759735.jpg")
        #self.checkImage("../testimages/", "card__859724459716.jpg")
        #self.checkImage("../testimages/", "card_webcam_rotated_bad_background_card__898656717575.jpg")
        #self.checkImage("../testimages/", "card_webcam_box3__741798877941.jpg")
        
        cv2.waitKey(10001)
        pass

    def Xtest_cards_testimages(self):
        self.check_all_images("../testimages/")

 
 
def test_generator(directory, filename):
    def test(self):
        self.checkImage(directory,filename)
    return test

if __name__ == '__main__':
    directory = "../testimages/"
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") and "__" in filename: 
            test_name = 'test_%s' % filename
            test = test_generator(directory, filename)
            setattr(TestSlamVotes , test_name, test)
    unittest.main()
