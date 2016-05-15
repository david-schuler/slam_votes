import unittest
import os
from scanner import getVotesFromImage



class TestSlamVotes(unittest.TestCase):
 
    def setUp(self):
        self.longMessage = True
        pass

    def check_all_images(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") and "__" in filename: 
                name = os.path.splitext(filename)[0]
                expectedStr = name.split("__")[1]
                expectedList = [int(x) for x in expectedStr]
                fullFilename = os.path.join(directory, filename) 
                actualResult = getVotesFromImage(fullFilename)
                self.assertEqual(actualResult, expectedList, msg="\nFailure for test image: {}".format(fullFilename))
                print("Success for" + filename)

    def x_test_card_1(self):
        l = getVotesFromImage("../testimages/card__859724459716.jpg")
        expected = [8, 5, 9, 7, 2, 4, 4, 5, 9, 7, 1, 6]
        self.assertEqual(l, expected)

    def test_cards_testimages(self):
        self.check_all_images("../testimages/")

 
 
if __name__ == '__main__':
    unittest.main()
