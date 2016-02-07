import unittest
from scanner import getVotesFromImage



class TestSlamVotes(unittest.TestCase):
 
    def setUp(self):
        pass
    def test_card_1(self):
        l = getVotesFromImage("../testimages/card__859724459716.jpg")
        expected = [8, 5, 9, 7, 2, 4, 4, 5, 9, 7, 1, 6]
        self.assertEqual(l, expected)

 
 
if __name__ == '__main__':
    unittest.main()
