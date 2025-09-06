import unittest
from typing import List

from util.simple_math import generate_rand_nums

class SimpleMathTests(unittest.TestCase):

    def test_generate_rand_nums(self):
        range_min : float = 0
        range_max : float = 1
        size : int = 100
        percentage_in_range : float = 91
        abs_min : float = -0.5
        abs_max : float = 1.1

        rand_nums : List[float] = generate_rand_nums(
                                        range_min = range_min,
                                        range_max = range_max,
                                        size = size, 
                                        percent_in_range = percentage_in_range,
                                        abs_min = abs_min,
                                        abs_max = abs_max
                                    )
        
        assert(len(rand_nums)==size)
        assert(len([x for x in rand_nums if x>=range_min and x<=range_max]) == (percentage_in_range/100) * size)
        assert(len([x for x in rand_nums if x<abs_min or x>abs_max]) == 0)


        range_min = -1
        range_max  = 1
        percentage_in_range = 91
        abs_min = -1.5
        abs_max = 1.5

        rand_nums : List[float] = generate_rand_nums(
                                        range_min = range_min,
                                        range_max = range_max,
                                        size = size, 
                                        percent_in_range = percentage_in_range,
                                        abs_min = abs_min,
                                        abs_max = abs_max
                                    )
        
        assert(len(rand_nums)==size)
        assert(len([x for x in rand_nums if x>=range_min and x<=range_max]) == (percentage_in_range/100) * size)
        assert(len([x for x in rand_nums if x<abs_min or x>abs_max]) == 0)


        range_min = 0
        range_max  = 100
        percentage_in_range = 91
        abs_min = -150
        abs_max = 150

        rand_nums : List[float] = generate_rand_nums(
                                        range_min = range_min,
                                        range_max = range_max,
                                        size = size, 
                                        percent_in_range = percentage_in_range,
                                        abs_min = abs_min,
                                        abs_max = abs_max
                                    )
        
        assert(len(rand_nums)==size)
        assert(len([x for x in rand_nums if x>=range_min and x<=range_max]) == (percentage_in_range/100) * size)
        assert(len([x for x in rand_nums if x<abs_min or x>abs_max]) == 0)


        range_min = -100
        range_max  = 100
        percentage_in_range = 91
        abs_min = -150
        abs_max = 150

        rand_nums : List[float] = generate_rand_nums(
                                        range_min = range_min,
                                        range_max = range_max,
                                        size = size, 
                                        percent_in_range = percentage_in_range,
                                        abs_min = abs_min,
                                        abs_max = abs_max
                                    )
        
        assert(len(rand_nums)==size)
        assert(len([x for x in rand_nums if x>=range_min and x<=range_max]) == (percentage_in_range/100) * size)
        assert(len([x for x in rand_nums if x<abs_min or x>abs_max]) == 0)