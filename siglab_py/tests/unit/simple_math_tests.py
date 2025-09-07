import unittest
from typing import List, Dict, Union

from util.simple_math import generate_rand_nums, round_to_level, bucket_series, bucketize_val

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

    def test_round_to_level(self):
        prices = [ 
            { 'price' : 15080, 'rounded' : 15000}, 
            { 'price' : 15180, 'rounded' : 15200}, 
            { 'price' : 25080, 'rounded' : 25200}, 
            { 'price' : 25180, 'rounded' : 25200}, 
            { 'price' : 25380, 'rounded' : 25500}, 
            { 'price' : 95332, 'rounded' : 95000}, 
            { 'price' : 95878, 'rounded' : 96000}, 
            { 'price' : 103499, 'rounded' : 103000}, 
            { 'price' : 103500, 'rounded' : 104000}, 
            { 'price' : 150800, 'rounded' : 150000}, 
            { 'price' : 151800, 'rounded' : 152000}
        ]
        for entry in prices:
            price = entry['price']
            expected = entry['rounded']
            rounded_price = round_to_level(price, level_granularity=0.01)
            print(f"{price} rounded to: {rounded_price}")
            assert(rounded_price==expected)

    def test_bucket_series(self):

        level_granularity : float = 0.1

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

        buckets : Dict[
            str, 
            Dict[str,Union[float, List[float]]]
        ] = bucket_series(
                                                values = rand_nums,
                                                outlier_threshold_percent = 10,
                                                level_granularity=level_granularity
                                            )
        
        bucketized = []
        for num in rand_nums:
            bucketized.append(
                bucketize_val(num, buckets=buckets)
            )

        
        range_min = -1
        range_max = 1
        size : int = 100
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

        buckets = bucket_series(
                                    values = rand_nums,
                                    outlier_threshold_percent = 10,
                                    level_granularity=level_granularity
                                )
        

        range_min = 0
        range_max = 100
        size : int = 100
        percentage_in_range = 91
        abs_min = -0.5
        abs_max = 150

        rand_nums : List[float] = generate_rand_nums(
                                        range_min = range_min,
                                        range_max = range_max,
                                        size = size, 
                                        percent_in_range = percentage_in_range,
                                        abs_min = abs_min,
                                        abs_max = abs_max
                                    )

        buckets = bucket_series(
                                    values = rand_nums,
                                    outlier_threshold_percent = 10,
                                    level_granularity=level_granularity
                                )
        

        range_min = -100
        range_max = 100
        size : int = 100
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

        buckets = bucket_series(
                                    values = rand_nums,
                                    outlier_threshold_percent = 10,
                                    level_granularity=level_granularity
                                )
        

        range_min = 20_000
        range_max = 120_000
        size : int = 100
        percentage_in_range = 91
        abs_min = 15_000
        abs_max = 130_000

        rand_nums : List[float] = generate_rand_nums(
                                        range_min = range_min,
                                        range_max = range_max,
                                        size = size, 
                                        percent_in_range = percentage_in_range,
                                        abs_min = abs_min,
                                        abs_max = abs_max
                                    )

        buckets = bucket_series(
                                    values = rand_nums,
                                    outlier_threshold_percent = 10,
                                    level_granularity=level_granularity
                                )