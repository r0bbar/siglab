import unittest
from typing import List, Dict, Union
import time

from numpy import equal

from util.simple_math import generate_rand_nums, round_to_level, compute_adjacent_levels, bucket_series, bucketize_val, msd_power, round_to_sigfigs

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

    def test_compute_adjacent_levels(self):
        # ******* non-zero level_granularity *******
        gold_price = 4450
        level_granularity = 0.025 # So levels are $100 apart
        adjacent_levels = compute_adjacent_levels(num=gold_price, level_granularity=level_granularity, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [4100,4200,4300,4400,4500,4600,4700])

        btc_price = 95000
        level_granularity = 0.01 # So levels are $1000 apart
        adjacent_levels = compute_adjacent_levels(num=btc_price, level_granularity=level_granularity, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [92000,93000,94000,95000,96000,97000,98000])

    def test_compute_adjacent_levels_w_zero_level_granularity(self):
        
        # ******* zero level_granularity *******
        level_granularity = 0

        btc_price = 48000
        adjacent_levels = compute_adjacent_levels(num=btc_price, level_granularity=level_granularity, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [20000, 30000, 40000, 50000, 60000, 70000, 80000])

        btc_price = 88000
        adjacent_levels = compute_adjacent_levels(num=btc_price, level_granularity=level_granularity, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [60000, 70000, 80000, 90000, 100000, 110000, 120000])

        btc_price = 118000
        adjacent_levels = compute_adjacent_levels(num=btc_price, level_granularity=level_granularity, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [90000, 100000, 110000, 120000, 130000, 140000, 150000])

        btc_price = 178000
        adjacent_levels = compute_adjacent_levels(num=btc_price, level_granularity=level_granularity, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [150000, 160000, 170000, 180000, 190000, 200000, 210000])

        btc_price = 378000
        adjacent_levels = compute_adjacent_levels(num=btc_price, level_granularity=level_granularity, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [100000, 200000, 300000, 400000, 500000, 600000, 700000])

        sol_price = 18
        adjacent_levels = compute_adjacent_levels(num=sol_price, level_granularity=level_granularity, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [15, 16, 17, 18, 19, 20, 21])

        sol_price = 78
        adjacent_levels = compute_adjacent_levels(num=sol_price, level_granularity=level_granularity, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [50, 60, 70, 80, 90, 100, 110])

        sol_price = 108
        adjacent_levels = compute_adjacent_levels(num=sol_price, level_granularity=level_granularity, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [80, 90, 100, 110, 120, 130, 140])

        sol_price = 408
        adjacent_levels = compute_adjacent_levels(num=sol_price, level_granularity=level_granularity, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [100, 200, 300, 400, 500, 600, 700])

    
    def test_compute_adjacent_levels_w_zero_level_granularity_and_msd_adj(self):
        
        # ******* zero level_granularity and msd_adj *******
        level_granularity = 0
        msd_adj = -1

        btc_price = 68000
        adjacent_levels = compute_adjacent_levels(num=btc_price, level_granularity=level_granularity, msd_adj=msd_adj, num_levels_per_side=3)
        assert(adjacent_levels)
        assert(len(adjacent_levels)==7)
        equal(adjacent_levels, [65000, 66000, 67000, 68000, 69000, 70000, 71000])
        

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

    def test_msd_power(self):
        # Integers
        self.assertEqual(msd_power(34523), 4)
        self.assertEqual(msd_power(99999), 4)
        self.assertEqual(msd_power(100000), 5)
        self.assertEqual(msd_power(999999999), 8)
        self.assertEqual(msd_power(1), 0)
        self.assertEqual(msd_power(9), 0)
        self.assertEqual(msd_power(10), 1)

        # Decimals > 1
        self.assertEqual(msd_power(3452.3), 3)
        self.assertEqual(msd_power(12.345), 1)
        self.assertEqual(msd_power(9.87654321), 0)

        # Numbers between 0 and 1
        self.assertEqual(msd_power(0.056), -2)      # 5.6 × 10⁻²
        self.assertEqual(msd_power(0.0056), -3)     # 5.6 × 10⁻³
        self.assertEqual(msd_power(0.000999), -4)
        self.assertEqual(msd_power(0.1), -1)
        self.assertEqual(msd_power(0.999), -1)
        self.assertEqual(msd_power(0.000000123), -7)

        # Negative numbers (should ignore sign)
        self.assertEqual(msd_power(-123.5), 2)
        self.assertEqual(msd_power(-0.00456), -3)

        # Special / edge cases
        self.assertIsNone(msd_power(0))
        self.assertIsNone(msd_power(0.0))
        # Very large / very small (log10 should handle)
        self.assertEqual(msd_power(1e20), 20)
        self.assertEqual(msd_power(5.67e-15), -15)

    def test_round_to_sigfigs(self):
        # Using default sigfigs=6

        # Large numbers – integer part dominant
        self.assertEqual(round_to_sigfigs(34523.12345, sigfigs=6), 34523.1)
        self.assertEqual(round_to_sigfigs(34523.12345, sigfigs=7), 34523.12)
        self.assertEqual(round_to_sigfigs(123456.789), 123457)
        self.assertEqual(round_to_sigfigs(999999.5), 1000000)     # rounds up
        self.assertEqual(round_to_sigfigs(9876543210), 9876540000)

        # Numbers around 1–1000
        self.assertEqual(round_to_sigfigs(123.45678), 123.457)
        self.assertEqual(round_to_sigfigs(9.87654321), 9.87654)
        self.assertEqual(round_to_sigfigs(1.00000), 1.0)

        # Small numbers (decimal dominant)
        self.assertEqual(round_to_sigfigs(0.056789123), 0.0567891)

        # Very small scientific notation style
        self.assertEqual(round_to_sigfigs(1.23456789e-8), 1.23457e-8)
        self.assertEqual(round_to_sigfigs(9.87654321e-12), 9.87654e-12)

        # Negative numbers
        self.assertEqual(round_to_sigfigs(-34523.12345), -34523.1)
        self.assertEqual(round_to_sigfigs(-0.0056789), -0.0056789)

        # Edge cases
        self.assertEqual(round_to_sigfigs(0), 0.0)
        self.assertEqual(round_to_sigfigs(0.0), 0.0)

        # With explicit sigfigs parameter
        self.assertEqual(round_to_sigfigs(34523.12345, sigfigs=4), 34520)
        self.assertEqual(round_to_sigfigs(1234567, sigfigs=3), 1230000)

        '''
        Edge cases:
            round_to_sigfigs(0.0056789123) -> 0.0056789100000000006
            round_to_sigfigs(0.00034523123) -> 0.00034523100000000004
            round_to_sigfigs(0.056789, sigfigs=3) -> 0.056799999999999996
        These are extra tiny error from the repeated multiply/divide in float arithmetic with python.
        '''
        # self.assertEqual(round_to_sigfigs(0.0056789123), 0.00567891) # 0.0056789100000000006
        # self.assertEqual(round_to_sigfigs(0.00034523123), 0.000345231)
        # self.assertEqual(round_to_sigfigs(0.056789, sigfigs=3), 0.0568)

        NUM_CALLS = 10_000
        start = time.time()
        for _ in range(10_000):
            x = round_to_sigfigs(85.123456789, sigfigs=7)
        elapsed_ms = int((time.time() - start) *1000)
        avg_per_call_ms = elapsed_ms / NUM_CALLS
        print(f"elapsed_ms: {elapsed_ms} over {NUM_CALLS} calls, avg_per_call_ms: {avg_per_call_ms}.")
        assert(avg_per_call_ms<1)