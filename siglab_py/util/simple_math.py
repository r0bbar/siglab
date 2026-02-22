import math
import random
from typing import List, Dict, Union

from pandas import isna

def generate_rand_nums(
        range_min : float = 0,
        range_max : float = 1,
        size=100, # list size
        percent_in_range : float = 100,
        abs_min : float = 0,
        abs_max : float = 1
    ) -> List[float]:
    assert(range_min<range_max)

    if abs_min>range_min:
        abs_min = range_min
    if abs_max<range_max:
        abs_max = range_max

    result : List[float] = []
    for _ in range(int(size * percent_in_range/100)):
        result.append(random.uniform(range_min, range_max))
    for _ in range(size - len(result)):
        if random.uniform(0, 1)>0.5:
            result.append(random.uniform(abs_min, range_min))
        else:
            result.append(random.uniform(range_max, abs_max))
    
    random.shuffle(result)

    return result

def compute_level_increment(
    num : float,
    level_granularity : float = 0.01
) -> float:
    if math.isnan(num):
        return num
    level_size = num * level_granularity
    magnitude = math.floor(math.log10(abs(level_size)))
    base_increment = 10 ** magnitude
    rounded_level_size = round(level_size / base_increment) * base_increment
    return rounded_level_size

# https://norman-lm-fung.medium.com/levels-are-psychological-7176cdefb5f2
def round_to_level(
            num : float,
            level_granularity : float = 0.01
        ) -> float:
    if math.isnan(num):
        return num
    rounded_level_size = compute_level_increment(num, level_granularity)
    rounded_num = round(num / rounded_level_size) * rounded_level_size
    return rounded_num

def compute_adjacent_levels(
            num : float,
            level_granularity : float = 0.01,
            num_levels_per_side : int = 1
        ) -> Union[None, List[float]]:
    if math.isnan(num):
        return None
    if level_granularity!=0:
        rounded_level_size = compute_level_increment(num, level_granularity)
    else:
        # If caller don't specify level_granularity (i.e. level_granularity=0), then compute major rounded level based on msd (Most Significant Digit)
        msd = msd_power(num)
        rounded_level_size = 10**msd
        rounded_num = round(num / rounded_level_size) * rounded_level_size
        for i in range(num_levels_per_side):
            if rounded_num - (i+1)*rounded_level_size <= 0:
                rounded_level_size = 10**(msd-1)
                break
    rounded_num = round(num / rounded_level_size) * rounded_level_size
    levels = [ rounded_num ]
    levels = list(reversed([ rounded_num - (i+1)*rounded_level_size for i in list(range(num_levels_per_side))])) + levels + [ rounded_num + (i+1)*rounded_level_size for i in list(range(num_levels_per_side))]
    return levels

def bucket_series(
    values : List[float],
    outlier_threshold_percent : float = 0,
    level_granularity : float = 0.1 # 0.1 = 10%
) -> Dict[
            str, 
            Dict[str,Union[float, List[float]]]
        ]:
    buckets : Dict[
            str, 
            Dict[str,Union[float, List[float]]]
        ] = {}
    list_0_to_1 : bool = True if len([x for x in values if x<0 or x>1])/len(values)*100 <= outlier_threshold_percent else False
    list_m1_to_1 : bool = True if len([x for x in values if x<-1 or x>1])/len(values)*100 <= outlier_threshold_percent else False
    
    list_0_to_100 : bool = True if len([x for x in values if x<0 or x>100])/len(values)*100 <= outlier_threshold_percent else False
    if (
        list_0_to_100
        and (
            not min(values)<100*(outlier_threshold_percent/100) or not max(values)>100*(1-outlier_threshold_percent/100)
        )
    ):
        list_0_to_100 = False
    list_m100_to_100 : bool = True if len([x for x in values if x<-100 or x>100])/len(values)*100 <= outlier_threshold_percent else False
    if (
        list_m100_to_100
        and (
            not min(values)<-100*(1-outlier_threshold_percent/100) or not max(values)>100*(1-outlier_threshold_percent/100)
        )
    ):
        list_m100_to_100 = False

    def _generate_sequence(start, stop, step):
        result = []
        current = start
        num_steps = int((stop - start) / step) + 1
        for i in range(num_steps):
            result.append(round(start + i * step, 10))
        return result

    if list_0_to_1:
        step = round_to_level(
                1 * level_granularity,
                level_granularity=level_granularity
            )
        intervals = _generate_sequence(0.1, 1, step)
        last_interval = 0
        buckets[f"< 0"] = {
            'min' : float("-inf"),
            'max' : 0,
            'values' : [ x for x in values if x<0 ]
        }
        for interval in intervals:
            buckets[f"{last_interval} - {interval}"] = {
                'min' : last_interval,
                'max' : interval,
                'values' : [ x for x in values if x>=last_interval and x<interval ]
            }
            last_interval = interval
        buckets[f">1"] = {
                'min' : last_interval,
                'max' : float("inf"),
                'values' : [ x for x in values if x>=1 ]
        }
    
    elif not list_0_to_1 and list_m1_to_1:
        step = round_to_level(
                1 * level_granularity,
                level_granularity=level_granularity
            )
        intervals = _generate_sequence(-0.9, 1, step)
        last_interval = -1
        buckets[f"< -1"] = {
            'min' : float("-inf"),
            'max' : -1,
            'values' : [ x for x in values if x<-1 ]
        }
        for interval in intervals:
            buckets[f"{last_interval} - {interval}"] = {
                'min' : last_interval,
                'max' : interval,
                'values' : [ x for x in values if x>=last_interval and x<interval ]
            }
            last_interval = interval
        buckets[f">1"] = {
                'min' : last_interval,
                'max' : float("inf"),
                'values' : [ x for x in values if x>=1 ]
        }

    elif not list_0_to_1 and not list_m1_to_1 and list_0_to_100:
        step = round_to_level(
                100 * level_granularity,
                level_granularity=level_granularity
            )
        intervals = _generate_sequence(10, 100, step)
        last_interval = 0
        buckets[f"<0"] = {
            'min' : float("-inf"),
            'max' : 0,
            'values' : [ x for x in values if x<0 ]
        }
        for interval in intervals:
            buckets[f"{last_interval} - {interval}"] = {
                'min' : last_interval,
                'max' : interval,
                'values' : [ x for x in values if x>=last_interval and x<interval ]
            }
            last_interval = interval
        buckets[f">100"] = {
                'min' : last_interval,
                'max' : float("inf"),
                'values' : [ x for x in values if x>=100 ]
        }

    elif not list_0_to_1 and not list_m1_to_1 and not list_0_to_100 and list_m100_to_100:
        step = round_to_level(
                100 * level_granularity,
                level_granularity=level_granularity
            )
        intervals = _generate_sequence(-90, 100, step)
        last_interval = -100
        buckets[f"<-100"] = {
            'min' : float("-inf"),
            'max' : -100,
            'values' : [ x for x in values if x<-100 ]
        }
        for interval in intervals:
            buckets[f"{last_interval} - {interval}"] = {
                'min' : last_interval,
                'max' : interval,
                'values' : [ x for x in values if x>=last_interval and x<interval ]
            }
            last_interval = interval
        buckets[f">100"] = {
                'min' : last_interval,
                'max' : float("inf"),
                'values' : [ x for x in values if x>=100 ]
        }

    else:
        range_min = round_to_level(
            min(values),
            level_granularity=level_granularity
            )
        range_max = round_to_level(
            max(values),
            level_granularity=level_granularity
            )
        step = round_to_level(
                abs(range_max - range_min) * level_granularity,
                level_granularity=level_granularity
            )
        
        intervals = _generate_sequence(range_min+step, range_max, step)
        last_interval = range_min
        buckets[f"< {range_min}"] = {
            'min' : float("-inf"),
            'max' : range_min,
            'values' : [ x for x in values if x<range_min ]
        }
        for interval in intervals:
            buckets[f"{last_interval} - {interval}"] = {
                'min' : last_interval,
                'max' : interval,
                'values' : [ x for x in values if x>=last_interval and x<interval ]
            }
            last_interval = interval
        buckets[f"> {range_max}"] = {
                'min' : last_interval,
                'max' : float("inf"),
                'values' : [ x for x in values if x>=range_max ]
        }

    for key in buckets:
        bucket = buckets[key]
        assert(len([x for x in bucket['values'] if x<bucket['min'] or x>bucket['max']])==0) # type: ignore

    return buckets

def bucketize_val(
            x : float,
            buckets : Dict[
            str, 
            Dict[str,Union[float, List[float]]]
        ]
    ) -> Union[str,None]:
        for key in buckets:
            bucket = buckets[key]
            if x>=bucket['min'] and x<=bucket['max']: # type: ignore
                return key
        return None

def msd_power(x):
    """
    msd = Most Significant Digit.
    Returns the exponent of the most significant digit using logarithms.
    Very accurate and handles very large/small numbers well.
    """
    if x == 0:
        return None
    log = math.log10(abs(x))
    return math.floor(log)

def round_to_sigfigs(x, sigfigs=6):
    """
    Rounds a number to approximately `sigfigs` significant digits.
    
    Examples (with default sigfigs=6):
        34523.12345     → 34523.1
        123456.789      → 123457
        0.00034523123   → 0.000345231
        9876543210      → 9.87654e+09
        0.00123456789   → 0.001235
        9.87654321      → 9.87654
        123.0           → 123.0
    """
    if x == 0:
        return 0.0
    if not math.isfinite(x):
        return x
    power = msd_power(x)
    scale = 10 ** power
    normalized = abs(x) / scale
    rounded_normalized = round(normalized, sigfigs - 1)
    result = rounded_normalized * scale
    return math.copysign(result, x)