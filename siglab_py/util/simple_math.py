import math
import random
from typing import List

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

# https://norman-lm-fung.medium.com/levels-are-psychological-7176cdefb5f2
def round_to_level(
            num : float,
            level_granularity : float = 0.01
        ) -> float:
    level_size = num * level_granularity
    magnitude = math.floor(math.log10(level_size))
    base_increment = 10 ** magnitude
    rounded_level_size = round(level_size / base_increment) * base_increment
    rounded_num = round(num / rounded_level_size) * rounded_level_size
    return rounded_num