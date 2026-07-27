from datetime import datetime, timedelta, timezone
import zoneinfo
from typing import Dict

US_EASTERN = zoneinfo.ZoneInfo("America/New_York")
UTC = timezone.utc

'''
Be careful with Daylight Saving changes and US NY Cash Open
    Summer (EDT, DST is Active) (2nd Sun Mar - 1st Sun Nov) UTC offset -4, US Cash open at UTC 13:30
    Winter (Standard/EST, DST is Inactive)(1st Sun Nov - 2nd Sun Mar) UTC offset -5, US Cash open at UTC 14:30
'''
def is_us_dst(
    utc_now : datetime = datetime.now(UTC) # utc_now is passed as optional arg so for backtests, you can pass in historical time    
) -> bool:
    """Returns True if the US is currently in Daylight Saving Time (EDT)."""
    return utc_now.astimezone(US_EASTERN).dst() != timedelta(0)

def is_us_cash_open_effective(
    utc_now : datetime = datetime.now(UTC) # utc_now is passed as optional arg so for backtests, you can pass in historical time
) -> bool:
    if utc_now.hour in [13,14,15]:
        # initial half hour chaos after US NY Cash open: ET 9:30am to 10am. 
        if is_us_dst(utc_now): 
            if utc_now.hour>=13 and utc_now.hour<=14:
                return True
        else:
            if utc_now.hour>=14 and utc_now.hour<=15:
                return True

    return False

def parse_trading_window(
            today : datetime,
            window : Dict[str, str]
        ) :
        window_start : str = window['start']
        window_end : str = window['end']

        DayOfWeekMap : Dict[str, int] = {
            'Mon' : 0,
            'Tue' : 1,
            'Wed' : 2,
            'Thur' : 3,
            'Fri' : 4,
            'Sat' : 5,
            'Sun' : 6
        }
        today_dayofweek = today.weekday()

        window_start_dayofweek : int = DayOfWeekMap[window_start.split('_')[0]]
        window_start_hr : int = int(window_start.split('_')[-1].split(':')[0])
        window_start_min : int = int(window_start.split('_')[-1].split(':')[1])
        dt_window_start = today + timedelta(days=(window_start_dayofweek-today_dayofweek))
        dt_window_start = dt_window_start.replace(hour=window_start_hr, minute=window_start_min)

        window_end_dayofweek : int = DayOfWeekMap[window_end.split('_')[0]]
        window_end_hr : int = int(window_end.split('_')[-1].split(':')[0])
        window_end_min : int = int(window_end.split('_')[-1].split(':')[1])
        dt_window_end = today + timedelta(days=(window_end_dayofweek-today_dayofweek))
        dt_window_end = dt_window_end.replace(hour=window_end_hr, minute=window_end_min)

        return {
            'today' : today,
            'start' : dt_window_start,
            'end' : dt_window_end,
            'in_window' : (today<=dt_window_end) and (today>=dt_window_start)
        }