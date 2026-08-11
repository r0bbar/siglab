from datetime import datetime, timedelta, timezone
import zoneinfo
from typing import List, Dict

US_EASTERN = zoneinfo.ZoneInfo("America/New_York")
LONDON = zoneinfo.ZoneInfo("Europe/London")
CET = zoneinfo.ZoneInfo("Europe/Berlin")  # Central European Time (used by Germany, France, etc.)
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

'''
Returns True if the UK is currently in British Summer Time (BST).

BST runs from the last Sunday in March to the last Sunday in October.
During BST, UTC offset is +1; during GMT (winter), UTC offset is +0.
'''
def is_lse_dst(
    utc_now: datetime = datetime.now(UTC)
) -> bool:
    return utc_now.astimezone(LONDON).dst() != timedelta(0)

'''
Returns True if Central Europe is currently on Daylight Saving Time (CEST).
'''
def is_cet_dst(
    utc_now: datetime = datetime.now(UTC)
) -> bool: 
    return utc_now.astimezone(CET).dst() != timedelta(0)

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

'''
Returns True if utc_now falls within the 'effective' window around the LSE cash open.

Summer (BST, UTC+1): LSE open is 07:00 UTC → effective window 07:00-08:59 UTC
Winter (GMT, UTC+0): LSE open is 08:00 UTC → effective window 08:00-09:59 UTC
'''
def is_lse_cash_open_effective(
    utc_now: datetime = datetime.now(UTC)
) -> bool:
    if utc_now.hour in [7, 8, 9]:
        if is_lse_dst(utc_now):  # Summer (BST)
            # Check hours 7 and 8 (covers 07:00–08:59)
            if utc_now.hour >= 7 and utc_now.hour <= 8:
                return True
        else:  # Winter (GMT)
            # Check hours 8 and 9 (covers 08:00–09:59)
            if utc_now.hour >= 8 and utc_now.hour <= 9:
                return True
    return False

'''
Returns True if utc_now falls within the 'effective' window around the CET cash open.

Summer (CEST, UTC+2): Xetra/Euronext open is 07:00 UTC → effective window 07:00-08:59 UTC
Winter (CET, UTC+1): Xetra/Euronext open is 08:00 UTC → effective window 08:00-09:59 UTC
'''
def is_cet_cash_open_effective(
    utc_now: datetime = datetime.now(UTC)
) -> bool:
    if utc_now.hour in [7, 8, 9]:
        if is_cet_dst(utc_now):  # Summer (CEST)
            if utc_now.hour >= 7 and utc_now.hour <= 8:
                return True
        else:  # Winter (CET)
            if utc_now.hour >= 8 and utc_now.hour <= 9:
                return True
    return False

'''
APAC (Asia-Pacific) Trading Hours
    UTC 21:00 - 09:00 (approximate range)
    Major financial centers: Tokyo, Hong Kong, Singapore, Sydney

EMEA (Europe, Middle East, Africa) Trading Hours
    UTC 07:00 - 16:00 (approximate range)
    Major financial centers: London, Frankfurt, Paris, Zurich, Dubai

US Trading Hours
    UTC 13:00 - 22:00 (approximate range)
    Major financial centers: New York, Chicago
    Key markets: NYSE, NASDAQ

utcnow and utcfromtimestamp been deprecated in Python 3.12 
https://www.pythonmorsels.com/converting-to-utc-time/

Example, UTC 23:00 is 3rd hour in APAC trading session
    utc_hour = 23
    i = get_regions_trading_utc_hours()['APAC'].index(utc_hour)
    assert(i==2)
'''
def get_regions_trading_utc_hours(
    utc_now: datetime = datetime.now(UTC)
) -> Dict[str, List[int]]:
    # APAC is static, no Summer vs Winter time
    apac_hours = [21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # EMEA (union of LSE and CET)
    # Both open at 07:00 UTC in summer, 08:00 UTC in winter
    # Both close at 15:30 UTC in summer, 16:30 UTC in winter
    if is_lse_dst(utc_now):  # Summer (BST / CEST)
        emea_hours = [7, 8, 9, 10, 11, 12, 13, 14, 15]
    else:  # Winter (GMT / CET)
        emea_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16]

    # AMER (US)
    # Open at 13:30 UTC in summer, 14:30 UTC in winter
    # Close at 20:00 UTC in summer, 21:00 UTC in winter
    if is_us_dst(utc_now):  # Summer (EDT)
        amer_hours = [13, 14, 15, 16, 17, 18, 19, 20]
    else:  # Winter (EST)
        amer_hours = [14, 15, 16, 17, 18, 19, 20, 21]

    return {
        'APAC': apac_hours,
        'EMEA': emea_hours,
        'AMER': amer_hours
    }

def timestamp_to_active_trading_regions(
        timestamp_ms : int
) -> List[str]:
    active_trading_regions : List[str] = []

    dt_utc = datetime.fromtimestamp(int(timestamp_ms / 1000), tz=timezone.utc)
    utc_hour = dt_utc.hour
    if utc_hour in get_regions_trading_utc_hours(utc_now=dt_utc)['APAC']:
        active_trading_regions.append("APAC") 

    if utc_hour in get_regions_trading_utc_hours(utc_now=dt_utc)['EMEA']:
        active_trading_regions.append("EMEA")

    if utc_hour in get_regions_trading_utc_hours(utc_now=dt_utc)['AMER']:
        active_trading_regions.append("AMER")

    return active_trading_regions

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