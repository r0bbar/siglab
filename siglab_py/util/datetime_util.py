from datetime import datetime, timedelta
from typing import Dict

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