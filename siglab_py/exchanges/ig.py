import sys
from typing import Dict, Any
import logging
from datetime import datetime, timedelta

from trading_ig import IGService
from trading_ig.config import config

from siglab_py.exchanges.any_exchange import AnyExchange

'''
IG https://www.ig.com/

trading-ig (Open Source library):
    https://pypi.org/project/trading-ig/
    https://github.com/ig-python/trading-ig/blob/master/sample/rest.ipynb
    https://github.com/ig-python/trading-ig/blob/master/trading_ig/rest.py

    python -mpip install trading-ig
'''
class IG(AnyExchange):
    def __init__(self, *args: Dict[str, Any]) -> None:
        super().__init__(*args)
        
        self.name = 'ig'

if __name__ == '__main__':
    acc_number : str = "xxx"
    username : str = "xxx" # not your email
    password : str = "xxx"
    api_key : str = "xxx"
    acc_type : str = "demo" #  [live, demo] 

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )

    ig_service = IGService(
        username,
        password,
        api_key,
        acc_type,
        acc_number=acc_number,
    )

    ig_service.create_session()

    accounts = ig_service.fetch_accounts()
    print(f"accounts:\n{accounts}")

    # epic = "IX.D.ASX.IFM.IP"  # US (SPY) - mini
    epic = "CS.D.GBPUSD.CFD.IP"  # sample CFD epic

    # resolution = "D"
    # resolution = 'H'
    resolution = '15Min'

    num_points = 5000
    response = ig_service.fetch_historical_prices_by_epic_and_num_points(
        epic, resolution, num_points
    )
    print(response)
