import requests
import pandas as pd
from datetime import datetime
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class KlineFetcher:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.RequestException, ValueError)),
        reraise=True,
    )
    def _http_get(self, url, params=None, headers=None, timeout=10, encoding=None):
        response = requests.get(
            url, params=params, headers=headers or self.headers, timeout=timeout
        )

        if encoding:
            response.encoding = encoding

        if response.status_code != 200:
            raise requests.RequestException(f"HTTP {response.status_code}")

        return response

    def get_kline_sina(self, code, start_date="2020-01-01", end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        market = "sz" if code.startswith(("0", "1", "3")) else "sh"
        url = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
        params = {
            "symbol": f"{market}{code}",
            "scale": "240",
            "ma": "no",
            "datalen": "1023",
        }
        try:
            response = self._http_get(url, params=params, timeout=10, encoding="utf-8")
            json_data = response.json()
        except Exception as e:
            print(f"[error] sina api failed - code: {code}, error: {e}")
            return None

        if not json_data or len(json_data) == 0:
            print(f"[error] sina api empty data - code: {code}")
            return None

        data_list = []
        for item in json_data:
            date_str = item.get("day", "")
            if not date_str:
                continue
            data_list.append(
                {
                    "date": date_str.replace("-", ""),
                    "open": float(item.get("open", 0)),
                    "close": float(item.get("close", 0)),
                    "high": float(item.get("high", 0)),
                    "low": float(item.get("low", 0)),
                    "volume": float(item.get("volume", 0)),
                }
            )
        return pd.DataFrame(data_list)

    def get_stock_info_tencent(self, code):
        market = "sz" if code.startswith(("0", "1", "3")) else "sh"
        url = f"https://web.sqt.gtimg.cn/q={market}{code}"

        try:
            response = self._http_get(url, timeout=10, encoding="gbk")
        except Exception as e:
            return None

        match = re.search(r'="([^"]+)"', response.text)
        if not match:
            return None

        data = match.group(1).split("~")
        if len(data) < 2 or not data[1]:
            return None

        return {
            "code": code,
            "name": data[1],
            "market": "SZSE" if market == "sz" else "SSE",
        }

    def fetch_data(self, code, start_date="2020-01-01", end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        apis = [
            ("Sina", lambda: self.get_kline_sina(code, start_date, end_date)),
        ]

        for api_name, api_func in apis:
            print(f"[info] trying {api_name} api...")
            df = api_func()

            if df is not None and len(df) > 0:
                print(f"[success] {api_name} api fetched {len(df)} records for {code}")
                return df

        print(f"[error] all apis failed for {code}")
        return None


def main():
    fetcher = KlineFetcher()
    examples = [("002230", "iFlytek"), ("159625", "ETF Fund"), ("510050", "50ETF")]
    print("\nexample codes:")
    for code, name in examples:
        print(f"  {code} - {name}")

    code = input("\nenter stock/fund code:").strip()

    info = fetcher.get_stock_info_tencent(code)
    if info:
        print(f"\ncode: {info['code']}")
        print(f"name: {info['name']}")
        print(f"market: {info['market']}")

    df = fetcher.fetch_data(code, start_date="2020-01-01")

    if df is None or len(df) == 0:
        print(
            "\nfailed to fetch data, please check if the code is correct or try again later"
        )
    print("\nlast 5 records preview:")
    print(df.tail().to_string(index=False))

    csv_filename = f"{code}_history_data.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")

    print(f"\ndata saved to: {csv_filename}")
    print(f"total records: {len(df)}")
    print(f"date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")


if __name__ == "__main__":
    main()
