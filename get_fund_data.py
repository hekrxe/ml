import requests
import pandas as pd
import json


# 从天天基金网获取基金历史数据
def get_fund_history(fund_code):
    try:
        # 天天基金网基金历史数据接口
        url = f"https://fund.eastmoney.com/pingzhongdata/{fund_code}.js"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers)
        response.encoding = "utf-8"

        # 解析JavaScript数据
        content = response.text
        # 提取净值数据
        import re

        nav_data = re.search(r"var Data_netWorthTrend = (\[.*?\]);", content, re.DOTALL)

        if nav_data:
            data_list = json.loads(nav_data.group(1))

            # 解析数据
            data = []
            for item in data_list:
                date = pd.to_datetime(item["x"], unit="ms")
                unit_nav = item["y"]  # 单位净值
                accum_nav = item["equityReturn"]  # 累计净值

                data.append(
                    {"date": date, "unit_nav": unit_nav, "accum_nav": accum_nav}
                )

            if data:
                # 创建DataFrame
                df = pd.DataFrame(data)
                # 按日期排序
                df = df.sort_values("date")
                # 保存为CSV文件
                filename = f"fund_{fund_code}_history.csv"
                df.to_csv(filename, index=False, encoding="utf-8-sig")

                print(f"数据已保存到 {filename}")
                print("\n数据预览:")
                print(df.head())
                print("\n数据统计:")
                print(f"数据行数: {len(df)}")
                print(f"起始日期: {df['date'].min().strftime('%Y-%m-%d')}")
                print(f"结束日期: {df['date'].max().strftime('%Y-%m-%d')}")
            else:
                print("未获取到数据，可能基金代码不存在或无数据")
        else:
            print("未找到净值数据，可能基金代码不存在或接口已变化")
    except Exception as e:
        print(f"获取数据时出错: {e}")


if __name__ == "__main__":
    fund_code = "017057"
    print(f"正在获取基金 {fund_code} 的历史价格数据...")
    get_fund_history(fund_code)
