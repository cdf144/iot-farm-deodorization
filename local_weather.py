from datetime import datetime
from dateutil.relativedelta import relativedelta
from meteostat import hourly

# https://oscar.wmo.int/surface/#/search/station/stationReportDetails/0-20000-0-48840
station_id = "48840"

end_date = datetime.now()
start_date = end_date - relativedelta(months=3)
data = hourly(station_id, start_date, end_date)

df = data.fetch()

if df is not None and not df.empty:
    weather_data = df[["temp", "rhum"]]
    print(
        f"Downloaded Temperature and RH time-series data in local area with {len(weather_data)} rows"
    )
    print(weather_data.head())

    weather_data.to_csv("local_weather.csv")
else:
    print("No data found for local station in this time frame")
