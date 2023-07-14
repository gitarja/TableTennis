import time
import datetime

s = "2022-11-17 09:58:33.216478"
dt = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
start_ts = time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)

ttl_ts = start_ts + 104.077067


# Converting timestamp to DateTime object
datetime_object = datetime.datetime.fromtimestamp(ttl_ts)

# printing resultant datetime object
print("Resultant datetime object:",datetime_object)