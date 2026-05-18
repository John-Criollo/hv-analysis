from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from variables import *

perutime = UTCDateTime(analysis_day) - 5*3600 
start = perutime  
end = start + 24*3600

client = Client(url)
available_stations = []
missing_stations = []

print(f"Checking data availability for {analysis_day}...\n")

for station in stations:
    try:
        inv = client.get_stations(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=start,
            endtime=end,
            level="channel"
        )
    
        if len(inv) > 0:
            available_stations.append(station)
            print(f"Station {station}: Data available")
            
    except FDSNNoDataException:
        missing_stations.append(station)
        print(f"Station {station}: No data")
    except Exception as e:
        missing_stations.append(station)
        print(f"Station {station}: Error checking status ({e})")

print("\n" + "="*40)
print("FINAL SUMMARY REPORT:")
print("="*40)

# Evaluate the three conditional cases for the final output printout
if len(available_stations) == len(stations):
    # Case 1: All stations are available
    print("Data for all stations is available.")

elif len(available_stations) == 0:
    # Case 3: No data is available for any of the stations
    print("No data is available for these stations.")

else:
    # Case 2: Some stations are missing
    missing_str = ", ".join(missing_stations)
    print(f"Data not available for stations: {missing_str}")