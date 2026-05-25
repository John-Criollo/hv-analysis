import os
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from variables import *


client = Client(url)

def get_data(stations = ['PE50'], day = "2008-8-21T00:00:00" ,st=0,en=24):
    '''
    This function is for downloading data from the stations.
    The reference time is the Peru time which is UTC-5
    The variables used are:
    -stations = an array with all the stations that are going to be used.
    -day = Here we have to add the day of analysis. "year-month-dayT00:00:00"
    -start = Here we put the beggining of the data slice we want to analyze.
    -end = Here we put the end of the data that will be requested.
    The predetermined value will be 24 hours of data that will be downloaded.
    '''
    perutime = UTCDateTime(day) - 5*3600
    start = perutime + st*3600
    end = start + en * 3600

    os.makedirs(output_dir, exist_ok=True)
    for station in stations:
        print(f"Downloading 24 hour of data for {station}...")
        try:
            base_filename = f"{network}_{station}_{start.strftime('%Y%m%d_%H%M%S')}.mseed"
            filename = os.path.join(output_dir, base_filename)
            
            if os.path.exists(filename):
                print(f"File {filename} already exists. Skipping download.")
            else:
                # Request waveforms
                st = client.get_waveforms(network, station, location, channel, start, end)
                st.write(filename, format="MSEED")
                
                print(f"Success! Data saved to {filename}")
                print(st) # Displays info about the downloaded traces
        except Exception as e:
            print(f"Download failed: {e}")
def get_response(network = network, stations = ['PE50'], day = "2008-8-21T00:00:00",st = 0):
    perutime = UTCDateTime(day) - 5*3600
    start = perutime + st*3600
    '''
    This is a function to download the response of a station, its not necessary in this case.
    '''
    for station in stations:
        try:
            base_filename = f"{network}_{station}_response.xml"
            filename = os.path.join(output_dir, base_filename)
            inv = client.get_stations(network=network, station=station,location="*", channel="*", starttime= start, level = "response")
            inv.write(filename, format = 'STATIONXML')
            print(f"Successfully saved inventory to: {os.path.abspath(filename)}")
            return None

        except Exception as e:
            print(f"An error occurred while fetching/saving data: {e}")
            return None
