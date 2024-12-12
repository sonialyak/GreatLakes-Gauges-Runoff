

import pandas as pd
import requests
from dataretrieval import nwis


#Import list of gauges 


gauge_list= pd.read_csv("C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Locations/compiled_gauges_no_duplicates.csv")


#Separate out US and CA gauges 

US_gauges = gauge_list[gauge_list['Country']=='US']
CA_gauges = gauge_list[gauge_list['Country']=='CA']


US_gauge_list= US_gauges['ID'].to_list()

CA_gauge_list= CA_gauges['ID'].to_list()



#Canada data

for i in CA_gauge_list:

    # URL of the CSV data
    url = 'https://wateroffice.ec.gc.ca/services/daily_data/csv/inline?stations[]='+i+'&parameters[]=level&parameters[]=flow&start_date=1950-10-01&end_date=2024-10-01'  # Replace with your actual URL
    
    try:
        # Send a GET request
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
    
        # Open a file in write mode to save the CSV data
        with open('C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Data/'+i+'.csv', 'w', newline='', encoding='utf-8-sig') as file:
            file.write(response.text)
    
        print("CSV data has been successfully downloaded and saved")
    
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        




#US Data

for i in US_gauge_list:

    siteNumber = i
    parameterCode = "00060" # Discharge
    startDate = "1950-10-01"
    endDate = "2024-10-01"
    
    # Retrieve the data
    dailyStreamflow = nwis.get_dv(sites=siteNumber, parameterCd=parameterCode, start=startDate, end=endDate) 
    streamflow=dailyStreamflow[0]
    streamflow.to_csv('C:/Users/rg727/Documents/Great Lake Project/Runoff/Gauge_Data/'+i+'.csv')  
    #print("Retrieved " + str(len(dailyStreamflow[0])) + " data values.")

