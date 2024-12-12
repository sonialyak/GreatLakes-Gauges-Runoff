## Retrieve Data for all Natural Gauges in the region


```python
import pandas as pd
import numpy as np
from dataretrieval import nwis, utils, codes
import matplotlib.pyplot as plt
import dataretrieval.nwis as nwis
import statsmodels.api as sm
import matplotlib.dates as mdates
import os
import csv
import requests
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
```


```python
#Create dataset will all sites from US
dfSites = pd.read_csv('gauge_info.csv')
us_sites = dfSites[dfSites['Country'] == 'US']['ID'].tolist()
print(us_sites)
```

    ['04040500', '04066500', '04067958', '04074950', '04124000', '04125460', '04221000', '04224775', '04185000', '04196800', '04197100', '04136000', '04213000', '04256000', '04057510', '04126970', '04122500', '04085200', '04059500', '04056500', '04015330', '04031000', '04045500', '04024430', '04102776', '04122200', '04027000']
    


```python
#Create dataset will all sites from US
dfSites = pd.read_csv('gauge_info.csv')
ca_sites = dfSites[dfSites['Country'] == 'CA']['ID'].tolist()
print(ca_sites)

# Define the directory where you want to save the file
data_directory = './data/Great_Lakes'  # Example path in Jupyter environment
```

    ['02GG006', '02GG002', '02GD004', '02BF002', '02AD010', '02JC008', '02KF011', '02LB008', '02HL003', '02HL004', '02HL005', '02GA010', '02GA018', '02GA038', '02DC012', '02EA005', '02EB014', '02EC002', '02EC018', '02ED003', '02ED015', '02ED029', '02ED101', '02ED102', '02FE009', '02GA047', '02GC010', '02EA018', '02EC011', '02EC019', '02ED032', '02FA004', '02FC016', '02FE008', '02FE010', '02FE013', '02HC025', '02HL008', '02HM010', '02JB013', '02JE027', '02KC018', '02KD002', '02LB007', '02LA007', '02GC002', '02GG003', '02GG009', '02CB003', '02ED024', '02CF007', '02HD012', '02HC049', '02HC030', '02HA006', '02BF001', '02AB021', '02AB017', '02AC001', '02BB003', '02AC002', '02GB007', '02CF011', '02HB029', '02GE007', '02GG013', '02AE001', '02BA003', '02BA006', '02LG005', '02LD005', '02LB032']
    


```python
# Define water year function
def WaterYear(datetime):
    if datetime.month >= 10:
        return datetime.year + 1
    else:
        return datetime.year 


#Define season functions
def Season(date):
    # Convert the input date to a datetime object if it's not already
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date, format="%Y-%m-%d")
    # Calculate the day of year for the given date
    day_of_year = date.dayofyear
    # Define the seasons using dayofyear
    winter_start = pd.to_datetime("12-21", format="%m-%d").dayofyear
    spring_start = pd.to_datetime("03-21", format="%m-%d").dayofyear
    summer_start = pd.to_datetime("06-21", format="%m-%d").dayofyear
    fall_start = pd.to_datetime("09-21", format="%m-%d").dayofyear
    # Determine the season based on the day of year
    if day_of_year >= winter_start or day_of_year < spring_start:
        return 'Winter'
    elif day_of_year >= spring_start and day_of_year < summer_start:
        return 'Spring'
    elif day_of_year >= summer_start and day_of_year < fall_start:
        return 'Summer'
    else:
        return 'Fall'

```


```python
s='04256000'
```


```python
# Get daily values (dv)
DV = nwis.get_record(sites=s, service='dv',start='1970-10-01')
df = pd.DataFrame(DV)

#Pull coordinates
LocTest= nwis.get_record(sites=s, service='site')
Lat=LocTest['dec_lat_va'].values[0]
Long=LocTest['dec_long_va'].values[0]

# Convert index to datetime and apply water year function
df.index = pd.to_datetime(df.index)
df['water_year'] = df.index.to_series().apply(WaterYear)
df['Season'] = df.index.to_series().apply(Season)
# Ensure '00060_Mean' is numeric
df['00060_Mean'] = pd.to_numeric(df['00060_Mean'], errors='coerce')

# Find the last year
last_year = df['water_year'].max()

# Filter out the rows where the year is the last
df = df[(df['water_year'] != last_year)]

##Finding Attributes
#Mean
# Ensure the '00060_Mean' column is numeric
df['00060_Mean'] = pd.to_numeric(df['00060_Mean'], errors='coerce')
for i in df.index:
    if df.loc[i, '00060_Mean'] > 0:
        df.loc[i, '00060_Mean'] = df.loc[i, '00060_Mean']  # This keeps the value unchanged
    else:
        df.loc[i, '00060_Mean'] = np.nan  # Assign NaN for values <= 0
# Calculate the the mean of '00060_Mean' grouped by 'water_year'
water_yearly_avg = df.groupby('water_year')['00060_Mean'].mean()

# Calculate the maximum for each water year
annual_max = df.groupby('water_year')['00060_Mean'].max()

#7-Day Low Flow
# Compute 7-day rolling minimum
df['7_day_mean'] = df['00060_Mean'].rolling(window=7, min_periods=1).mean()
# Calculate the minimum of the 7 day rolling mean for each water year
seven_day_low_flow = df.groupby('water_year')['7_day_mean'].min()

# Check the 'value' column for discharge data
df['discharge'] = df['00060_Mean'] 
seconds_per_day = 24 * 60 * 60
df['daily_volume_cubic_ft'] = df['discharge'] * seconds_per_day
df['cumulative_volume'] = df.groupby('water_year')['daily_volume_cubic_ft'].cumsum()
df['date_column'] = df.index  # Ensure the index is datetime


## Calculations for center volume / center mass
total_volume_each_wateryear =df.groupby('water_year')['daily_volume_cubic_ft'].sum()
half_volume_each_wateryear = total_volume_each_wateryear / 2
half_volume_dates = {}
# Iterate over each unique water year
for water_year in df['water_year'].unique():
    # Get the half volume for the current water year
    half_volume = half_volume_each_wateryear[water_year]
    # Get the subset of data for the current water year
    year_data = df[df['water_year'] == water_year]
    # Find the index where cumulative volume reaches or exceeds half of total volume
    half_volume_index = year_data[year_data['cumulative_volume'] >= half_volume].index
    if not half_volume_index.empty:
        # Retrieve the actual date from the DataFrame using the first index
        first_index = half_volume_index[0]  # Get the first index
        half_volume_dates[water_year] = df.at[first_index, 'date_column']  # Use 'at' to get the value
 # Convert dates to just month and day format (MM-DD)
# Convert dates to just month and day format for display, but keep the original datetime for plotting
half_volume_dates_values = list(half_volume_dates.values())  # Extract the dates first
formatted_dates = [date.timetuple().tm_yday if isinstance(date, pd.Timestamp) else None for date in half_volume_dates_values]
# Create plot for center of volume date
water_years = list(half_volume_dates.keys())
half_volume_dates = list(half_volume_dates.values())
# Convert dates to day of year for sorting and plotting
formatted_dates = [date.timetuple().tm_yday for date in half_volume_dates]


#Calculates linear regression lines
# Linear Regression Yearly Avg
X = water_yearly_avg.index.values.reshape(-1, 1)  # Independent variable (water years)
y = water_yearly_avg.values  # Dependent variable (average flow values)
# Add a constant for the intercept
X_const = sm.add_constant(X)
# Fit the model using statsmodels
model = sm.OLS(y, X_const).fit()
# Predict values
y_pred_water_yearly_avg = model.predict(X_const)
# Get the slope and p-value
slope_water_yearly_avg = model.params[1]  # The slope coefficient
p_value_water_yearly_avg = model.pvalues[1]  # The p-value for the slope
# Print the results
print('\n',f'For Water Year Average of site {s}', f'Slope: {slope_water_yearly_avg}, P-value: {p_value_water_yearly_avg}')


# Linear Regression Annual Max
X = annual_max.index.values.reshape(-1, 1)  # Independent variable (water years)
y = annual_max.values  # Dependent variable (average flow values)
# Add a constant for the intercept
X_const = sm.add_constant(X)
# Fit the model using statsmodels
model = sm.OLS(y, X_const).fit()
# Predict values
y_pred_annual_max = model.predict(X_const)
# Get the slope and p-value
slope_water_annual_max = model.params[1]  # The slope coefficient
p_value_water_annual_max = model.pvalues[1]  # The p-value for the slope
# Print the results
print('\n',f'For Annual Maximum Flow of site {s}', f'Slope: {slope_water_annual_max}, P-value: {p_value_water_annual_max}')


# Linear Regression 7-Day Low Flow
X = seven_day_low_flow.index.values.reshape(-1, 1)  # Independent variable (water years)
y = seven_day_low_flow.values  # Dependent variable (average flow values)
# Add a constant for the intercept
X_const = sm.add_constant(X)
# Fit the model using statsmodels
model = sm.OLS(y, X_const).fit()
# Predict values
y_pred_seven_day_low_flow = model.predict(X_const)
# Get the slope and p-value
slope_seven_day_low_flow = model.params[1]  # The slope coefficient
p_value_seven_day_low_flow = model.pvalues[1]  # The p-value for the slope
print('\n',f'For 7-Day Low Flow of site {s}', f'Slope: {slope_seven_day_low_flow}, P-value: {p_value_seven_day_low_flow}')




# Create a plot for the information with linear regression lines
plt.title(f'Post-1970s Daily Average Flow for Site {s}', fontsize=16)
plt.plot(water_yearly_avg.index, water_yearly_avg.values, label='Daily Average Flow', color='blue', marker='o')
plt.plot(water_yearly_avg.index, y_pred_water_yearly_avg, label='Daily Avg Linear Regression Line', color='red')
plt.ylabel('Flow (units)')
plt.legend()
plt.show()

plt.title(f'Post-1970s Yearly Max Flow for Site {s}', fontsize=16)
plt.plot(annual_max.index, annual_max.values, label='Yearly Max Flow', color='orange', marker='o')
plt.plot(annual_max.index, y_pred_annual_max, label='Yearly Max Linear Regression Line', color='red')
plt.ylabel('Flow (units)')
plt.legend()
plt.show()

plt.title(f'Post-1970s 7-Day Low Flow for Site {s}', fontsize=16)
plt.plot(seven_day_low_flow.index, seven_day_low_flow.values, label='7 Day Low Flow', color='pink', marker='o')
plt.plot(seven_day_low_flow.index, y_pred_seven_day_low_flow, label='7-Day Low Flow Linear Regression Line', color='red')
plt.ylabel('Flow (units)')
plt.legend()
plt.show()



#Linear Regression for Center Mass Volume Date
X = water_years  # Independent variable (water years)
y = formatted_dates  # Dependent variable (average flow values)
# Add a constant for the intercept
X_const = sm.add_constant(X)
# Fit the model using statsmodels
model = sm.OLS(y, X_const).fit()
# Predict values
y_Winter_Spring_Center_Volume_Date = model.predict(X_const)
# Get the slope and p-value
slope_WSCV = model.params[1]  # The slope coefficient
p_value_WSCV = model.pvalues[1]  # The p-value for the slope
print('\n',f'For Winter-Spring Center Volume Date of site {s}', f'Slope: {slope_WSCV}, P-value: {p_value_WSCV}')



# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(water_years, formatted_dates, color='blue', marker='o')
plt.plot(water_years, y_Winter_Spring_Center_Volume_Date, label='Center Volume Date Regression Line', color='red')
plt.title(f'Post-1970s Winter-Spring Center Volume Date for Site {s}', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Date')
plt.xticks(rotation=45)
# Format the y-axis labels to show MM-DD
ax = plt.gca()
ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.show()

    # Calculate seasonal averages
water_seasonly_avg = df.groupby([df['Season'], df.index.year])['00060_Mean'].mean().reset_index()
water_seasonly_avg.columns = ['Season', 'Year', 'Seasonal_Avg']
# Ensure the Year column is converted to integers
water_seasonly_avg['Year'] = water_seasonly_avg['Year'].astype(int)

# Split the data into separate DataFrames for each season
fall = water_seasonly_avg[water_seasonly_avg['Season'] == 'Fall']
winter = water_seasonly_avg[water_seasonly_avg['Season'] == 'Winter']
spring = water_seasonly_avg[water_seasonly_avg['Season'] == 'Spring']
summer = water_seasonly_avg[water_seasonly_avg['Season'] == 'Summer']


# Linear Regression for each season
##Fall

X = fall['Year']  # Independent variable (year)
y = fall['Seasonal_Avg']  # Dependent variable (average flow values)

# Add a constant for the intercept
X_const = sm.add_constant(X)

# Fit the model using statsmodels
model = sm.OLS(y, X_const).fit()

# Get the slope and p-value
p_value = model.pvalues.iloc[1]  # The p-value for the slope
slope = model.params.iloc[1]  # The slope coefficient


y_pred_Fallavg = model.predict(X_const)


# Plot the data and regression line
plt.plot(X, y, label='Fall Seasonal Avg', color='peru')
plt.plot(X, model.predict(X_const), label='Regression Line', color='saddlebrown')



##Winter
# Linear Regression for each season
X = winter['Year']  # Independent variable (year)
y = winter['Seasonal_Avg']  # Dependent variable (average flow values)

# Add a constant for the intercept
X_const = sm.add_constant(X)

# Fit the model using statsmodels
model = sm.OLS(y, X_const).fit()

# Get the slope and p-value
p_value = model.pvalues.iloc[1]  # The p-value for the slope
slope = model.params.iloc[1]  # The slope coefficient



y_pred_Winteravg = model.predict(X_const)

# Plot the data and regression 
plt.plot(X, y, label='Winter Seasonal Avg', color='lightskyblue')
plt.plot(X, model.predict(X_const), label='Regression Line', color='deepskyblue')


##Spring
# Linear Regression for each season
X = spring['Year']  # Independent variable (year)
y = spring['Seasonal_Avg']  # Dependent variable (average flow values)

# Add a constant for the intercept
X_const = sm.add_constant(X)

# Fit the model using statsmodels
model = sm.OLS(y, X_const).fit()

# Get the slope and p-value
p_value = model.pvalues.iloc[1]  # The p-value for the slope
slope = model.params.iloc[1]  # The slope coefficient


y_pred_Springavg = model.predict(X_const)

# Plot the data and regression line
plt.plot(X, y, label='Spring Seasonal Avg', color='pink')
plt.plot(X, model.predict(X_const), label='Regression Line', color='palevioletred')

##Summer
# Linear Regression for each season
X = summer['Year']  # Independent variable (year)
y = summer['Seasonal_Avg']  # Dependent variable (average flow values)

# Add a constant for the intercept
X_const = sm.add_constant(X)

# Fit the model using statsmodels
model = sm.OLS(y, X_const).fit()

# Get the slope and p-value
p_value = model.pvalues.iloc[1]  # The p-value for the slope
slope = model.params.iloc[1]  # The slope coefficient


y_pred_Summeravg = model.predict(X_const)

# Plot the data and regression line
plt.plot(X, y, label='Summer Seasonal Avg', color='olive')
plt.plot(X, model.predict(X_const), label='Regression Line', color='darkolivegreen')

plt.title(f'Post-1970s Seasonal Data with Linear Regression for site {s}')
plt.xlabel('Year')
plt.ylabel('Seasonal Avg Flow')
plt.show()
```

    
     For Water Year Average of site 04256000 Slope: 0.332450694876628, P-value: 0.2449547726579893
    
     For Annual Maximum Flow of site 04256000 Slope: -0.9971031065371196, P-value: 0.8833303154303436
    
     For 7-Day Low Flow of site 04256000 Slope: -0.15290533366005052, P-value: 0.11958141835490681
    


    
![png](output_6_1.png)
    



    
![png](output_6_2.png)
    



    
![png](output_6_3.png)
    


    
     For Winter-Spring Center Volume Date of site 04256000 Slope: -0.22088812654850332, P-value: 0.23958387103006493
    


    
![png](output_6_5.png)
    



    
![png](output_6_6.png)
    


## Create Graphs for all data after 1970


```python
us_site_list_1970 = []

df_us_1970 = pd.DataFrame()  # To store US data for sites starting before 1970

for s in us_sites:

    # Get daily values (dv) for the site to check the start date
    DV = nwis.get_record(sites=s, service='dv', start='1800-01-01')
    df_temp = pd.DataFrame(DV)

    # Ensure the index is a DateTimeIndex
    df_temp.index = pd.to_datetime(df_temp.index)

    # Check if the site has data starting before 1970, and collect from df
    if df_temp.index.year.min() < 1970:
        us_site_list_1970.append(s)

    # Check if the site has data starting before 1990, and collect from df
    if df_temp.index.year.min() < 1990:
        us_site_list_1990.append(s)

# Print the results
print("US Sites with data starting before 1970:", us_site_list_1970)
```

    US Sites with data starting before 1970: ['04040500', '04066500', '04074950', '04124000', '04125460', '04221000', '04185000', '04196800', '04136000', '04213000', '04256000', '04057510', '04122500', '04085200', '04059500', '04056500', '04031000', '04045500', '04122200', '04027000']
    US Sites with data starting before 1990: ['04040500', '04066500', '04074950', '04124000', '04125460', '04221000', '04224775', '04185000', '04196800', '04197100', '04136000', '04213000', '04256000', '04057510', '04122500', '04085200', '04059500', '04056500', '04015330', '04031000', '04045500', '04024430', '04122200', '04027000']
    


```python
ca_site_list_1970 = []
ca_site_list_1990 = []

df_ca_1970 = pd.DataFrame()  # To store data for sites starting before 1970
df_ca_1990 = pd.DataFrame()  # To store data for sites starting before 1990

# Iterate through each CSV file in the directory
for filename in os.listdir(data_directory):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(data_directory, filename)
        # Load the CSV data into a DataFrame
        df_temp = pd.read_csv(file_path)
            # Ensure that the 'date' column exists and convert it to datetime if necessary
        if 'Date' not in df_temp.columns:
            print(f"The expected 'Date' column is not found for file {filename}")
            continue
        #Check if mean DV exists
        if 'Parameter/Paramètre' in df_temp.columns:
            if not df_temp['Parameter/Paramètre'].str.contains('discharge/débit', case=False, na=False).any():
                print(f"No discharge data for site {s}")
                continue
        # Access the first row of column 'ID'
        if '﻿ ID' not in df_temp.columns:
            print(f"The expected 'ID' column is not found in the data for file {filename}.")
            continue
            
        df_temp = df_temp[df_temp['Parameter/Paramètre'].str.contains('discharge', case=False, na=False)]
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        
        # Set the 'date' column as the index
        df_temp.set_index('Date', inplace=True)
        df_temp.sort_index(inplace=True)    
            
        s = df_temp.iloc[0]['﻿ ID']

        # Check if the site has data starting before 1970, and collect from df
        if df_temp.index.year.min() < 1970:
            ca_site_list_1970.append(s)
    
        # Check if the site has data starting before 1990, and collect from df
        if df_temp.index.year.min() < 1990:
            ca_site_list_1990.append(s)

# Print the results
print("Canadian Sites with data starting before 1970:", ca_site_list_1970)
```

    Canadian Sites with data starting before 1970: ['02BF001', '02BF002', '02CF007', '02EA005', '02EC002', '02EC011', '02ED003', '02ED101', '02ED102', '02FE008', '02FE009', '02FE010', '02GA010', '02GA018', '02GB007', '02GC002', '02GC010', '02GD004', '02GG002', '02GG006', '02HA006', '02HC025', '02HC030', '02HL003', '02HL004', '02HL005', '02JB013', '02JC008', '02KD002', '02LA007', '02LB007', '02LB008', '02LD005']
    Canadian Sites with data starting before 1990: ['02AB017', '02AB021', '02AC001', '02AC002', '02AD010', '02AE001', '02BA003', '02BB003', '02BF001', '02BF002', '02CB003', '02CF007', '02CF011', '02DC012', '02EA005', '02EB014', '02EC002', '02EC011', '02EC018', '02ED003', '02ED015', '02ED024', '02ED101', '02ED102', '02FA004', '02FC016', '02FE008', '02FE009', '02FE010', '02FE013', '02GA010', '02GA018', '02GA038', '02GB007', '02GC002', '02GC010', '02GD004', '02GE007', '02GG002', '02GG003', '02GG006', '02GG009', '02HA006', '02HC025', '02HC030', '02HC049', '02HD012', '02HL003', '02HL004', '02HL005', '02JB013', '02JC008', '02KD002', '02KF011', '02LA007', '02LB007', '02LB008', '02LD005', '02LG005']
    


```python
site_list=[]
latitude=[]
longitude=[]
p_values_yearly_avg=[]
slopes_yearly_avg=[]
p_values_seven_day_low_flow=[]
slopes_seven_day_low_flow=[]
p_values_max=[]
slopes_max=[]
p_values_WSCV=[]
slopes_WSCV=[]
p_values_Fallavg=[]
slopes_Fallavg=[]
p_values_Winteravg=[]
slopes_Winteravg=[]
p_values_Springavg=[]
slopes_Springavg=[]
p_values_Summeravg=[]
slopes_Summeravg=[]
```


```python
sites=us_site_list_1970

for s in sites:

    # Get daily values (dv)
    DV = nwis.get_record(sites=s, service='dv',start='1970-10-01')
    df = pd.DataFrame(DV)
    
    #Check if mean DV exists
    if '00060_Mean' not in df.columns:
        print(f"No '00060_Mean' data for site {s}")
        continue
    #Pull coordinates
    site_list.append(s)
    LocTest= nwis.get_record(sites=s, service='site')
    Lat=LocTest['dec_lat_va'].values[0]
    Long=LocTest['dec_long_va'].values[0]
    latitude.append(Lat)
    longitude.append(Long)
   
    # Convert index to datetime and apply water year function
    df.index = pd.to_datetime(df.index)
    df['water_year'] = df.index.to_series().apply(WaterYear)
    df['Season'] = df.index.to_series().apply(Season)
    # Ensure '00060_Mean' is numeric
    df['00060_Mean'] = pd.to_numeric(df['00060_Mean'], errors='coerce')

    # Find the last year
    last_year = df['water_year'].max()

    # Filter out the rows where the year is the last
    df = df[(df['water_year'] != last_year)]
    
    ##Finding Attributes
    #Mean
    # Ensure the '00060_Mean' column is numeric
    df['00060_Mean'] = pd.to_numeric(df['00060_Mean'], errors='coerce')
    for i in df.index:
        if df.loc[i, '00060_Mean'] > 0:
            df.loc[i, '00060_Mean'] = df.loc[i, '00060_Mean']  # This keeps the value unchanged
        else:
            df.loc[i, '00060_Mean'] = np.nan  # Assign NaN for values <= 0
    # Calculate the the mean of '00060_Mean' grouped by 'water_year'
    water_yearly_avg = df.groupby('water_year')['00060_Mean'].mean()
    
    # Calculate the maximum for each water year
    annual_max = df.groupby('water_year')['00060_Mean'].max()

    #7-Day Low Flow
    # Compute 7-day rolling minimum
    df['7_day_mean'] = df['00060_Mean'].rolling(window=7, min_periods=1).mean()
    # Calculate the minimum of the 7 day rolling mean for each water year
    seven_day_low_flow = df.groupby('water_year')['7_day_mean'].min()

   # Check the 'value' column for discharge data
    df['discharge'] = df['00060_Mean'] 
    seconds_per_day = 24 * 60 * 60
    df['daily_volume_cubic_ft'] = df['discharge'] * seconds_per_day
    df['cumulative_volume'] = df.groupby('water_year')['daily_volume_cubic_ft'].cumsum()
    df['date_column'] = df.index  # Ensure the index is datetime
    

    ## Calculations for center volume / center mass
    total_volume_each_wateryear =df.groupby('water_year')['daily_volume_cubic_ft'].sum()
    half_volume_each_wateryear = total_volume_each_wateryear / 2
    half_volume_dates = {}
    # Iterate over each unique water year
    for water_year in df['water_year'].unique():
        # Get the half volume for the current water year
        half_volume = half_volume_each_wateryear[water_year]
        # Get the subset of data for the current water year
        year_data = df[df['water_year'] == water_year]
        # Find the index where cumulative volume reaches or exceeds half of total volume
        half_volume_index = year_data[year_data['cumulative_volume'] >= half_volume].index
        if not half_volume_index.empty:
            # Retrieve the actual date from the DataFrame using the first index
            first_index = half_volume_index[0]  # Get the first index
            half_volume_dates[water_year] = df.at[first_index, 'date_column']  # Use 'at' to get the value
     # Convert dates to just month and day format (MM-DD)
    # Convert dates to just month and day format for display, but keep the original datetime for plotting
    half_volume_dates_values = list(half_volume_dates.values())  # Extract the dates first
    formatted_dates = [date.timetuple().tm_yday if isinstance(date, pd.Timestamp) else None for date in half_volume_dates_values]
    # Create plot for center of volume date
    water_years = list(half_volume_dates.keys())
    half_volume_dates = list(half_volume_dates.values())
    # Convert dates to day of year for sorting and plotting
    formatted_dates = [date.timetuple().tm_yday for date in half_volume_dates]

    
    #Calculates linear regression lines
    # Linear Regression Yearly Avg
    X = water_yearly_avg.index.values.reshape(-1, 1)  # Independent variable (water years)
    y = water_yearly_avg.values  # Dependent variable (average flow values)
    # Add a constant for the intercept
    X_const = sm.add_constant(X)
    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()
    # Predict values
    y_pred_water_yearly_avg = model.predict(X_const)
    # Get the slope and p-value
    slope_water_yearly_avg = model.params[1]  # The slope coefficient
    p_value_water_yearly_avg = model.pvalues[1]  # The p-value for the slope
    # Print the results
    print('\n',f'For Water Year Average of site {s}', f'Slope: {slope_water_yearly_avg}, P-value: {p_value_water_yearly_avg}')
    #Add values to lists
    p_values_yearly_avg.append(p_value_water_yearly_avg)
    slopes_yearly_avg.append(slope_water_yearly_avg)
    
    # Linear Regression Annual Max
    X = annual_max.index.values.reshape(-1, 1)  # Independent variable (water years)
    y = annual_max.values  # Dependent variable (average flow values)
    # Add a constant for the intercept
    X_const = sm.add_constant(X)
    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()
    # Predict values
    y_pred_annual_max = model.predict(X_const)
    # Get the slope and p-value
    slope_water_annual_max = model.params[1]  # The slope coefficient
    p_value_water_annual_max = model.pvalues[1]  # The p-value for the slope
    # Print the results
    print('\n',f'For Annual Maximum Flow of site {s}', f'Slope: {slope_water_annual_max}, P-value: {p_value_water_annual_max}')
    #Add values to lists
    p_values_max.append(p_value_water_annual_max)
    slopes_max.append(slope_water_annual_max)

    # Linear Regression 7-Day Low Flow
    X = seven_day_low_flow.index.values.reshape(-1, 1)  # Independent variable (water years)
    y = seven_day_low_flow.values  # Dependent variable (average flow values)
    # Add a constant for the intercept
    X_const = sm.add_constant(X)
    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()
    # Predict values
    y_pred_seven_day_low_flow = model.predict(X_const)
    # Get the slope and p-value
    slope_seven_day_low_flow = model.params[1]  # The slope coefficient
    p_value_seven_day_low_flow = model.pvalues[1]  # The p-value for the slope
    print('\n',f'For 7-Day Low Flow of site {s}', f'Slope: {slope_seven_day_low_flow}, P-value: {p_value_seven_day_low_flow}')
    #Add values to lists
    p_values_seven_day_low_flow.append(p_value_seven_day_low_flow)
    slopes_seven_day_low_flow.append(slope_seven_day_low_flow)
    


    # Create a plot for the information with linear regression lines
    plt.title(f'Post-1970s Data for Site {s}', fontsize=16)
    plt.plot(water_yearly_avg.index, water_yearly_avg.values, label='Daily Average Flow', color='blue', marker='o')
    plt.plot(water_yearly_avg.index, y_pred_water_yearly_avg, label='Daily Avg Linear Regression Line', color='red')
    plt.plot(annual_max.index, annual_max.values, label='Yearly Max Flow', color='orange', marker='o')
    plt.plot(annual_max.index, y_pred_annual_max, label='Yearly Max Linear Regression Line', color='red')
    plt.plot(seven_day_low_flow.index, seven_day_low_flow.values, label='7 Day Low Flow', color='pink', marker='o')
    plt.plot(seven_day_low_flow.index, y_pred_seven_day_low_flow, label='7-Day Low Flow Linear Regression Line', color='red')
    plt.ylabel('Flow (units)')
    plt.legend()

    plt.show()



    
    #Linear Regression for Center Mass Volume Date
    X = water_years  # Independent variable (water years)
    y = formatted_dates  # Dependent variable (average flow values)
    # Add a constant for the intercept
    X_const = sm.add_constant(X)
    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()
    # Predict values
    y_Winter_Spring_Center_Volume_Date = model.predict(X_const)
    # Get the slope and p-value
    slope_WSCV = model.params[1]  # The slope coefficient
    p_value_WSCV = model.pvalues[1]  # The p-value for the slope
    print('\n',f'For Winter-Spring Center Volume Date of site {s}', f'Slope: {slope_WSCV}, P-value: {p_value_WSCV}')
    #Add values to list
    p_values_WSCV.append(p_value_WSCV)
    slopes_WSCV.append(slope_WSCV)



# Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(water_years, formatted_dates, color='blue', marker='o')
    plt.plot(water_years, y_Winter_Spring_Center_Volume_Date, label='Center Volume Date Regression Line', color='red')
    plt.title(f'Post-1970s Winter-Spring Center Volume Date for Site {s}', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Date')
    plt.xticks(rotation=45)
    # Format the y-axis labels to show MM-DD
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    plt.show()

        # Calculate seasonal averages
    water_seasonly_avg = df.groupby([df['Season'], df.index.year])['00060_Mean'].mean().reset_index()
    water_seasonly_avg.columns = ['Season', 'Year', 'Seasonal_Avg']
    # Ensure the Year column is converted to integers
    water_seasonly_avg['Year'] = water_seasonly_avg['Year'].astype(int)

    # Split the data into separate DataFrames for each season
    fall = water_seasonly_avg[water_seasonly_avg['Season'] == 'Fall']
    winter = water_seasonly_avg[water_seasonly_avg['Season'] == 'Winter']
    spring = water_seasonly_avg[water_seasonly_avg['Season'] == 'Spring']
    summer = water_seasonly_avg[water_seasonly_avg['Season'] == 'Summer']
    

    # Linear Regression for each season
##Fall

    X = fall['Year']  # Independent variable (year)
    y = fall['Seasonal_Avg']  # Dependent variable (average flow values)

    # Add a constant for the intercept
    X_const = sm.add_constant(X)

    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()

    # Get the slope and p-value
    p_value = model.pvalues.iloc[1]  # The p-value for the slope
    slope = model.params.iloc[1]  # The slope coefficient

    p_values_Fallavg.append(p_value)
    slopes_Fallavg.append(slope)
    y_pred_Fallavg = model.predict(X_const)


    # Plot the data and regression line
    plt.plot(X, y, label='Fall Seasonal Avg', color='peru')
    plt.plot(X, model.predict(X_const), label='Regression Line', color='saddlebrown')



##Winter
    # Linear Regression for each season
    X = winter['Year']  # Independent variable (year)
    y = winter['Seasonal_Avg']  # Dependent variable (average flow values)

    # Add a constant for the intercept
    X_const = sm.add_constant(X)

    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()

    # Get the slope and p-value
    p_value = model.pvalues.iloc[1]  # The p-value for the slope
    slope = model.params.iloc[1]  # The slope coefficient


    p_values_Winteravg.append(p_value)
    slopes_Winteravg.append(slope)
    y_pred_Winteravg = model.predict(X_const)

# Plot the data and regression 
    plt.plot(X, y, label='Winter Seasonal Avg', color='lightskyblue')
    plt.plot(X, model.predict(X_const), label='Regression Line', color='deepskyblue')
    

##Spring
    # Linear Regression for each season
    X = spring['Year']  # Independent variable (year)
    y = spring['Seasonal_Avg']  # Dependent variable (average flow values)

    # Add a constant for the intercept
    X_const = sm.add_constant(X)

    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()

    # Get the slope and p-value
    p_value = model.pvalues.iloc[1]  # The p-value for the slope
    slope = model.params.iloc[1]  # The slope coefficient

    p_values_Springavg.append(p_value)
    slopes_Springavg.append(slope)
    y_pred_Springavg = model.predict(X_const)

    # Plot the data and regression line
    plt.plot(X, y, label='Spring Seasonal Avg', color='pink')
    plt.plot(X, model.predict(X_const), label='Regression Line', color='palevioletred')

##Summer
    # Linear Regression for each season
    X = summer['Year']  # Independent variable (year)
    y = summer['Seasonal_Avg']  # Dependent variable (average flow values)

    # Add a constant for the intercept
    X_const = sm.add_constant(X)

    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()

    # Get the slope and p-value
    p_value = model.pvalues.iloc[1]  # The p-value for the slope
    slope = model.params.iloc[1]  # The slope coefficient

    p_values_Summeravg.append(p_value)
    slopes_Summeravg.append(slope)
    y_pred_Summeravg = model.predict(X_const)

    # Plot the data and regression line
    plt.plot(X, y, label='Summer Seasonal Avg', color='olive')
    plt.plot(X, model.predict(X_const), label='Regression Line', color='darkolivegreen')
  
    plt.title(f'Post-1970s Seasonal Data with Linear Regression for site {s}')
    plt.xlabel('Year')
    plt.ylabel('Seasonal Avg Flow')
    plt.show()
```

    
     For Water Year Average of site 04040500 Slope: -0.6495244978687618, P-value: 0.12383717098743277
    
     For Annual Maximum Flow of site 04040500 Slope: -3.9356203544882775, P-value: 0.5997939696599401
    
     For 7-Day Low Flow of site 04040500 Slope: -0.19021519780010437, P-value: 0.05978340611004086
    


    
![png](output_11_1.png)
    


    
     For Winter-Spring Center Volume Date of site 04040500 Slope: -0.13821231179721802, P-value: 0.16229495750194137
    


    
![png](output_11_3.png)
    



    
![png](output_11_4.png)
    


    
     For Water Year Average of site 04066500 Slope: 3.392505293463914, P-value: 0.02414495052175088
    
     For Annual Maximum Flow of site 04066500 Slope: 9.539230769230793, P-value: 0.1525239898229185
    
     For 7-Day Low Flow of site 04066500 Slope: 1.6746923076922946, P-value: 0.023724819722037913
    


    
![png](output_11_6.png)
    


    
     For Winter-Spring Center Volume Date of site 04066500 Slope: -1.1338461538461884, P-value: 0.1314569160206079
    


    
![png](output_11_8.png)
    



    
![png](output_11_9.png)
    


    
     For Water Year Average of site 04074950 Slope: -0.37347511400445976, P-value: 0.6642951992789111
    
     For Annual Maximum Flow of site 04074950 Slope: 0.5594816085382242, P-value: 0.8773802154851308
    
     For 7-Day Low Flow of site 04074950 Slope: -0.4976666938931128, P-value: 0.2598612252291923
    


    
![png](output_11_11.png)
    


    
     For Winter-Spring Center Volume Date of site 04074950 Slope: -0.12502382313703206, P-value: 0.5940363166830841
    


    
![png](output_11_13.png)
    



    
![png](output_11_14.png)
    


    
     For Water Year Average of site 04124000 Slope: 0.8383840233106996, P-value: 0.27052590964569345
    
     For Annual Maximum Flow of site 04124000 Slope: -4.557270821421739, P-value: 0.23832219647936323
    
     For 7-Day Low Flow of site 04124000 Slope: 0.7961011734596648, P-value: 0.0957860756321315
    


    
![png](output_11_16.png)
    


    
     For Winter-Spring Center Volume Date of site 04124000 Slope: 0.23777396607585355, P-value: 0.343113933161338
    


    
![png](output_11_18.png)
    



    
![png](output_11_19.png)
    


    
     For Water Year Average of site 04125460 Slope: -0.11887061899678741, P-value: 0.6450042687033708
    
     For Annual Maximum Flow of site 04125460 Slope: -1.6155867949728968, P-value: 0.6830319829541769
    
     For 7-Day Low Flow of site 04125460 Slope: -0.3853231542951018, P-value: 0.003171198378957456
    


    
![png](output_11_21.png)
    


    
     For Winter-Spring Center Volume Date of site 04125460 Slope: -0.13469572959830062, P-value: 0.10282667585580543
    


    
![png](output_11_23.png)
    



    
![png](output_11_24.png)
    


    
     For Water Year Average of site 04221000 Slope: 0.30548260155633433, P-value: 0.7264368872809381
    
     For Annual Maximum Flow of site 04221000 Slope: -8.556731836421061, P-value: 0.6701425230002895
    
     For 7-Day Low Flow of site 04221000 Slope: -0.0974552084984939, P-value: 0.3719458865823063
    


    
![png](output_11_26.png)
    


    
     For Winter-Spring Center Volume Date of site 04221000 Slope: -0.30342354648680564, P-value: 0.22367908783298177
    


    
![png](output_11_28.png)
    



    
![png](output_11_29.png)
    


    
     For Water Year Average of site 04185000 Slope: 0.5488308522788837, P-value: 0.5521282383610908
    
     For Annual Maximum Flow of site 04185000 Slope: 2.087669144272877, P-value: 0.8737130552153869
    
     For 7-Day Low Flow of site 04185000 Slope: 0.03515995534863427, P-value: 0.6611557899704701
    


    
![png](output_11_31.png)
    


    
     For Winter-Spring Center Volume Date of site 04185000 Slope: 0.34671240708976425, P-value: 0.5114337123033066
    


    
![png](output_11_33.png)
    



    
![png](output_11_34.png)
    


    
     For Water Year Average of site 04196800 Slope: 1.4919359242891381, P-value: 0.03868635575373688
    
     For Annual Maximum Flow of site 04196800 Slope: 24.562988374309082, P-value: 0.03972074062799647
    
     For 7-Day Low Flow of site 04196800 Slope: 0.018063546516372625, P-value: 0.26385549317746243
    


    
![png](output_11_36.png)
    


    
     For Winter-Spring Center Volume Date of site 04196800 Slope: 0.19916142557651834, P-value: 0.7233089932220422
    


    
![png](output_11_38.png)
    



    
![png](output_11_39.png)
    


    
     For Water Year Average of site 04136000 Slope: 6.1423820991878095, P-value: 0.007622862244935213
    
     For Annual Maximum Flow of site 04136000 Slope: 5.27586206896556, P-value: 0.6788477456739044
    
     For 7-Day Low Flow of site 04136000 Slope: 3.619000703729564, P-value: 0.0037084255118428277
    


    
![png](output_11_41.png)
    


    
     For Winter-Spring Center Volume Date of site 04136000 Slope: -0.14039408866996117, P-value: 0.4872287027148706
    


    
![png](output_11_43.png)
    



    
![png](output_11_44.png)
    


    
     For Water Year Average of site 04213000 Slope: -0.1950293230436192, P-value: 0.7244648076157153
    
     For Annual Maximum Flow of site 04213000 Slope: -29.77320373546809, P-value: 0.07156252548104709
    
     For 7-Day Low Flow of site 04213000 Slope: -0.015040050096652692, P-value: 0.8402462827691967
    


    
![png](output_11_46.png)
    


    
     For Winter-Spring Center Volume Date of site 04213000 Slope: 0.00266819134743557, P-value: 0.9892734050415397
    


    
![png](output_11_48.png)
    



    
![png](output_11_49.png)
    


    
     For Water Year Average of site 04256000 Slope: 0.332450694876628, P-value: 0.2449547726579893
    
     For Annual Maximum Flow of site 04256000 Slope: -0.9971031065371196, P-value: 0.8833303154303436
    
     For 7-Day Low Flow of site 04256000 Slope: -0.15290533366005052, P-value: 0.11958141835490681
    


    
![png](output_11_51.png)
    


    
     For Winter-Spring Center Volume Date of site 04256000 Slope: -0.22088812654850332, P-value: 0.23958387103006493
    


    
![png](output_11_53.png)
    



    
![png](output_11_54.png)
    


    
     For Water Year Average of site 04057510 Slope: -0.014110270291149862, P-value: 0.9749479208485068
    
     For Annual Maximum Flow of site 04057510 Slope: -2.330398322851135, P-value: 0.4579787783481851
    
     For 7-Day Low Flow of site 04057510 Slope: -0.16153502681804652, P-value: 0.24889548326615335
    


    
![png](output_11_56.png)
    


    
     For Winter-Spring Center Volume Date of site 04057510 Slope: 0.0014865637507144314, P-value: 0.9913814351765703
    


    
![png](output_11_58.png)
    



    
![png](output_11_59.png)
    


    
     For Water Year Average of site 04122500 Slope: -0.0643661571828783, P-value: 0.9498075162150207
    
     For Annual Maximum Flow of site 04122500 Slope: -3.127501429388245, P-value: 0.6568137592364451
    
     For 7-Day Low Flow of site 04122500 Slope: 0.6194233439516419, P-value: 0.17174266434263125
    


    
![png](output_11_61.png)
    


    
     For Winter-Spring Center Volume Date of site 04122500 Slope: 0.05008576329331088, P-value: 0.6561043596709073
    


    
![png](output_11_63.png)
    



    
![png](output_11_64.png)
    


    
     For Water Year Average of site 04085200 Slope: 0.053066589024565874, P-value: 0.8751793962827079
    
     For Annual Maximum Flow of site 04085200 Slope: -11.845932331627916, P-value: 0.23109250122875247
    
     For 7-Day Low Flow of site 04085200 Slope: -0.012299762580814402, P-value: 0.796028052202286
    


    
![png](output_11_66.png)
    


    
     For Winter-Spring Center Volume Date of site 04085200 Slope: 0.31193848225151455, P-value: 0.14760240224766258
    


    
![png](output_11_68.png)
    



    
![png](output_11_69.png)
    


    
     For Water Year Average of site 04059500 Slope: -0.8141899019687258, P-value: 0.46387406981853774
    
     For Annual Maximum Flow of site 04059500 Slope: -10.064417762530923, P-value: 0.2137042219231582
    
     For 7-Day Low Flow of site 04059500 Slope: -0.07961049851616032, P-value: 0.7886720908561943
    


    
![png](output_11_71.png)
    


    
     For Winter-Spring Center Volume Date of site 04059500 Slope: -0.021879169048980865, P-value: 0.8760571720147775
    


    
![png](output_11_73.png)
    



    
![png](output_11_74.png)
    


    
     For Water Year Average of site 04056500 Slope: -1.0930405284863545, P-value: 0.7055697234979756
    
     For Annual Maximum Flow of site 04056500 Slope: -38.443300933866894, P-value: 0.056285809826271016
    
     For 7-Day Low Flow of site 04056500 Slope: -2.5784747747011956, P-value: 0.010992552668734395
    


    
![png](output_11_76.png)
    


    
     For Winter-Spring Center Volume Date of site 04056500 Slope: -0.27996950638460183, P-value: 0.015880952156205785
    


    
![png](output_11_78.png)
    



    
![png](output_11_79.png)
    


    
     For Water Year Average of site 04031000 Slope: 0.4247504312366934, P-value: 0.4732654936369213
    
     For Annual Maximum Flow of site 04031000 Slope: 8.866735271417467, P-value: 0.645021544881664
    
     For 7-Day Low Flow of site 04031000 Slope: 0.004577652002399624, P-value: 0.9469831293229796
    


    
![png](output_11_81.png)
    


    
     For Winter-Spring Center Volume Date of site 04031000 Slope: -0.11379813052053808, P-value: 0.31343078165233146
    


    
![png](output_11_83.png)
    



    
![png](output_11_84.png)
    


    
     For Water Year Average of site 04045500 Slope: -0.5659779909380378, P-value: 0.7085699313524481
    
     For Annual Maximum Flow of site 04045500 Slope: -11.894797026872459, P-value: 0.2438792921104055
    
     For 7-Day Low Flow of site 04045500 Slope: -0.8521821993520049, P-value: 0.08668418646206749
    


    
![png](output_11_86.png)
    


    
     For Winter-Spring Center Volume Date of site 04045500 Slope: -0.2667047836859166, P-value: 0.023359854559996996
    


    
![png](output_11_88.png)
    



    
![png](output_11_89.png)
    


    
     For Water Year Average of site 04122200 Slope: -0.13646135933896741, P-value: 0.842752390069506
    
     For Annual Maximum Flow of site 04122200 Slope: -7.344577854011856, P-value: 0.34089208120176495
    
     For 7-Day Low Flow of site 04122200 Slope: -0.029252089629451042, P-value: 0.912249311009854
    


    
![png](output_11_91.png)
    


    
     For Winter-Spring Center Volume Date of site 04122200 Slope: 0.06113969887554779, P-value: 0.5930209181482895
    


    
![png](output_11_93.png)
    



    
![png](output_11_94.png)
    


    
     For Water Year Average of site 04027000 Slope: 0.6063473900220693, P-value: 0.7087825096856437
    
     For Annual Maximum Flow of site 04027000 Slope: 46.93767867352799, P-value: 0.26986573737787944
    
     For 7-Day Low Flow of site 04027000 Slope: -0.07063519289934478, P-value: 0.8101956890409127
    


    
![png](output_11_96.png)
    


    
     For Winter-Spring Center Volume Date of site 04027000 Slope: 0.02854964741757139, P-value: 0.8299193169658627
    


    
![png](output_11_98.png)
    



    
![png](output_11_99.png)
    


    
     For Water Year Average of site 04040500 Slope: -0.6495244978687618, P-value: 0.12383717098743277
    
     For Annual Maximum Flow of site 04040500 Slope: -3.9356203544882775, P-value: 0.5997939696599401
    
     For 7-Day Low Flow of site 04040500 Slope: -0.19021519780010437, P-value: 0.05978340611004086
    


    
![png](output_11_101.png)
    


    
     For Winter-Spring Center Volume Date of site 04040500 Slope: -0.13821231179721802, P-value: 0.16229495750194137
    


    
![png](output_11_103.png)
    



    
![png](output_11_104.png)
    


    
     For Water Year Average of site 04066500 Slope: 3.392505293463914, P-value: 0.02414495052175088
    
     For Annual Maximum Flow of site 04066500 Slope: 9.539230769230793, P-value: 0.1525239898229185
    
     For 7-Day Low Flow of site 04066500 Slope: 1.6746923076922946, P-value: 0.023724819722037913
    


    
![png](output_11_106.png)
    


    
     For Winter-Spring Center Volume Date of site 04066500 Slope: -1.1338461538461884, P-value: 0.1314569160206079
    


    
![png](output_11_108.png)
    



    
![png](output_11_109.png)
    


    
     For Water Year Average of site 04074950 Slope: -0.37347511400445976, P-value: 0.6642951992789111
    
     For Annual Maximum Flow of site 04074950 Slope: 0.5594816085382242, P-value: 0.8773802154851308
    
     For 7-Day Low Flow of site 04074950 Slope: -0.4976666938931128, P-value: 0.2598612252291923
    


    
![png](output_11_111.png)
    


    
     For Winter-Spring Center Volume Date of site 04074950 Slope: -0.12502382313703206, P-value: 0.5940363166830841
    


    
![png](output_11_113.png)
    



    
![png](output_11_114.png)
    


    
     For Water Year Average of site 04124000 Slope: 0.8383840233106996, P-value: 0.27052590964569345
    
     For Annual Maximum Flow of site 04124000 Slope: -4.557270821421739, P-value: 0.23832219647936323
    
     For 7-Day Low Flow of site 04124000 Slope: 0.7961011734596648, P-value: 0.0957860756321315
    


    
![png](output_11_116.png)
    


    
     For Winter-Spring Center Volume Date of site 04124000 Slope: 0.23777396607585355, P-value: 0.343113933161338
    


    
![png](output_11_118.png)
    



    
![png](output_11_119.png)
    


    
     For Water Year Average of site 04125460 Slope: -0.11887061899678741, P-value: 0.6450042687033708
    
     For Annual Maximum Flow of site 04125460 Slope: -1.6155867949728968, P-value: 0.6830319829541769
    
     For 7-Day Low Flow of site 04125460 Slope: -0.3853231542951018, P-value: 0.003171198378957456
    


    
![png](output_11_121.png)
    


    
     For Winter-Spring Center Volume Date of site 04125460 Slope: -0.13469572959830062, P-value: 0.10282667585580543
    


    
![png](output_11_123.png)
    



    
![png](output_11_124.png)
    


    
     For Water Year Average of site 04221000 Slope: 0.30548260155633433, P-value: 0.7264368872809381
    
     For Annual Maximum Flow of site 04221000 Slope: -8.556731836421061, P-value: 0.6701425230002895
    
     For 7-Day Low Flow of site 04221000 Slope: -0.0974552084984939, P-value: 0.3719458865823063
    


    
![png](output_11_126.png)
    


    
     For Winter-Spring Center Volume Date of site 04221000 Slope: -0.30342354648680564, P-value: 0.22367908783298177
    


    
![png](output_11_128.png)
    



    
![png](output_11_129.png)
    


    
     For Water Year Average of site 04185000 Slope: 0.5488308522788837, P-value: 0.5521282383610908
    
     For Annual Maximum Flow of site 04185000 Slope: 2.087669144272877, P-value: 0.8737130552153869
    
     For 7-Day Low Flow of site 04185000 Slope: 0.03515995534863427, P-value: 0.6611557899704701
    


    
![png](output_11_131.png)
    


    
     For Winter-Spring Center Volume Date of site 04185000 Slope: 0.34671240708976425, P-value: 0.5114337123033066
    


    
![png](output_11_133.png)
    



    
![png](output_11_134.png)
    


    
     For Water Year Average of site 04196800 Slope: 1.4919359242891381, P-value: 0.03868635575373688
    
     For Annual Maximum Flow of site 04196800 Slope: 24.562988374309082, P-value: 0.03972074062799647
    
     For 7-Day Low Flow of site 04196800 Slope: 0.018063546516372625, P-value: 0.26385549317746243
    


    
![png](output_11_136.png)
    


    
     For Winter-Spring Center Volume Date of site 04196800 Slope: 0.19916142557651834, P-value: 0.7233089932220422
    


    
![png](output_11_138.png)
    



    
![png](output_11_139.png)
    


    
     For Water Year Average of site 04136000 Slope: 6.1423820991878095, P-value: 0.007622862244935213
    
     For Annual Maximum Flow of site 04136000 Slope: 5.27586206896556, P-value: 0.6788477456739044
    
     For 7-Day Low Flow of site 04136000 Slope: 3.619000703729564, P-value: 0.0037084255118428277
    


    
![png](output_11_141.png)
    


    
     For Winter-Spring Center Volume Date of site 04136000 Slope: -0.14039408866996117, P-value: 0.4872287027148706
    


    
![png](output_11_143.png)
    



    
![png](output_11_144.png)
    


    
     For Water Year Average of site 04213000 Slope: -0.1950293230436192, P-value: 0.7244648076157153
    
     For Annual Maximum Flow of site 04213000 Slope: -29.77320373546809, P-value: 0.07156252548104709
    
     For 7-Day Low Flow of site 04213000 Slope: -0.015040050096652692, P-value: 0.8402462827691967
    


    
![png](output_11_146.png)
    


    
     For Winter-Spring Center Volume Date of site 04213000 Slope: 0.00266819134743557, P-value: 0.9892734050415397
    


    
![png](output_11_148.png)
    



    
![png](output_11_149.png)
    


    
     For Water Year Average of site 04256000 Slope: 0.332450694876628, P-value: 0.2449547726579893
    
     For Annual Maximum Flow of site 04256000 Slope: -0.9971031065371196, P-value: 0.8833303154303436
    
     For 7-Day Low Flow of site 04256000 Slope: -0.15290533366005052, P-value: 0.11958141835490681
    


    
![png](output_11_151.png)
    


    
     For Winter-Spring Center Volume Date of site 04256000 Slope: -0.22088812654850332, P-value: 0.23958387103006493
    


    
![png](output_11_153.png)
    



    
![png](output_11_154.png)
    


    
     For Water Year Average of site 04057510 Slope: -0.014110270291149862, P-value: 0.9749479208485068
    
     For Annual Maximum Flow of site 04057510 Slope: -2.330398322851135, P-value: 0.4579787783481851
    
     For 7-Day Low Flow of site 04057510 Slope: -0.16153502681804652, P-value: 0.24889548326615335
    


    
![png](output_11_156.png)
    


    
     For Winter-Spring Center Volume Date of site 04057510 Slope: 0.0014865637507144314, P-value: 0.9913814351765703
    


    
![png](output_11_158.png)
    



    
![png](output_11_159.png)
    


    
     For Water Year Average of site 04122500 Slope: -0.0643661571828783, P-value: 0.9498075162150207
    
     For Annual Maximum Flow of site 04122500 Slope: -3.127501429388245, P-value: 0.6568137592364451
    
     For 7-Day Low Flow of site 04122500 Slope: 0.6194233439516419, P-value: 0.17174266434263125
    


    
![png](output_11_161.png)
    


    
     For Winter-Spring Center Volume Date of site 04122500 Slope: 0.05008576329331088, P-value: 0.6561043596709073
    


    
![png](output_11_163.png)
    



    
![png](output_11_164.png)
    


    
     For Water Year Average of site 04085200 Slope: 0.053066589024565874, P-value: 0.8751793962827079
    
     For Annual Maximum Flow of site 04085200 Slope: -11.845932331627916, P-value: 0.23109250122875247
    
     For 7-Day Low Flow of site 04085200 Slope: -0.012299762580814402, P-value: 0.796028052202286
    


    
![png](output_11_166.png)
    


    
     For Winter-Spring Center Volume Date of site 04085200 Slope: 0.31193848225151455, P-value: 0.14760240224766258
    


    
![png](output_11_168.png)
    



    
![png](output_11_169.png)
    


    
     For Water Year Average of site 04059500 Slope: -0.8141899019687258, P-value: 0.46387406981853774
    
     For Annual Maximum Flow of site 04059500 Slope: -10.064417762530923, P-value: 0.2137042219231582
    
     For 7-Day Low Flow of site 04059500 Slope: -0.07961049851616032, P-value: 0.7886720908561943
    


    
![png](output_11_171.png)
    


    
     For Winter-Spring Center Volume Date of site 04059500 Slope: -0.021879169048980865, P-value: 0.8760571720147775
    


    
![png](output_11_173.png)
    



    
![png](output_11_174.png)
    


    
     For Water Year Average of site 04056500 Slope: -1.0930405284863545, P-value: 0.7055697234979756
    
     For Annual Maximum Flow of site 04056500 Slope: -38.443300933866894, P-value: 0.056285809826271016
    
     For 7-Day Low Flow of site 04056500 Slope: -2.5784747747011956, P-value: 0.010992552668734395
    


    
![png](output_11_176.png)
    


    
     For Winter-Spring Center Volume Date of site 04056500 Slope: -0.27996950638460183, P-value: 0.015880952156205785
    


    
![png](output_11_178.png)
    



    
![png](output_11_179.png)
    


    
     For Water Year Average of site 04031000 Slope: 0.4247504312366934, P-value: 0.4732654936369213
    
     For Annual Maximum Flow of site 04031000 Slope: 8.866735271417467, P-value: 0.645021544881664
    
     For 7-Day Low Flow of site 04031000 Slope: 0.004577652002399624, P-value: 0.9469831293229796
    


    
![png](output_11_181.png)
    


    
     For Winter-Spring Center Volume Date of site 04031000 Slope: -0.11379813052053808, P-value: 0.31343078165233146
    


    
![png](output_11_183.png)
    



    
![png](output_11_184.png)
    


    
     For Water Year Average of site 04045500 Slope: -0.5659779909380378, P-value: 0.7085699313524481
    
     For Annual Maximum Flow of site 04045500 Slope: -11.894797026872459, P-value: 0.2438792921104055
    
     For 7-Day Low Flow of site 04045500 Slope: -0.8521821993520049, P-value: 0.08668418646206749
    


    
![png](output_11_186.png)
    


    
     For Winter-Spring Center Volume Date of site 04045500 Slope: -0.2667047836859166, P-value: 0.023359854559996996
    


    
![png](output_11_188.png)
    



    
![png](output_11_189.png)
    


    
     For Water Year Average of site 04122200 Slope: -0.13646135933896741, P-value: 0.842752390069506
    
     For Annual Maximum Flow of site 04122200 Slope: -7.344577854011856, P-value: 0.34089208120176495
    
     For 7-Day Low Flow of site 04122200 Slope: -0.029252089629451042, P-value: 0.912249311009854
    


    
![png](output_11_191.png)
    


    
     For Winter-Spring Center Volume Date of site 04122200 Slope: 0.06113969887554779, P-value: 0.5930209181482895
    


    
![png](output_11_193.png)
    



    
![png](output_11_194.png)
    


    
     For Water Year Average of site 04027000 Slope: 0.6063473900220693, P-value: 0.7087825096856437
    
     For Annual Maximum Flow of site 04027000 Slope: 46.93767867352799, P-value: 0.26986573737787944
    
     For 7-Day Low Flow of site 04027000 Slope: -0.07063519289934478, P-value: 0.8101956890409127
    


    
![png](output_11_196.png)
    


    
     For Winter-Spring Center Volume Date of site 04027000 Slope: 0.02854964741757139, P-value: 0.8299193169658627
    


    
![png](output_11_198.png)
    



    
![png](output_11_199.png)
    


    
     For Water Year Average of site 04040500 Slope: -0.6495244978687618, P-value: 0.12383717098743277
    
     For Annual Maximum Flow of site 04040500 Slope: -3.9356203544882775, P-value: 0.5997939696599401
    
     For 7-Day Low Flow of site 04040500 Slope: -0.19021519780010437, P-value: 0.05978340611004086
    


    
![png](output_11_201.png)
    


    
     For Winter-Spring Center Volume Date of site 04040500 Slope: -0.13821231179721802, P-value: 0.16229495750194137
    


    
![png](output_11_203.png)
    



    
![png](output_11_204.png)
    


    
     For Water Year Average of site 04066500 Slope: 3.392505293463914, P-value: 0.02414495052175088
    
     For Annual Maximum Flow of site 04066500 Slope: 9.539230769230793, P-value: 0.1525239898229185
    
     For 7-Day Low Flow of site 04066500 Slope: 1.6746923076922946, P-value: 0.023724819722037913
    


    
![png](output_11_206.png)
    


    
     For Winter-Spring Center Volume Date of site 04066500 Slope: -1.1338461538461884, P-value: 0.1314569160206079
    


    
![png](output_11_208.png)
    



    
![png](output_11_209.png)
    


    
     For Water Year Average of site 04074950 Slope: -0.37347511400445976, P-value: 0.6642951992789111
    
     For Annual Maximum Flow of site 04074950 Slope: 0.5594816085382242, P-value: 0.8773802154851308
    
     For 7-Day Low Flow of site 04074950 Slope: -0.4976666938931128, P-value: 0.2598612252291923
    


    
![png](output_11_211.png)
    


    
     For Winter-Spring Center Volume Date of site 04074950 Slope: -0.12502382313703206, P-value: 0.5940363166830841
    


    
![png](output_11_213.png)
    



    
![png](output_11_214.png)
    


    
     For Water Year Average of site 04124000 Slope: 0.8383840233106996, P-value: 0.27052590964569345
    
     For Annual Maximum Flow of site 04124000 Slope: -4.557270821421739, P-value: 0.23832219647936323
    
     For 7-Day Low Flow of site 04124000 Slope: 0.7961011734596648, P-value: 0.0957860756321315
    


    
![png](output_11_216.png)
    


    
     For Winter-Spring Center Volume Date of site 04124000 Slope: 0.23777396607585355, P-value: 0.343113933161338
    


    
![png](output_11_218.png)
    



    
![png](output_11_219.png)
    


    
     For Water Year Average of site 04125460 Slope: -0.11887061899678741, P-value: 0.6450042687033708
    
     For Annual Maximum Flow of site 04125460 Slope: -1.6155867949728968, P-value: 0.6830319829541769
    
     For 7-Day Low Flow of site 04125460 Slope: -0.3853231542951018, P-value: 0.003171198378957456
    


    
![png](output_11_221.png)
    


    
     For Winter-Spring Center Volume Date of site 04125460 Slope: -0.13469572959830062, P-value: 0.10282667585580543
    


    
![png](output_11_223.png)
    



    
![png](output_11_224.png)
    


    
     For Water Year Average of site 04221000 Slope: 0.30548260155633433, P-value: 0.7264368872809381
    
     For Annual Maximum Flow of site 04221000 Slope: -8.556731836421061, P-value: 0.6701425230002895
    
     For 7-Day Low Flow of site 04221000 Slope: -0.0974552084984939, P-value: 0.3719458865823063
    


    
![png](output_11_226.png)
    


    
     For Winter-Spring Center Volume Date of site 04221000 Slope: -0.30342354648680564, P-value: 0.22367908783298177
    


    
![png](output_11_228.png)
    



    
![png](output_11_229.png)
    


    
     For Water Year Average of site 04185000 Slope: 0.5488308522788837, P-value: 0.5521282383610908
    
     For Annual Maximum Flow of site 04185000 Slope: 2.087669144272877, P-value: 0.8737130552153869
    
     For 7-Day Low Flow of site 04185000 Slope: 0.03515995534863427, P-value: 0.6611557899704701
    


    
![png](output_11_231.png)
    


    
     For Winter-Spring Center Volume Date of site 04185000 Slope: 0.34671240708976425, P-value: 0.5114337123033066
    


    
![png](output_11_233.png)
    



    
![png](output_11_234.png)
    


    
     For Water Year Average of site 04196800 Slope: 1.4919359242891381, P-value: 0.03868635575373688
    
     For Annual Maximum Flow of site 04196800 Slope: 24.562988374309082, P-value: 0.03972074062799647
    
     For 7-Day Low Flow of site 04196800 Slope: 0.018063546516372625, P-value: 0.26385549317746243
    


    
![png](output_11_236.png)
    


    
     For Winter-Spring Center Volume Date of site 04196800 Slope: 0.19916142557651834, P-value: 0.7233089932220422
    


    
![png](output_11_238.png)
    



    
![png](output_11_239.png)
    


    
     For Water Year Average of site 04136000 Slope: 6.1423820991878095, P-value: 0.007622862244935213
    
     For Annual Maximum Flow of site 04136000 Slope: 5.27586206896556, P-value: 0.6788477456739044
    
     For 7-Day Low Flow of site 04136000 Slope: 3.619000703729564, P-value: 0.0037084255118428277
    


    
![png](output_11_241.png)
    


    
     For Winter-Spring Center Volume Date of site 04136000 Slope: -0.14039408866996117, P-value: 0.4872287027148706
    


    
![png](output_11_243.png)
    



    
![png](output_11_244.png)
    


    
     For Water Year Average of site 04213000 Slope: -0.1950293230436192, P-value: 0.7244648076157153
    
     For Annual Maximum Flow of site 04213000 Slope: -29.77320373546809, P-value: 0.07156252548104709
    
     For 7-Day Low Flow of site 04213000 Slope: -0.015040050096652692, P-value: 0.8402462827691967
    


    
![png](output_11_246.png)
    


    
     For Winter-Spring Center Volume Date of site 04213000 Slope: 0.00266819134743557, P-value: 0.9892734050415397
    


    
![png](output_11_248.png)
    



    
![png](output_11_249.png)
    


    
     For Water Year Average of site 04256000 Slope: 0.332450694876628, P-value: 0.2449547726579893
    
     For Annual Maximum Flow of site 04256000 Slope: -0.9971031065371196, P-value: 0.8833303154303436
    
     For 7-Day Low Flow of site 04256000 Slope: -0.15290533366005052, P-value: 0.11958141835490681
    


    
![png](output_11_251.png)
    


    
     For Winter-Spring Center Volume Date of site 04256000 Slope: -0.22088812654850332, P-value: 0.23958387103006493
    


    
![png](output_11_253.png)
    



    
![png](output_11_254.png)
    


    
     For Water Year Average of site 04057510 Slope: -0.014110270291149862, P-value: 0.9749479208485068
    
     For Annual Maximum Flow of site 04057510 Slope: -2.330398322851135, P-value: 0.4579787783481851
    
     For 7-Day Low Flow of site 04057510 Slope: -0.16153502681804652, P-value: 0.24889548326615335
    


    
![png](output_11_256.png)
    


    
     For Winter-Spring Center Volume Date of site 04057510 Slope: 0.0014865637507144314, P-value: 0.9913814351765703
    


    
![png](output_11_258.png)
    



    
![png](output_11_259.png)
    


    
     For Water Year Average of site 04122500 Slope: -0.0643661571828783, P-value: 0.9498075162150207
    
     For Annual Maximum Flow of site 04122500 Slope: -3.127501429388245, P-value: 0.6568137592364451
    
     For 7-Day Low Flow of site 04122500 Slope: 0.6194233439516419, P-value: 0.17174266434263125
    


    
![png](output_11_261.png)
    


    
     For Winter-Spring Center Volume Date of site 04122500 Slope: 0.05008576329331088, P-value: 0.6561043596709073
    


    
![png](output_11_263.png)
    



    
![png](output_11_264.png)
    


    
     For Water Year Average of site 04085200 Slope: 0.053066589024565874, P-value: 0.8751793962827079
    
     For Annual Maximum Flow of site 04085200 Slope: -11.845932331627916, P-value: 0.23109250122875247
    
     For 7-Day Low Flow of site 04085200 Slope: -0.012299762580814402, P-value: 0.796028052202286
    


    
![png](output_11_266.png)
    


    
     For Winter-Spring Center Volume Date of site 04085200 Slope: 0.31193848225151455, P-value: 0.14760240224766258
    


    
![png](output_11_268.png)
    



    
![png](output_11_269.png)
    


    
     For Water Year Average of site 04059500 Slope: -0.8141899019687258, P-value: 0.46387406981853774
    
     For Annual Maximum Flow of site 04059500 Slope: -10.064417762530923, P-value: 0.2137042219231582
    
     For 7-Day Low Flow of site 04059500 Slope: -0.07961049851616032, P-value: 0.7886720908561943
    


    
![png](output_11_271.png)
    


    
     For Winter-Spring Center Volume Date of site 04059500 Slope: -0.021879169048980865, P-value: 0.8760571720147775
    


    
![png](output_11_273.png)
    



    
![png](output_11_274.png)
    


    
     For Water Year Average of site 04056500 Slope: -1.0930405284863545, P-value: 0.7055697234979756
    
     For Annual Maximum Flow of site 04056500 Slope: -38.443300933866894, P-value: 0.056285809826271016
    
     For 7-Day Low Flow of site 04056500 Slope: -2.5784747747011956, P-value: 0.010992552668734395
    


    
![png](output_11_276.png)
    


    
     For Winter-Spring Center Volume Date of site 04056500 Slope: -0.27996950638460183, P-value: 0.015880952156205785
    


    
![png](output_11_278.png)
    



    
![png](output_11_279.png)
    


    
     For Water Year Average of site 04031000 Slope: 0.4247504312366934, P-value: 0.4732654936369213
    
     For Annual Maximum Flow of site 04031000 Slope: 8.866735271417467, P-value: 0.645021544881664
    
     For 7-Day Low Flow of site 04031000 Slope: 0.004577652002399624, P-value: 0.9469831293229796
    


    
![png](output_11_281.png)
    


    
     For Winter-Spring Center Volume Date of site 04031000 Slope: -0.11379813052053808, P-value: 0.31343078165233146
    


    
![png](output_11_283.png)
    



    
![png](output_11_284.png)
    


    
     For Water Year Average of site 04045500 Slope: -0.5659779909380378, P-value: 0.7085699313524481
    
     For Annual Maximum Flow of site 04045500 Slope: -11.894797026872459, P-value: 0.2438792921104055
    
     For 7-Day Low Flow of site 04045500 Slope: -0.8521821993520049, P-value: 0.08668418646206749
    


    
![png](output_11_286.png)
    


    
     For Winter-Spring Center Volume Date of site 04045500 Slope: -0.2667047836859166, P-value: 0.023359854559996996
    


    
![png](output_11_288.png)
    



    
![png](output_11_289.png)
    


    
     For Water Year Average of site 04122200 Slope: -0.13646135933896741, P-value: 0.842752390069506
    
     For Annual Maximum Flow of site 04122200 Slope: -7.344577854011856, P-value: 0.34089208120176495
    
     For 7-Day Low Flow of site 04122200 Slope: -0.029252089629451042, P-value: 0.912249311009854
    


    
![png](output_11_291.png)
    


    
     For Winter-Spring Center Volume Date of site 04122200 Slope: 0.06113969887554779, P-value: 0.5930209181482895
    


    
![png](output_11_293.png)
    



    
![png](output_11_294.png)
    


    
     For Water Year Average of site 04027000 Slope: 0.6063473900220693, P-value: 0.7087825096856437
    
     For Annual Maximum Flow of site 04027000 Slope: 46.93767867352799, P-value: 0.26986573737787944
    
     For 7-Day Low Flow of site 04027000 Slope: -0.07063519289934478, P-value: 0.8101956890409127
    


    
![png](output_11_296.png)
    


    
     For Winter-Spring Center Volume Date of site 04027000 Slope: 0.02854964741757139, P-value: 0.8299193169658627
    


    
![png](output_11_298.png)
    



    
![png](output_11_299.png)
    


    
     For Water Year Average of site 04040500 Slope: -0.6495244978687618, P-value: 0.12383717098743277
    
     For Annual Maximum Flow of site 04040500 Slope: -3.9356203544882775, P-value: 0.5997939696599401
    
     For 7-Day Low Flow of site 04040500 Slope: -0.19021519780010437, P-value: 0.05978340611004086
    


    
![png](output_11_301.png)
    


    
     For Winter-Spring Center Volume Date of site 04040500 Slope: -0.13821231179721802, P-value: 0.16229495750194137
    


    
![png](output_11_303.png)
    



    
![png](output_11_304.png)
    


    
     For Water Year Average of site 04066500 Slope: 3.392505293463914, P-value: 0.02414495052175088
    
     For Annual Maximum Flow of site 04066500 Slope: 9.539230769230793, P-value: 0.1525239898229185
    
     For 7-Day Low Flow of site 04066500 Slope: 1.6746923076922946, P-value: 0.023724819722037913
    


    
![png](output_11_306.png)
    


    
     For Winter-Spring Center Volume Date of site 04066500 Slope: -1.1338461538461884, P-value: 0.1314569160206079
    


    
![png](output_11_308.png)
    



    
![png](output_11_309.png)
    


    
     For Water Year Average of site 04074950 Slope: -0.37347511400445976, P-value: 0.6642951992789111
    
     For Annual Maximum Flow of site 04074950 Slope: 0.5594816085382242, P-value: 0.8773802154851308
    
     For 7-Day Low Flow of site 04074950 Slope: -0.4976666938931128, P-value: 0.2598612252291923
    


    
![png](output_11_311.png)
    


    
     For Winter-Spring Center Volume Date of site 04074950 Slope: -0.12502382313703206, P-value: 0.5940363166830841
    


    
![png](output_11_313.png)
    



    
![png](output_11_314.png)
    


    
     For Water Year Average of site 04124000 Slope: 0.8383840233106996, P-value: 0.27052590964569345
    
     For Annual Maximum Flow of site 04124000 Slope: -4.557270821421739, P-value: 0.23832219647936323
    
     For 7-Day Low Flow of site 04124000 Slope: 0.7961011734596648, P-value: 0.0957860756321315
    


    
![png](output_11_316.png)
    


    
     For Winter-Spring Center Volume Date of site 04124000 Slope: 0.23777396607585355, P-value: 0.343113933161338
    


    
![png](output_11_318.png)
    



    
![png](output_11_319.png)
    


    
     For Water Year Average of site 04125460 Slope: -0.11887061899678741, P-value: 0.6450042687033708
    
     For Annual Maximum Flow of site 04125460 Slope: -1.6155867949728968, P-value: 0.6830319829541769
    
     For 7-Day Low Flow of site 04125460 Slope: -0.3853231542951018, P-value: 0.003171198378957456
    


    
![png](output_11_321.png)
    


    
     For Winter-Spring Center Volume Date of site 04125460 Slope: -0.13469572959830062, P-value: 0.10282667585580543
    


    
![png](output_11_323.png)
    



    
![png](output_11_324.png)
    


    
     For Water Year Average of site 04221000 Slope: 0.30548260155633433, P-value: 0.7264368872809381
    
     For Annual Maximum Flow of site 04221000 Slope: -8.556731836421061, P-value: 0.6701425230002895
    
     For 7-Day Low Flow of site 04221000 Slope: -0.0974552084984939, P-value: 0.3719458865823063
    


    
![png](output_11_326.png)
    


    
     For Winter-Spring Center Volume Date of site 04221000 Slope: -0.30342354648680564, P-value: 0.22367908783298177
    


    
![png](output_11_328.png)
    



    
![png](output_11_329.png)
    


    
     For Water Year Average of site 04185000 Slope: 0.5488308522788837, P-value: 0.5521282383610908
    
     For Annual Maximum Flow of site 04185000 Slope: 2.087669144272877, P-value: 0.8737130552153869
    
     For 7-Day Low Flow of site 04185000 Slope: 0.03515995534863427, P-value: 0.6611557899704701
    


    
![png](output_11_331.png)
    


    
     For Winter-Spring Center Volume Date of site 04185000 Slope: 0.34671240708976425, P-value: 0.5114337123033066
    


    
![png](output_11_333.png)
    



    
![png](output_11_334.png)
    


    
     For Water Year Average of site 04196800 Slope: 1.4919359242891381, P-value: 0.03868635575373688
    
     For Annual Maximum Flow of site 04196800 Slope: 24.562988374309082, P-value: 0.03972074062799647
    
     For 7-Day Low Flow of site 04196800 Slope: 0.018063546516372625, P-value: 0.26385549317746243
    


    
![png](output_11_336.png)
    


    
     For Winter-Spring Center Volume Date of site 04196800 Slope: 0.19916142557651834, P-value: 0.7233089932220422
    


    
![png](output_11_338.png)
    



    
![png](output_11_339.png)
    


    
     For Water Year Average of site 04136000 Slope: 6.1423820991878095, P-value: 0.007622862244935213
    
     For Annual Maximum Flow of site 04136000 Slope: 5.27586206896556, P-value: 0.6788477456739044
    
     For 7-Day Low Flow of site 04136000 Slope: 3.619000703729564, P-value: 0.0037084255118428277
    


    
![png](output_11_341.png)
    


    
     For Winter-Spring Center Volume Date of site 04136000 Slope: -0.14039408866996117, P-value: 0.4872287027148706
    


    
![png](output_11_343.png)
    



    
![png](output_11_344.png)
    


    
     For Water Year Average of site 04213000 Slope: -0.1950293230436192, P-value: 0.7244648076157153
    
     For Annual Maximum Flow of site 04213000 Slope: -29.77320373546809, P-value: 0.07156252548104709
    
     For 7-Day Low Flow of site 04213000 Slope: -0.015040050096652692, P-value: 0.8402462827691967
    


    
![png](output_11_346.png)
    


    
     For Winter-Spring Center Volume Date of site 04213000 Slope: 0.00266819134743557, P-value: 0.9892734050415397
    


    
![png](output_11_348.png)
    



    
![png](output_11_349.png)
    


    
     For Water Year Average of site 04256000 Slope: 0.332450694876628, P-value: 0.2449547726579893
    
     For Annual Maximum Flow of site 04256000 Slope: -0.9971031065371196, P-value: 0.8833303154303436
    
     For 7-Day Low Flow of site 04256000 Slope: -0.15290533366005052, P-value: 0.11958141835490681
    


    
![png](output_11_351.png)
    


    
     For Winter-Spring Center Volume Date of site 04256000 Slope: -0.22088812654850332, P-value: 0.23958387103006493
    


    
![png](output_11_353.png)
    



    
![png](output_11_354.png)
    


    
     For Water Year Average of site 04057510 Slope: -0.014110270291149862, P-value: 0.9749479208485068
    
     For Annual Maximum Flow of site 04057510 Slope: -2.330398322851135, P-value: 0.4579787783481851
    
     For 7-Day Low Flow of site 04057510 Slope: -0.16153502681804652, P-value: 0.24889548326615335
    


    
![png](output_11_356.png)
    


    
     For Winter-Spring Center Volume Date of site 04057510 Slope: 0.0014865637507144314, P-value: 0.9913814351765703
    


    
![png](output_11_358.png)
    



    
![png](output_11_359.png)
    


    
     For Water Year Average of site 04122500 Slope: -0.0643661571828783, P-value: 0.9498075162150207
    
     For Annual Maximum Flow of site 04122500 Slope: -3.127501429388245, P-value: 0.6568137592364451
    
     For 7-Day Low Flow of site 04122500 Slope: 0.6194233439516419, P-value: 0.17174266434263125
    


    
![png](output_11_361.png)
    


    
     For Winter-Spring Center Volume Date of site 04122500 Slope: 0.05008576329331088, P-value: 0.6561043596709073
    


    
![png](output_11_363.png)
    



    
![png](output_11_364.png)
    


    
     For Water Year Average of site 04085200 Slope: 0.053066589024565874, P-value: 0.8751793962827079
    
     For Annual Maximum Flow of site 04085200 Slope: -11.845932331627916, P-value: 0.23109250122875247
    
     For 7-Day Low Flow of site 04085200 Slope: -0.012299762580814402, P-value: 0.796028052202286
    


    
![png](output_11_366.png)
    


    
     For Winter-Spring Center Volume Date of site 04085200 Slope: 0.31193848225151455, P-value: 0.14760240224766258
    


    
![png](output_11_368.png)
    



    
![png](output_11_369.png)
    


    
     For Water Year Average of site 04059500 Slope: -0.8141899019687258, P-value: 0.46387406981853774
    
     For Annual Maximum Flow of site 04059500 Slope: -10.064417762530923, P-value: 0.2137042219231582
    
     For 7-Day Low Flow of site 04059500 Slope: -0.07961049851616032, P-value: 0.7886720908561943
    


    
![png](output_11_371.png)
    


    
     For Winter-Spring Center Volume Date of site 04059500 Slope: -0.021879169048980865, P-value: 0.8760571720147775
    


    
![png](output_11_373.png)
    



    
![png](output_11_374.png)
    


    
     For Water Year Average of site 04056500 Slope: -1.0930405284863545, P-value: 0.7055697234979756
    
     For Annual Maximum Flow of site 04056500 Slope: -38.443300933866894, P-value: 0.056285809826271016
    
     For 7-Day Low Flow of site 04056500 Slope: -2.5784747747011956, P-value: 0.010992552668734395
    


    
![png](output_11_376.png)
    


    
     For Winter-Spring Center Volume Date of site 04056500 Slope: -0.27996950638460183, P-value: 0.015880952156205785
    


    
![png](output_11_378.png)
    



    
![png](output_11_379.png)
    


    
     For Water Year Average of site 04031000 Slope: 0.4247504312366934, P-value: 0.4732654936369213
    
     For Annual Maximum Flow of site 04031000 Slope: 8.866735271417467, P-value: 0.645021544881664
    
     For 7-Day Low Flow of site 04031000 Slope: 0.004577652002399624, P-value: 0.9469831293229796
    


    
![png](output_11_381.png)
    


    
     For Winter-Spring Center Volume Date of site 04031000 Slope: -0.11379813052053808, P-value: 0.31343078165233146
    


    
![png](output_11_383.png)
    



    
![png](output_11_384.png)
    


    
     For Water Year Average of site 04045500 Slope: -0.5659779909380378, P-value: 0.7085699313524481
    
     For Annual Maximum Flow of site 04045500 Slope: -11.894797026872459, P-value: 0.2438792921104055
    
     For 7-Day Low Flow of site 04045500 Slope: -0.8521821993520049, P-value: 0.08668418646206749
    


    
![png](output_11_386.png)
    


    
     For Winter-Spring Center Volume Date of site 04045500 Slope: -0.2667047836859166, P-value: 0.023359854559996996
    


    
![png](output_11_388.png)
    



    
![png](output_11_389.png)
    


    
     For Water Year Average of site 04122200 Slope: -0.13646135933896741, P-value: 0.842752390069506
    
     For Annual Maximum Flow of site 04122200 Slope: -7.344577854011856, P-value: 0.34089208120176495
    
     For 7-Day Low Flow of site 04122200 Slope: -0.029252089629451042, P-value: 0.912249311009854
    


    
![png](output_11_391.png)
    


    
     For Winter-Spring Center Volume Date of site 04122200 Slope: 0.06113969887554779, P-value: 0.5930209181482895
    


    
![png](output_11_393.png)
    



    
![png](output_11_394.png)
    


    
     For Water Year Average of site 04027000 Slope: 0.6063473900220693, P-value: 0.7087825096856437
    
     For Annual Maximum Flow of site 04027000 Slope: 46.93767867352799, P-value: 0.26986573737787944
    
     For 7-Day Low Flow of site 04027000 Slope: -0.07063519289934478, P-value: 0.8101956890409127
    


    
![png](output_11_396.png)
    


    
     For Winter-Spring Center Volume Date of site 04027000 Slope: 0.02854964741757139, P-value: 0.8299193169658627
    


    
![png](output_11_398.png)
    



    
![png](output_11_399.png)
    


    
     For Water Year Average of site 04040500 Slope: -0.6495244978687618, P-value: 0.12383717098743277
    
     For Annual Maximum Flow of site 04040500 Slope: -3.9356203544882775, P-value: 0.5997939696599401
    
     For 7-Day Low Flow of site 04040500 Slope: -0.19021519780010437, P-value: 0.05978340611004086
    


    
![png](output_11_401.png)
    


    
     For Winter-Spring Center Volume Date of site 04040500 Slope: -0.13821231179721802, P-value: 0.16229495750194137
    


    
![png](output_11_403.png)
    



    
![png](output_11_404.png)
    


    
     For Water Year Average of site 04066500 Slope: 3.392505293463914, P-value: 0.02414495052175088
    
     For Annual Maximum Flow of site 04066500 Slope: 9.539230769230793, P-value: 0.1525239898229185
    
     For 7-Day Low Flow of site 04066500 Slope: 1.6746923076922946, P-value: 0.023724819722037913
    


    
![png](output_11_406.png)
    


    
     For Winter-Spring Center Volume Date of site 04066500 Slope: -1.1338461538461884, P-value: 0.1314569160206079
    


    
![png](output_11_408.png)
    



    
![png](output_11_409.png)
    


    
     For Water Year Average of site 04074950 Slope: -0.37347511400445976, P-value: 0.6642951992789111
    
     For Annual Maximum Flow of site 04074950 Slope: 0.5594816085382242, P-value: 0.8773802154851308
    
     For 7-Day Low Flow of site 04074950 Slope: -0.4976666938931128, P-value: 0.2598612252291923
    


    
![png](output_11_411.png)
    


    
     For Winter-Spring Center Volume Date of site 04074950 Slope: -0.12502382313703206, P-value: 0.5940363166830841
    


    
![png](output_11_413.png)
    



    
![png](output_11_414.png)
    


    
     For Water Year Average of site 04124000 Slope: 0.8383840233106996, P-value: 0.27052590964569345
    
     For Annual Maximum Flow of site 04124000 Slope: -4.557270821421739, P-value: 0.23832219647936323
    
     For 7-Day Low Flow of site 04124000 Slope: 0.7961011734596648, P-value: 0.0957860756321315
    


    
![png](output_11_416.png)
    


    
     For Winter-Spring Center Volume Date of site 04124000 Slope: 0.23777396607585355, P-value: 0.343113933161338
    


    
![png](output_11_418.png)
    



    
![png](output_11_419.png)
    


    
     For Water Year Average of site 04125460 Slope: -0.11887061899678741, P-value: 0.6450042687033708
    
     For Annual Maximum Flow of site 04125460 Slope: -1.6155867949728968, P-value: 0.6830319829541769
    
     For 7-Day Low Flow of site 04125460 Slope: -0.3853231542951018, P-value: 0.003171198378957456
    


    
![png](output_11_421.png)
    


    
     For Winter-Spring Center Volume Date of site 04125460 Slope: -0.13469572959830062, P-value: 0.10282667585580543
    


    
![png](output_11_423.png)
    



    
![png](output_11_424.png)
    


    
     For Water Year Average of site 04221000 Slope: 0.30548260155633433, P-value: 0.7264368872809381
    
     For Annual Maximum Flow of site 04221000 Slope: -8.556731836421061, P-value: 0.6701425230002895
    
     For 7-Day Low Flow of site 04221000 Slope: -0.0974552084984939, P-value: 0.3719458865823063
    


    
![png](output_11_426.png)
    


    
     For Winter-Spring Center Volume Date of site 04221000 Slope: -0.30342354648680564, P-value: 0.22367908783298177
    


    
![png](output_11_428.png)
    



    
![png](output_11_429.png)
    


    
     For Water Year Average of site 04185000 Slope: 0.5488308522788837, P-value: 0.5521282383610908
    
     For Annual Maximum Flow of site 04185000 Slope: 2.087669144272877, P-value: 0.8737130552153869
    
     For 7-Day Low Flow of site 04185000 Slope: 0.03515995534863427, P-value: 0.6611557899704701
    


    
![png](output_11_431.png)
    


    
     For Winter-Spring Center Volume Date of site 04185000 Slope: 0.34671240708976425, P-value: 0.5114337123033066
    


    
![png](output_11_433.png)
    



    
![png](output_11_434.png)
    


    
     For Water Year Average of site 04196800 Slope: 1.4919359242891381, P-value: 0.03868635575373688
    
     For Annual Maximum Flow of site 04196800 Slope: 24.562988374309082, P-value: 0.03972074062799647
    
     For 7-Day Low Flow of site 04196800 Slope: 0.018063546516372625, P-value: 0.26385549317746243
    


    
![png](output_11_436.png)
    


    
     For Winter-Spring Center Volume Date of site 04196800 Slope: 0.19916142557651834, P-value: 0.7233089932220422
    


    
![png](output_11_438.png)
    



    
![png](output_11_439.png)
    


    
     For Water Year Average of site 04136000 Slope: 6.1423820991878095, P-value: 0.007622862244935213
    
     For Annual Maximum Flow of site 04136000 Slope: 5.27586206896556, P-value: 0.6788477456739044
    
     For 7-Day Low Flow of site 04136000 Slope: 3.619000703729564, P-value: 0.0037084255118428277
    


    
![png](output_11_441.png)
    


    
     For Winter-Spring Center Volume Date of site 04136000 Slope: -0.14039408866996117, P-value: 0.4872287027148706
    


    
![png](output_11_443.png)
    



    
![png](output_11_444.png)
    


    
     For Water Year Average of site 04213000 Slope: -0.1950293230436192, P-value: 0.7244648076157153
    
     For Annual Maximum Flow of site 04213000 Slope: -29.77320373546809, P-value: 0.07156252548104709
    
     For 7-Day Low Flow of site 04213000 Slope: -0.015040050096652692, P-value: 0.8402462827691967
    


    
![png](output_11_446.png)
    


    
     For Winter-Spring Center Volume Date of site 04213000 Slope: 0.00266819134743557, P-value: 0.9892734050415397
    


    
![png](output_11_448.png)
    



    
![png](output_11_449.png)
    


    
     For Water Year Average of site 04256000 Slope: 0.332450694876628, P-value: 0.2449547726579893
    
     For Annual Maximum Flow of site 04256000 Slope: -0.9971031065371196, P-value: 0.8833303154303436
    
     For 7-Day Low Flow of site 04256000 Slope: -0.15290533366005052, P-value: 0.11958141835490681
    


    
![png](output_11_451.png)
    


    
     For Winter-Spring Center Volume Date of site 04256000 Slope: -0.22088812654850332, P-value: 0.23958387103006493
    


    
![png](output_11_453.png)
    



    
![png](output_11_454.png)
    


    
     For Water Year Average of site 04057510 Slope: -0.014110270291149862, P-value: 0.9749479208485068
    
     For Annual Maximum Flow of site 04057510 Slope: -2.330398322851135, P-value: 0.4579787783481851
    
     For 7-Day Low Flow of site 04057510 Slope: -0.16153502681804652, P-value: 0.24889548326615335
    


    
![png](output_11_456.png)
    


    
     For Winter-Spring Center Volume Date of site 04057510 Slope: 0.0014865637507144314, P-value: 0.9913814351765703
    


    
![png](output_11_458.png)
    



    
![png](output_11_459.png)
    


    
     For Water Year Average of site 04122500 Slope: -0.0643661571828783, P-value: 0.9498075162150207
    
     For Annual Maximum Flow of site 04122500 Slope: -3.127501429388245, P-value: 0.6568137592364451
    
     For 7-Day Low Flow of site 04122500 Slope: 0.6194233439516419, P-value: 0.17174266434263125
    


    
![png](output_11_461.png)
    


    
     For Winter-Spring Center Volume Date of site 04122500 Slope: 0.05008576329331088, P-value: 0.6561043596709073
    


    
![png](output_11_463.png)
    



    
![png](output_11_464.png)
    



```python
## Create a matrix to store all of the info below
# dictionary of lists 
dict = {'Site Numbers': site_list, 'Latitude': latitude, 'Longitude': longitude, 
        'P-value Yearly Average Flow': p_values_yearly_avg, 'Slope Yearly Average Flow':slopes_yearly_avg,
        'P-value Max Flow':p_values_max,'Slope Max Flow':slopes_max,
        'P-value 7 Day Low Flow':p_values_seven_day_low_flow,'Slope 7 Day Low Flow':slopes_seven_day_low_flow,
        'P-value WSCV':p_values_WSCV,'Slope WSCV':slopes_WSCV,
        'P-value Fall Average': p_values_Fallavg, 'Slope Yearly Fall Average':slopes_Fallavg,
        'P-value Winter Average': p_values_Winteravg, 'Slope Yearly Winter Average':slopes_Winteravg,
        'P-value Spring Average': p_values_Springavg, 'Slope Yearly Spring Average':slopes_Springavg,
        'P-value Summer Average': p_values_Summeravg, 'Slope Yearly Summer Average':slopes_Summeravg} 

for key, value in dict.items():
    print(f"{key}: {len(value)}")
    
df_1970 = pd.DataFrame(dict)

df_1970
```

    Site Numbers: 93
    Latitude: 93
    Longitude: 93
    P-value Yearly Average Flow: 93
    Slope Yearly Average Flow: 93
    P-value Max Flow: 93
    Slope Max Flow: 93
    P-value 7 Day Low Flow: 93
    Slope 7 Day Low Flow: 93
    P-value WSCV: 93
    Slope WSCV: 93
    P-value Fall Average: 93
    Slope Yearly Fall Average: 93
    P-value Winter Average: 93
    Slope Yearly Winter Average: 93
    P-value Spring Average: 93
    Slope Yearly Spring Average: 93
    P-value Summer Average: 93
    Slope Yearly Summer Average: 93
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Site Numbers</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>P-value Yearly Average Flow</th>
      <th>Slope Yearly Average Flow</th>
      <th>P-value Max Flow</th>
      <th>Slope Max Flow</th>
      <th>P-value 7 Day Low Flow</th>
      <th>Slope 7 Day Low Flow</th>
      <th>P-value WSCV</th>
      <th>Slope WSCV</th>
      <th>P-value Fall Average</th>
      <th>Slope Yearly Fall Average</th>
      <th>P-value Winter Average</th>
      <th>Slope Yearly Winter Average</th>
      <th>P-value Spring Average</th>
      <th>Slope Yearly Spring Average</th>
      <th>P-value Summer Average</th>
      <th>Slope Yearly Summer Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>04040500</td>
      <td>46.584106</td>
      <td>-88.575970</td>
      <td>0.123837</td>
      <td>-0.649524</td>
      <td>0.599794</td>
      <td>-3.935620</td>
      <td>0.059783</td>
      <td>-0.190215</td>
      <td>0.162295</td>
      <td>-0.138212</td>
      <td>0.289686</td>
      <td>-0.836804</td>
      <td>0.792583</td>
      <td>0.086896</td>
      <td>0.226602</td>
      <td>-1.453729</td>
      <td>0.108683</td>
      <td>-0.737168</td>
    </tr>
    <tr>
      <th>1</th>
      <td>04066500</td>
      <td>45.500000</td>
      <td>-88.000000</td>
      <td>0.024145</td>
      <td>3.392505</td>
      <td>0.152524</td>
      <td>9.539231</td>
      <td>0.023725</td>
      <td>1.674692</td>
      <td>0.131457</td>
      <td>-1.133846</td>
      <td>0.332455</td>
      <td>1.697717</td>
      <td>0.001316</td>
      <td>3.166834</td>
      <td>0.078319</td>
      <td>5.315140</td>
      <td>0.091305</td>
      <td>3.328921</td>
    </tr>
    <tr>
      <th>2</th>
      <td>04074950</td>
      <td>45.189970</td>
      <td>-88.733440</td>
      <td>0.664295</td>
      <td>-0.373475</td>
      <td>0.877380</td>
      <td>0.559482</td>
      <td>0.259861</td>
      <td>-0.497667</td>
      <td>0.594036</td>
      <td>-0.125024</td>
      <td>0.342814</td>
      <td>-0.953179</td>
      <td>0.217537</td>
      <td>-0.720996</td>
      <td>0.906749</td>
      <td>0.198010</td>
      <td>0.423357</td>
      <td>0.844939</td>
    </tr>
    <tr>
      <th>3</th>
      <td>04124000</td>
      <td>44.436392</td>
      <td>-85.698679</td>
      <td>0.270526</td>
      <td>0.838384</td>
      <td>0.238322</td>
      <td>-4.557271</td>
      <td>0.095786</td>
      <td>0.796101</td>
      <td>0.343114</td>
      <td>0.237774</td>
      <td>0.497209</td>
      <td>0.891642</td>
      <td>0.006854</td>
      <td>2.294331</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>04125460</td>
      <td>44.193340</td>
      <td>-85.769786</td>
      <td>0.645004</td>
      <td>-0.118871</td>
      <td>0.683032</td>
      <td>-1.615587</td>
      <td>0.003171</td>
      <td>-0.385323</td>
      <td>0.102827</td>
      <td>-0.134696</td>
      <td>0.970094</td>
      <td>0.011597</td>
      <td>0.242333</td>
      <td>0.396432</td>
      <td>0.366599</td>
      <td>-0.519323</td>
      <td>0.186078</td>
      <td>-0.346672</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>04136000</td>
      <td>44.676959</td>
      <td>-84.292515</td>
      <td>0.007623</td>
      <td>6.142382</td>
      <td>0.678848</td>
      <td>5.275862</td>
      <td>0.003708</td>
      <td>3.619001</td>
      <td>0.487229</td>
      <td>-0.140394</td>
      <td>0.088034</td>
      <td>4.960098</td>
      <td>0.010863</td>
      <td>6.175218</td>
      <td>0.116894</td>
      <td>6.716486</td>
      <td>0.016616</td>
      <td>4.582668</td>
    </tr>
    <tr>
      <th>89</th>
      <td>04213000</td>
      <td>41.926999</td>
      <td>-80.603966</td>
      <td>0.724465</td>
      <td>-0.195029</td>
      <td>0.071563</td>
      <td>-29.773204</td>
      <td>0.840246</td>
      <td>-0.015040</td>
      <td>0.989273</td>
      <td>0.002668</td>
      <td>0.260526</td>
      <td>-1.434367</td>
      <td>0.668896</td>
      <td>0.590684</td>
      <td>0.432925</td>
      <td>0.761809</td>
      <td>0.406629</td>
      <td>-0.694397</td>
    </tr>
    <tr>
      <th>90</th>
      <td>04256000</td>
      <td>43.746806</td>
      <td>-75.333444</td>
      <td>0.244955</td>
      <td>0.332451</td>
      <td>0.883330</td>
      <td>-0.997103</td>
      <td>0.119581</td>
      <td>-0.152905</td>
      <td>0.239584</td>
      <td>-0.220888</td>
      <td>0.654825</td>
      <td>-0.248046</td>
      <td>0.056373</td>
      <td>1.196070</td>
      <td>0.150902</td>
      <td>-0.955905</td>
      <td>0.015071</td>
      <td>1.152925</td>
    </tr>
    <tr>
      <th>91</th>
      <td>04057510</td>
      <td>45.943024</td>
      <td>-86.705700</td>
      <td>0.974948</td>
      <td>-0.014110</td>
      <td>0.457979</td>
      <td>-2.330398</td>
      <td>0.248895</td>
      <td>-0.161535</td>
      <td>0.991381</td>
      <td>0.001487</td>
      <td>0.124272</td>
      <td>-1.066100</td>
      <td>0.521404</td>
      <td>0.221176</td>
      <td>0.470310</td>
      <td>-0.657460</td>
      <td>0.723322</td>
      <td>-0.159039</td>
    </tr>
    <tr>
      <th>92</th>
      <td>04122500</td>
      <td>43.945006</td>
      <td>-86.278690</td>
      <td>0.949808</td>
      <td>-0.064366</td>
      <td>0.656814</td>
      <td>-3.127501</td>
      <td>0.171743</td>
      <td>0.619423</td>
      <td>0.656104</td>
      <td>0.050086</td>
      <td>0.731208</td>
      <td>-0.562194</td>
      <td>0.961396</td>
      <td>-0.061647</td>
      <td>0.796605</td>
      <td>0.432828</td>
      <td>0.938778</td>
      <td>0.085391</td>
    </tr>
  </tbody>
</table>
<p>93 rows × 19 columns</p>
</div>




```python
sites=ca_site_list_1970

print(dfSites.columns)


for s in sites:

    # Get daily values (dv)

    # Construct the full file path
    file_path = os.path.join(data_directory, f'{s}.csv')
    # Load the CSV data into a DataFrame
    df = pd.read_csv(file_path)
        # Ensure that the 'date' column exists and convert it to datetime if necessary
    if 'Date' not in df.columns:
        print(f"The expected 'Date' column is not found for file {filename}")
        continue
    #Check if mean DV exists
    if 'Parameter/Paramètre' in df.columns:
        if not df['Parameter/Paramètre'].str.contains('discharge/débit', case=False, na=False).any():
            print(f"No discharge data for site {s}")
            continue
    # Access the first row of column 'ID'
    if '﻿ ID' not in df.columns:
        print(f"The expected 'ID' column is not found in the data for file {filename}.")
        continue
        
    df = df[df['Parameter/Paramètre'].str.contains('discharge', case=False, na=False)]
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['water_year'] = df['Date'].apply(WaterYear) 
    df['Season'] = df['Date'].apply(Season)
    

    # Set the 'date' column as the index
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)    
        
    s = df.iloc[0]['﻿ ID']
    
    # Find the first and last year
    first_year = df['water_year'].min()
    last_year = df['water_year'].max()

    # Filter out the rows where the year is the first or last
    df = df[(df['water_year'] != first_year) & (df['water_year'] != last_year)]

    latitude.append(dfSites[dfSites['ID'] == s]['Lat'].values)
    longitude.append(dfSites[dfSites['ID'] == s]['Lon'].values)

    ##Finding Attributes
    #Mean
    # Ensure the 'Value/Valeur' column is numeric
    df['Value/Valeur'] = pd.to_numeric(df['Value/Valeur'], errors='coerce')
    # Set values <= 0 to NaN
    df['Value/Valeur'] = df['Value/Valeur'].where(df['Value/Valeur'] > 0, np.nan)
    # Calculate the the mean of 'Value/Valeur' grouped by 'water_year'
    water_yearly_avg = df.groupby('water_year')['Value/Valeur'].mean()
    
    # Calculate the maximum for each water year
    annual_max = df.groupby('water_year')['Value/Valeur'].max()

    #7-Day Low Flow
    # Compute 7-day rolling minimum
    df['7_day_mean'] = df['Value/Valeur'].rolling(window=7, min_periods=1).mean()
    # Calculate the minimum of the 7 day rolling mean for each water year
    seven_day_low_flow = df.groupby('water_year')['7_day_mean'].min()
    
   # Check the 'value' column for discharge data
    seconds_per_day = 24 * 60 * 60
    df['daily_volume_cubic_ft'] = df['Value/Valeur'] * seconds_per_day
    df['cumulative_volume'] = df.groupby('water_year')['daily_volume_cubic_ft'].cumsum()
    df['date_column'] = df.index  # Ensure the index is datetime
    
## Calculations for center volume / center mass
    total_volume_each_wateryear =df.groupby('water_year')['daily_volume_cubic_ft'].sum()
    half_volume_each_wateryear = total_volume_each_wateryear / 2
    half_volume_dates = {}
    # Iterate over each unique water year
    for water_year in df['water_year'].unique():
        # Get the half volume for the current water year
        half_volume = half_volume_each_wateryear[water_year]
        # Get the subset of data for the current water year
        year_data = df[df['water_year'] == water_year]
        # Find the index where cumulative volume reaches or exceeds half of total volume
        half_volume_index = year_data[year_data['cumulative_volume'] >= half_volume].index
        if not half_volume_index.empty:
            # Retrieve the actual date from the DataFrame using the first index
            first_index = half_volume_index[0]  # Get the first index
            half_volume_dates[water_year] = df.at[first_index, 'date_column']  # Use 'at' to get the value
     # Convert dates to just month and day format (MM-DD)
    # Convert dates to just month and day format for display, but keep the original datetime for plotting
    half_volume_dates_values = list(half_volume_dates.values())  # Extract the dates first
    formatted_dates = [date.timetuple().tm_yday if isinstance(date, pd.Timestamp) else None for date in half_volume_dates_values]
    # Create plot for center of volume date
    water_years = list(half_volume_dates.keys())
    half_volume_dates = list(half_volume_dates.values())
    # Convert dates to day of year for sorting and plotting
    # half_volume_dates is a list of datetime objects or possibly None
    formatted_dates = [date.timetuple().tm_yday if isinstance(date, pd.Timestamp) and pd.notna(date) else None for date in half_volume_dates]

    # Ensure to filter out None values (in case there are missing or invalid dates)
    valid_water_years = []
    valid_dates = []
    
    for year, day in zip(water_years, formatted_dates):
        # Ensure that we only use valid (not None) values
        if day is not None:
            valid_water_years.append(year)
            valid_dates.append(day)
    
    # Ensure that X and y have the same length and there are valid data points
    if len(valid_water_years) == 0 or len(valid_dates) == 0:
        print("No valid data points available for linear regression. Skipping this site.")
        continue


    #Calculates linear regression lines
    # Linear Regression Yearly Avg
    X = water_yearly_avg.index.values.reshape(-1, 1)  # Independent variable (water years)
    y = water_yearly_avg.values  # Dependent variable (average flow values)
    # Add a constant for the intercept
    X_const = sm.add_constant(X)
    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()
    # Predict values
    y_pred_water_yearly_avg = model.predict(X_const)
    # Get the slope and p-value
    slope_water_yearly_avg = model.params[1]  # The slope coefficient
    p_value_water_yearly_avg = model.pvalues[1]  # The p-value for the slope
    # Print the results
    print('\n',f'For Water Year Average of site {s}', f'Slope: {slope_water_yearly_avg}, P-value: {p_value_water_yearly_avg}')
    #Add values to lists
    p_values_yearly_avg.append(p_value_water_yearly_avg)
    slopes_yearly_avg.append(slope_water_yearly_avg)
    
    # Linear Regression Annual Max
    X = annual_max.index.values.reshape(-1, 1)  # Independent variable (water years)
    y = annual_max.values  # Dependent variable (average flow values)
    # Add a constant for the intercept
    X_const = sm.add_constant(X)
    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()
    # Predict values
    y_pred_annual_max = model.predict(X_const)
    # Get the slope and p-value
    slope_water_annual_max = model.params[1]  # The slope coefficient
    p_value_water_annual_max = model.pvalues[1]  # The p-value for the slope
    # Print the results
    print('\n',f'For Annual Maximum Flow of site {s}', f'Slope: {slope_water_annual_max}, P-value: {p_value_water_annual_max}')
    #Add values to lists
    p_values_max.append(p_value_water_annual_max)
    slopes_max.append(slope_water_annual_max)

    # Linear Regression 7-Day Low Flow
    X = seven_day_low_flow.index.values.reshape(-1, 1)  # Independent variable (water years)
    y = seven_day_low_flow.values  # Dependent variable (average flow values)
    # Add a constant for the intercept
    X_const = sm.add_constant(X)
    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()
    # Predict values
    y_pred_seven_day_low_flow = model.predict(X_const)
    # Get the slope and p-value
    slope_seven_day_low_flow = model.params[1]  # The slope coefficient
    p_value_seven_day_low_flow = model.pvalues[1]  # The p-value for the slope
    print('\n',f'For 7-Day Low Flow of site {s}', f'Slope: {slope_seven_day_low_flow}, P-value: {p_value_seven_day_low_flow}')
    #Add values to lists
    p_values_seven_day_low_flow.append(p_value_seven_day_low_flow)
    slopes_seven_day_low_flow.append(slope_seven_day_low_flow)
    


    # Create a plot for the information with linear regression lines
    plt.title(f'Post- 1970s Data for Site {s}', fontsize=16)
    plt.plot(water_yearly_avg.index, water_yearly_avg.values, label='Daily Average Flow', color='blue', marker='o')
    plt.plot(water_yearly_avg.index, y_pred_water_yearly_avg, label='Daily Avg Linear Regression Line', color='red')
    plt.plot(annual_max.index, annual_max.values, label='Yearly Max Flow', color='orange', marker='o')
    plt.plot(annual_max.index, y_pred_annual_max, label='Yearly Max Linear Regression Line', color='red')
    plt.plot(seven_day_low_flow.index, seven_day_low_flow.values, label='7 Day Low Flow', color='pink', marker='o')
    plt.plot(seven_day_low_flow.index, y_pred_seven_day_low_flow, label='7-Day Low Flow Linear Regression Line', color='red')
    plt.ylabel('Flow (units)')
    plt.legend()

    plt.show()

    
    # Linear Regression for Center Mass Volume Date
    X = np.array(valid_water_years).reshape(-1, 1)  # Independent variable (water years)
    y = np.array(valid_dates)  # Dependent variable (day of year values)
    # Add a constant for the intercept
    X_const = sm.add_constant(X)
    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()
    # Predict values
    y_Winter_Spring_Center_Volume_Date = model.predict(X_const)        
    # Get the slope and p-value
    slope_WSCV = model.params[1]  # The slope coefficient
    p_value_WSCV = model.pvalues[1]  # The p-value for the slope        
    # Print the results
    print('\n', f'For Winter-Spring Center Volume Date of site {s}', f'Slope: {slope_WSCV}, P-value: {p_value_WSCV}')        
    # Add values to lists
    p_values_WSCV.append(p_value_WSCV)
    slopes_WSCV.append(slope_WSCV)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(valid_water_years, valid_dates, color='blue', marker='o', linestyle='-', label='Observed Data')
    plt.plot(valid_water_years, y_Winter_Spring_Center_Volume_Date, label='Center Volume Date Regression Line', color='red', linestyle='--')
    plt.title(f'Post- 1970s Winter-Spring Center Volume Date for Site {s}', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Day of Year')
    plt.xticks(rotation=45)
    
    # Format the y-axis labels to show MM-DD
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    plt.legend()
    plt.show()

        # Calculate seasonal averages
    water_seasonly_avg = df.groupby([df['Season'], df.index.year])['Value/Valeur'].mean().reset_index()
    water_seasonly_avg.columns = ['Season', 'Year', 'Seasonal_Avg']
    # Ensure the Year column is converted to integers
    water_seasonly_avg['Year'] = water_seasonly_avg['Year'].astype(int)

 # Split the data into separate DataFrames for each season
    fall = water_seasonly_avg[water_seasonly_avg['Season'] == 'Fall']
    winter = water_seasonly_avg[water_seasonly_avg['Season'] == 'Winter']
    spring = water_seasonly_avg[water_seasonly_avg['Season'] == 'Spring']
    summer = water_seasonly_avg[water_seasonly_avg['Season'] == 'Summer']
    

   # Linear Regression for each season
##Fall

    X = fall['Year']  # Independent variable (year)
    y = fall['Seasonal_Avg']  # Dependent variable (average flow values)

    # Add a constant for the intercept
    X_const = sm.add_constant(X)

    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()

    # Get the slope and p-value
    p_value = model.pvalues.iloc[1]  # The p-value for the slope
    slope = model.params.iloc[1]  # The slope coefficient

    p_values_Fallavg.append(p_value)
    slopes_Fallavg.append(slope)
    y_pred_Fallavg = model.predict(X_const)


    # Plot the data and regression line
    plt.plot(X, y, label='Fall Seasonal Avg', color='peru')
    plt.plot(X, model.predict(X_const), label='Regression Line', color='saddlebrown')



##Winter
    # Linear Regression for each season
    X = winter['Year']  # Independent variable (year)
    y = winter['Seasonal_Avg']  # Dependent variable (average flow values)

    # Add a constant for the intercept
    X_const = sm.add_constant(X)

    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()

    # Get the slope and p-value
    p_value = model.pvalues.iloc[1]  # The p-value for the slope
    slope = model.params.iloc[1]  # The slope coefficient


    p_values_Winteravg.append(p_value)
    slopes_Winteravg.append(slope)
    y_pred_Winteravg = model.predict(X_const)

# Plot the data and regression 
    plt.plot(X, y, label='Winter Seasonal Avg', color='lightskyblue')
    plt.plot(X, model.predict(X_const), label='Regression Line', color='deepskyblue')
    

##Spring
    # Linear Regression for each season
    X = spring['Year']  # Independent variable (year)
    y = spring['Seasonal_Avg']  # Dependent variable (average flow values)

    # Add a constant for the intercept
    X_const = sm.add_constant(X)

    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()

    # Get the slope and p-value
    p_value = model.pvalues.iloc[1]  # The p-value for the slope
    slope = model.params.iloc[1]  # The slope coefficient

    p_values_Springavg.append(p_value)
    slopes_Springavg.append(slope)
    y_pred_Springavg = model.predict(X_const)

    # Plot the data and regression line
    plt.plot(X, y, label='Spring Seasonal Avg', color='pink')
    plt.plot(X, model.predict(X_const), label='Regression Line', color='palevioletred')

##Summer
    # Linear Regression for each season
    X = summer['Year']  # Independent variable (year)
    y = summer['Seasonal_Avg']  # Dependent variable (average flow values)

    # Add a constant for the intercept
    X_const = sm.add_constant(X)

    # Fit the model using statsmodels
    model = sm.OLS(y, X_const).fit()

    # Get the slope and p-value
    p_value = model.pvalues.iloc[1]  # The p-value for the slope
    slope = model.params.iloc[1]  # The slope coefficient

    p_values_Summeravg.append(p_value)
    slopes_Summeravg.append(slope)
    y_pred_Summeravg = model.predict(X_const)

    # Plot the data and regression line
    plt.plot(X, y, label='Summer Seasonal Avg', color='olive')
    plt.plot(X, model.predict(X_const), label='Regression Line', color='darkolivegreen')
  
    plt.title(f'Post- 1970s Seasonal Data with Linear Regression for site {s}')
    plt.xlabel('Year')
    plt.ylabel('Seasonal Avg Flow')
    plt.show()
```

    Index(['FID', 'ID', 'Name', 'Lat', 'Lon', 'lat_snap_90m', 'lon_snap_90m',
           'Country', 'Area(km2)', 'Derived_area(km2)', 'Area_error', 'Regulation',
           'Regulated_period', 'Reference', 'Q_period', 'Q_length', 'Hongren',
           'Tim', 'Amin', 'Frank', 'Flow_info1', 'Flow_info2', 'Lake_info',
           'Missing_data_periods', 'Observation_percent(%)', 'Objective_info',
           'Calibration/Validation', 'Watershed_NO', 'Most_downstream',
           'Q-t_check', 'Annual_runoff(mm)', 'Elevation(m)', 'SubId_version1_2',
           'DowSubId_version1_2', 'SubId_version3', 'DowSubId_version3',
           'Annual_runoff(mm).1'],
          dtype='object')
    
     For Water Year Average of site 02BF001 Slope: -0.011619102448565809, P-value: 0.7511039757843234
    
     For Annual Maximum Flow of site 02BF001 Slope: 0.438253968253967, P-value: 0.5088085048016653
    
     For 7-Day Low Flow of site 02BF001 Slope: -0.019627808699237435, P-value: 0.0721275799016675
    


    
![png](output_13_1.png)
    


    
     For Winter-Spring Center Volume Date of site 02BF001 Slope: -0.31572871572871586, P-value: 0.01812443414471823
    


    
![png](output_13_3.png)
    



    
![png](output_13_4.png)
    


    
     For Water Year Average of site 02BF002 Slope: -0.014571111537778889, P-value: 0.6541719425579047
    
     For Annual Maximum Flow of site 02BF002 Slope: -0.10337662337662168, P-value: 0.8472843113658852
    
     For 7-Day Low Flow of site 02BF002 Slope: -0.005587528344671521, P-value: 0.5907675526035966
    


    
![png](output_13_6.png)
    


    
     For Winter-Spring Center Volume Date of site 02BF002 Slope: -0.5490620490620499, P-value: 0.0024575240643498098
    


    
![png](output_13_8.png)
    



    
![png](output_13_9.png)
    


    
     For Water Year Average of site 02CF007 Slope: 0.0022050874618203173, P-value: 0.639482102561026
    
     For Annual Maximum Flow of site 02CF007 Slope: -0.0677371351766515, P-value: 0.5280567454692254
    
     For 7-Day Low Flow of site 02CF007 Slope: -0.00046148782093482035, P-value: 0.6064378697205128
    


    
![png](output_13_11.png)
    


    
     For Winter-Spring Center Volume Date of site 02CF007 Slope: -0.26588901689708117, P-value: 0.047328690473618665
    


    
![png](output_13_13.png)
    



    
![png](output_13_14.png)
    


    
     For Water Year Average of site 02EA005 Slope: 0.019156162173605815, P-value: 0.008061078281498013
    
     For Annual Maximum Flow of site 02EA005 Slope: 0.13870666923917901, P-value: 0.16235525132127485
    
     For 7-Day Low Flow of site 02EA005 Slope: -0.0015401357919204905, P-value: 0.5741580721045153
    


    
![png](output_13_16.png)
    


    
     For Winter-Spring Center Volume Date of site 02EA005 Slope: -0.16951250884301175, P-value: 0.10884840934116613
    


    
![png](output_13_18.png)
    



    
![png](output_13_19.png)
    


    
     For Water Year Average of site 02EC002 Slope: 0.07210849021010274, P-value: 0.009310419311848023
    
     For Annual Maximum Flow of site 02EC002 Slope: 0.21843205350826306, P-value: 0.2981794696461327
    
     For 7-Day Low Flow of site 02EC002 Slope: -0.0006471086794742542, P-value: 0.9011190464447785
    


    
![png](output_13_21.png)
    


    
     For Winter-Spring Center Volume Date of site 02EC002 Slope: -0.20604218920830894, P-value: 0.13678886921508915
    


    
![png](output_13_23.png)
    



    
![png](output_13_24.png)
    


    
     For Water Year Average of site 02EC011 Slope: 0.004291307512122429, P-value: 0.5426312860743283
    
     For Annual Maximum Flow of site 02EC011 Slope: -0.32547688848952516, P-value: 0.006282208208626357
    
     For 7-Day Low Flow of site 02EC011 Slope: 0.000994225158118547, P-value: 0.4599656653856027
    


    
![png](output_13_26.png)
    


    
     For Winter-Spring Center Volume Date of site 02EC011 Slope: -0.01649533537103881, P-value: 0.9529184322710347
    


    
![png](output_13_28.png)
    



    
![png](output_13_29.png)
    


    
     For Water Year Average of site 02ED003 Slope: 0.02674924064346107, P-value: 0.047674897162037225
    
     For Annual Maximum Flow of site 02ED003 Slope: -0.26026914914142063, P-value: 0.28675267393626613
    
     For 7-Day Low Flow of site 02ED003 Slope: 0.004609520134506106, P-value: 0.1655615849448305
    


    
![png](output_13_31.png)
    


    
     For Winter-Spring Center Volume Date of site 02ED003 Slope: -0.10127660942825825, P-value: 0.19300415176054728
    


    
![png](output_13_33.png)
    



    
![png](output_13_34.png)
    


    
     For Water Year Average of site 02ED101 Slope: -0.002604164348249954, P-value: 0.6563796489662184
    
     For Annual Maximum Flow of site 02ED101 Slope: -0.040423557867187675, P-value: 0.6551252533218407
    
     For 7-Day Low Flow of site 02ED101 Slope: 0.00143013407514345, P-value: 0.5310783502335734
    


    
![png](output_13_36.png)
    


    
     For Winter-Spring Center Volume Date of site 02ED101 Slope: -0.1570697694193709, P-value: 0.24785502462661624
    


    
![png](output_13_38.png)
    



    
![png](output_13_39.png)
    


    
     For Water Year Average of site 02ED102 Slope: 0.0019897253327547755, P-value: 0.560343925167412
    
     For Annual Maximum Flow of site 02ED102 Slope: -0.21586300408371292, P-value: 0.11731967289772306
    
     For 7-Day Low Flow of site 02ED102 Slope: 0.00017023672792251538, P-value: 0.9198468910385702
    


    
![png](output_13_41.png)
    


    
     For Winter-Spring Center Volume Date of site 02ED102 Slope: -0.3098328228688084, P-value: 0.3149585568285823
    


    
![png](output_13_43.png)
    



    
![png](output_13_44.png)
    


    
     For Water Year Average of site 02FE008 Slope: 0.013144685227178043, P-value: 0.4944050701610172
    
     For Annual Maximum Flow of site 02FE008 Slope: -0.7851571150680807, P-value: 0.03494960144933332
    
     For 7-Day Low Flow of site 02FE008 Slope: 0.0009807434642267225, P-value: 0.6325751215538032
    


    
![png](output_13_46.png)
    


    
     For Winter-Spring Center Volume Date of site 02FE008 Slope: -0.39155020200508694, P-value: 0.0938700397442736
    


    
![png](output_13_48.png)
    



    
![png](output_13_49.png)
    


    
     For Water Year Average of site 02FE009 Slope: 0.01361196900728012, P-value: 0.3356888132119282
    
     For Annual Maximum Flow of site 02FE009 Slope: 0.10928571428571479, P-value: 0.6870206038482308
    
     For 7-Day Low Flow of site 02FE009 Slope: 0.0021607915893628094, P-value: 0.02194200683296199
    


    
![png](output_13_51.png)
    


    
     For Winter-Spring Center Volume Date of site 02FE009 Slope: -0.3160894660894661, P-value: 0.5593665754853384
    


    
![png](output_13_53.png)
    



    
![png](output_13_54.png)
    


    
     For Water Year Average of site 02FE010 Slope: 0.011957817449688638, P-value: 0.21080056236983807
    
     For Annual Maximum Flow of site 02FE010 Slope: -0.25640406937436, P-value: 0.2166827470501571
    
     For 7-Day Low Flow of site 02FE010 Slope: 0.0003263468589417156, P-value: 0.2421409011737125
    


    
![png](output_13_56.png)
    


    
     For Winter-Spring Center Volume Date of site 02FE010 Slope: -0.6065630938062612, P-value: 0.16675548437650153
    


    
![png](output_13_58.png)
    



    
![png](output_13_59.png)
    


    
     For Water Year Average of site 02GA010 Slope: 0.03866564069024979, P-value: 0.013308150788004139
    
     For Annual Maximum Flow of site 02GA010 Slope: -0.2699626985658182, P-value: 0.5302161344281404
    
     For 7-Day Low Flow of site 02GA010 Slope: 0.006875361759598482, P-value: 0.010075811940257308
    


    
![png](output_13_61.png)
    


    
     For Winter-Spring Center Volume Date of site 02GA010 Slope: -0.22173451668917482, P-value: 0.06265957731880807
    


    
![png](output_13_63.png)
    



    
![png](output_13_64.png)
    


    
     For Water Year Average of site 02GA018 Slope: 0.012613547410713468, P-value: 0.19991347466157033
    
     For Annual Maximum Flow of site 02GA018 Slope: -0.3044060542476159, P-value: 0.4068599980074392
    
     For 7-Day Low Flow of site 02GA018 Slope: 0.0011596095817416417, P-value: 0.1980532400301004
    


    
![png](output_13_66.png)
    


    
     For Winter-Spring Center Volume Date of site 02GA018 Slope: -0.26649597645852746, P-value: 0.09168088585705084
    


    
![png](output_13_68.png)
    



    
![png](output_13_69.png)
    


    
     For Water Year Average of site 02GB007 Slope: 0.008821819067442862, P-value: 0.37498521875431623
    
     For Annual Maximum Flow of site 02GB007 Slope: -0.08933061224489686, P-value: 0.5903273614544673
    
     For 7-Day Low Flow of site 02GB007 Slope: 0.002340085748585192, P-value: 0.0665582900132534
    


    
![png](output_13_71.png)
    


    
     For Winter-Spring Center Volume Date of site 02GB007 Slope: -0.09795918367346829, P-value: 0.6594730611689257
    


    
![png](output_13_73.png)
    



    
![png](output_13_74.png)
    


    
     For Water Year Average of site 02GC002 Slope: -0.02138800917915719, P-value: 0.022808191808146396
    
     For Annual Maximum Flow of site 02GC002 Slope: 0.5002838952092589, P-value: 0.03354633790231235
    
     For 7-Day Low Flow of site 02GC002 Slope: -0.008134179253582396, P-value: 0.0024983279420679565
    


    
![png](output_13_76.png)
    


    
     For Winter-Spring Center Volume Date of site 02GC002 Slope: 0.1522805552656298, P-value: 0.7205932849922383
    


    
![png](output_13_78.png)
    



    
![png](output_13_79.png)
    


    
     For Water Year Average of site 02GC010 Slope: 0.017427856101609313, P-value: 0.025060027728678297
    
     For Annual Maximum Flow of site 02GC010 Slope: 0.02264547904829685, P-value: 0.9102142559470406
    
     For 7-Day Low Flow of site 02GC010 Slope: 0.0055353507092858155, P-value: 0.001980837285403467
    


    
![png](output_13_81.png)
    


    
     For Winter-Spring Center Volume Date of site 02GC010 Slope: -0.21893118225358477, P-value: 0.1745686824598516
    


    
![png](output_13_83.png)
    



    
![png](output_13_84.png)
    


    
     For Water Year Average of site 02GD004 Slope: 0.010026408124961288, P-value: 0.09734870969746581
    
     For Annual Maximum Flow of site 02GD004 Slope: 0.03521609106695078, P-value: 0.8341368559556608
    
     For 7-Day Low Flow of site 02GD004 Slope: 0.0018704665205847408, P-value: 0.027034295524038302
    


    
![png](output_13_86.png)
    


    
     For Winter-Spring Center Volume Date of site 02GD004 Slope: -0.22210431539005543, P-value: 0.4642806182603907
    


    
![png](output_13_88.png)
    



    
![png](output_13_89.png)
    


    
     For Water Year Average of site 02GG002 Slope: 0.016756139257320307, P-value: 0.14257324448301473
    
     For Annual Maximum Flow of site 02GG002 Slope: 0.2883031063090876, P-value: 0.22266938702874203
    
     For 7-Day Low Flow of site 02GG002 Slope: 0.0012328192901704357, P-value: 0.33791277438681666
    


    
![png](output_13_91.png)
    


    
     For Winter-Spring Center Volume Date of site 02GG002 Slope: -0.23921152485690128, P-value: 0.44231953152774695
    


    
![png](output_13_93.png)
    



    
![png](output_13_94.png)
    


    
     For Water Year Average of site 02GG006 Slope: 0.01019583025469106, P-value: 0.10364277534779218
    
     For Annual Maximum Flow of site 02GG006 Slope: 0.19002463054187227, P-value: 0.4445047999778625
    
     For 7-Day Low Flow of site 02GG006 Slope: 0.0001687241502772419, P-value: 0.18613969504300923
    


    
![png](output_13_96.png)
    


    
     For Winter-Spring Center Volume Date of site 02GG006 Slope: -0.06267824734249369, P-value: 0.8983611064189769
    


    
![png](output_13_98.png)
    



    
![png](output_13_99.png)
    


    
     For Water Year Average of site 02HA006 Slope: 0.0063169976653836895, P-value: 0.33359561069057686
    
     For Annual Maximum Flow of site 02HA006 Slope: -0.33120133597746876, P-value: 0.026328612564387826
    
     For 7-Day Low Flow of site 02HA006 Slope: 3.9032112166487966e-05, P-value: 0.45115374751694026
    


    
![png](output_13_101.png)
    


    
     For Winter-Spring Center Volume Date of site 02HA006 Slope: -0.5431583342031181, P-value: 0.04411015461856737
    


    
![png](output_13_103.png)
    



    
![png](output_13_104.png)
    


    
     For Water Year Average of site 02HC025 Slope: 0.004906484702628252, P-value: 0.22823233208795277
    
     For Annual Maximum Flow of site 02HC025 Slope: -0.12151191428293755, P-value: 0.19658489233574916
    
     For 7-Day Low Flow of site 02HC025 Slope: 0.003848974878353252, P-value: 0.0017878081188664509
    


    
![png](output_13_106.png)
    


    
     For Winter-Spring Center Volume Date of site 02HC025 Slope: 0.1932632727653239, P-value: 0.23586263642862887
    


    
![png](output_13_108.png)
    



    
![png](output_13_109.png)
    


    
     For Water Year Average of site 02HC030 Slope: 0.015480565788079447, P-value: 0.0016810051233479815
    
     For Annual Maximum Flow of site 02HC030 Slope: -0.04655160628844231, P-value: 0.7801158249324356
    
     For 7-Day Low Flow of site 02HC030 Slope: 0.002891724440972632, P-value: 7.206106698312897e-07
    


    
![png](output_13_111.png)
    


    
     For Winter-Spring Center Volume Date of site 02HC030 Slope: 0.20198222829801754, P-value: 0.3007290528577575
    


    
![png](output_13_113.png)
    



    
![png](output_13_114.png)
    


    
     For Water Year Average of site 02HL003 Slope: 0.030864315008576095, P-value: 0.00047045170879288125
    
     For Annual Maximum Flow of site 02HL003 Slope: 0.22010344695957626, P-value: 0.03477732057064806
    
     For 7-Day Low Flow of site 02HL003 Slope: -0.0011950420986043843, P-value: 0.49197346810396547
    


    
![png](output_13_116.png)
    


    
     For Winter-Spring Center Volume Date of site 02HL003 Slope: -0.4882620147345127, P-value: 0.0006979715708561423
    


    
![png](output_13_118.png)
    



    
![png](output_13_119.png)
    


    
     For Water Year Average of site 02HL004 Slope: 0.02387485021415476, P-value: 0.062133143864509105
    
     For Annual Maximum Flow of site 02HL004 Slope: -0.011089055998778685, P-value: 0.9296141741387975
    
     For 7-Day Low Flow of site 02HL004 Slope: 0.0013729025679057134, P-value: 0.30071031487617467
    


    
![png](output_13_121.png)
    


    
     For Winter-Spring Center Volume Date of site 02HL004 Slope: -0.41930373706913066, P-value: 0.002861973614437395
    


    
![png](output_13_123.png)
    



    
![png](output_13_124.png)
    


    
     For Water Year Average of site 02HL005 Slope: 0.014933896480487903, P-value: 0.0326807352485067
    
     For Annual Maximum Flow of site 02HL005 Slope: 0.024294810667815706, P-value: 0.7516864798172634
    
     For 7-Day Low Flow of site 02HL005 Slope: 0.0023106963785851906, P-value: 0.006009189852806878
    


    
![png](output_13_126.png)
    


    
     For Winter-Spring Center Volume Date of site 02HL005 Slope: -0.18890153495954745, P-value: 0.26358992316905594
    


    
![png](output_13_128.png)
    



    
![png](output_13_129.png)
    


    
     For Water Year Average of site 02JB013 Slope: 0.05373981757820534, P-value: 0.3514591174084837
    
     For Annual Maximum Flow of site 02JB013 Slope: 0.7066048742546065, P-value: 0.25896188222772254
    
     For 7-Day Low Flow of site 02JB013 Slope: -0.010305659468869785, P-value: 0.5471753730971993
    


    
![png](output_13_131.png)
    


    
     For Winter-Spring Center Volume Date of site 02JB013 Slope: -0.582123411978225, P-value: 0.023738956323013
    


    
![png](output_13_133.png)
    



    
![png](output_13_134.png)
    


    
     For Water Year Average of site 02JC008 Slope: 0.07567765847390573, P-value: 0.05933458091141985
    
     For Annual Maximum Flow of site 02JC008 Slope: 0.1190851915380356, P-value: 0.7906929008728725
    
     For 7-Day Low Flow of site 02JC008 Slope: 0.021856353290315503, P-value: 0.0073472543236347086
    


    
![png](output_13_136.png)
    


    
     For Winter-Spring Center Volume Date of site 02JC008 Slope: -0.2736039641700031, P-value: 0.01171672365973597
    


    
![png](output_13_138.png)
    



    
![png](output_13_139.png)
    


    
     For Water Year Average of site 02KD002 Slope: 0.03330758588443791, P-value: 0.03292732635078686
    
     For Annual Maximum Flow of site 02KD002 Slope: 0.22673135651318904, P-value: 0.10821457116164987
    
     For 7-Day Low Flow of site 02KD002 Slope: -0.021944161294120298, P-value: 0.00040052713543106325
    


    
![png](output_13_141.png)
    


    
     For Winter-Spring Center Volume Date of site 02KD002 Slope: -0.08620080744008413, P-value: 0.6069067402366031
    


    
![png](output_13_143.png)
    



    
![png](output_13_144.png)
    


    
     For Water Year Average of site 02LA007 Slope: 0.015089686839409436, P-value: 0.3349765920577399
    
     For Annual Maximum Flow of site 02LA007 Slope: -0.8119254958877649, P-value: 0.0031443766500886677
    
     For 7-Day Low Flow of site 02LA007 Slope: 0.0013690418596076954, P-value: 0.30299402964486927
    


    
![png](output_13_146.png)
    


    
     For Winter-Spring Center Volume Date of site 02LA007 Slope: -0.17150459603289592, P-value: 0.271365651000797
    


    
![png](output_13_148.png)
    



    
![png](output_13_149.png)
    


    
     For Water Year Average of site 02LB007 Slope: 0.0041595180822292625, P-value: 0.44464631762317475
    
     For Annual Maximum Flow of site 02LB007 Slope: -0.23608064827319963, P-value: 0.009211723360202307
    
     For 7-Day Low Flow of site 02LB007 Slope: -0.00016948877435074788, P-value: 0.6118468797972998
    


    
![png](output_13_151.png)
    


    
     For Winter-Spring Center Volume Date of site 02LB007 Slope: -0.1566017107209456, P-value: 0.09973629534250199
    


    
![png](output_13_153.png)
    



    
![png](output_13_154.png)
    


    
     For Water Year Average of site 02LB008 Slope: -0.20044780700110015, P-value: 7.321793175107942e-09
    
     For Annual Maximum Flow of site 02LB008 Slope: -0.44165209913440806, P-value: 0.12656555468776853
    
     For 7-Day Low Flow of site 02LB008 Slope: -0.022685713278742484, P-value: 6.294352540767892e-05
    


    
![png](output_13_156.png)
    


    
     For Winter-Spring Center Volume Date of site 02LB008 Slope: 0.21422730989370453, P-value: 0.4499077790683985
    


    
![png](output_13_158.png)
    



    
![png](output_13_159.png)
    


    
     For Water Year Average of site 02LD005 Slope: 0.02226490402213182, P-value: 0.61150522971401
    
     For Annual Maximum Flow of site 02LD005 Slope: -0.07943586811511721, P-value: 0.7676956138365923
    
     For 7-Day Low Flow of site 02LD005 Slope: 0.007484058918021137, P-value: 0.595904202185132
    


    
![png](output_13_161.png)
    


    
     For Winter-Spring Center Volume Date of site 02LD005 Slope: -0.3046693348580144, P-value: 0.031146695662292844
    


    
![png](output_13_163.png)
    



    
![png](output_13_164.png)
    



```python
latitude=[]
longitude=[]

sites=us_site_list_1970

for s in sites:
    LocTest= nwis.get_record(sites=s, service='site')
    Lat=LocTest['dec_lat_va'].values[0]
    Long=LocTest['dec_long_va'].values[0]
    latitude.append(Lat)
    longitude.append(Long)

sites=ca_site_list_1970

# get_gauge_coordinates(s)
for s in sites:
    latitude.append(dfSites[dfSites['ID'] == s]['Lat'].item())
    longitude.append(dfSites[dfSites['ID'] == s]['Lon'].item())

print(latitude)
print(longitude)
```

    [46.58410589, 45.5, 45.18997026, 44.43639239, 44.19334009, 42.1222222, 41.50449567, 40.9228332, 44.6769593, 41.92699898, 43.74680556, 45.9430235, 43.9450063, 44.45833125, 45.7549674, 46.03052857, 46.5113361, 46.5745833, 43.46417856, 46.4866144, 46.58410589, 45.5, 45.18997026, 44.43639239, 44.19334009, 42.1222222, 41.50449567, 40.9228332, 44.6769593, 41.92699898, 43.74680556, 45.9430235, 43.9450063, 44.45833125, 45.7549674, 46.03052857, 46.5113361, 46.5745833, 43.46417856, 46.4866144, 46.58410589, 45.5, 45.18997026, 44.43639239, 44.19334009, 42.1222222, 41.50449567, 40.9228332, 44.6769593, 41.92699898, 43.74680556, 45.9430235, 43.9450063, 44.45833125, 45.7549674, 46.03052857, 46.5113361, 46.5745833, 43.46417856, 46.4866144, 46.58410589, 45.5, 45.18997026, 44.43639239, 44.19334009, 42.1222222, 41.50449567, 40.9228332, 44.6769593, 41.92699898, 43.74680556, 45.9430235, 43.9450063, 44.45833125, 45.7549674, 46.03052857, 46.5113361, 46.5745833, 43.46417856, 46.4866144, 46.58410589, 45.5, 45.18997026, 44.43639239, 44.19334009, 42.1222222, 41.50449567, 40.9228332, 44.6769593, 41.92699898, 43.74680556, 45.9430235, 43.9450063, 47.00352859, 46.86093903, 46.58335876, 45.66947174, 44.71366882, 44.39707947, 44.24980927, 44.11058044, 44.15250015, 43.8128891, 43.6843605, 43.6763382, 43.18972015, 43.37722015, 43.14738846, 42.77769089, 42.85731125, 43.05910873, 42.83081055, 42.90583038, 43.13346863, 43.81130981, 43.60174942, 44.53960037, 44.54956818, 44.49972916, 48.36667, 47.88914, 45.05214, 45.24942, 44.84225, 45.426, 45.79167]
    [-88.5759697, -88.0, -88.7334398, -85.6986792, -85.7697863, -77.9572222, -84.4296719, -83.3488116, -84.2925146, -80.6039659, -75.3334444, -86.7056997, -86.2786896, -87.5564746, -87.2020793, -86.161249, -90.0746179, -85.26963889, -86.2325668, -90.696297, -88.5759697, -88.0, -88.7334398, -85.6986792, -85.7697863, -77.9572222, -84.4296719, -83.3488116, -84.2925146, -80.6039659, -75.3334444, -86.7056997, -86.2786896, -87.5564746, -87.2020793, -86.161249, -90.0746179, -85.26963889, -86.2325668, -90.696297, -88.5759697, -88.0, -88.7334398, -85.6986792, -85.7697863, -77.9572222, -84.4296719, -83.3488116, -84.2925146, -80.6039659, -75.3334444, -86.7056997, -86.2786896, -87.5564746, -87.2020793, -86.161249, -90.0746179, -85.26963889, -86.2325668, -90.696297, -88.5759697, -88.0, -88.7334398, -85.6986792, -85.7697863, -77.9572222, -84.4296719, -83.3488116, -84.2925146, -80.6039659, -75.3334444, -86.7056997, -86.2786896, -87.5564746, -87.2020793, -86.161249, -90.0746179, -85.26963889, -86.2325668, -90.696297, -88.5759697, -88.0, -88.7334398, -85.6986792, -85.7697863, -77.9572222, -84.4296719, -83.3488116, -84.2925146, -80.6039659, -75.3334444, -86.7056997, -86.2786896, -84.51555634, -83.97180939, -81.19911194, -79.37918854, -79.28160858, -79.0708313, -79.82141876, -79.89028168, -79.89663696, -81.3068924, -81.54116821, -81.07485199, -80.45503235, -80.7108078, -80.15460968, -81.21399689, -80.72357941, -80.99485779, -81.85172272, -82.11911011, -79.38324738, -79.62757874, -79.55632782, -77.36965179, -77.32813263, -77.61836243, -78.8533, -79.8793, -77.846, -75.7906, -75.5444, -75.1532, -75.0914]
    

## Create Maps


```python
site_list=us_site_list_1970 + ca_site_list_1970
## Create a matrix to store all of the info below
# dictionary of lists 
dict = {'Site Numbers': site_list, 'Latitude': latitude, 'Longitude': longitude, 
        'P-value Yearly Average Flow': p_values_yearly_avg, 'Slope Yearly Average Flow':slopes_yearly_avg,
        'P-value Max Flow':p_values_max,'Slope Max Flow':slopes_max,
        'P-value 7 Day Low Flow':p_values_seven_day_low_flow,'Slope 7 Day Low Flow':slopes_seven_day_low_flow,
        'P-value WSCV':p_values_WSCV,'Slope WSCV':slopes_WSCV,
        'P-value Fall Average': p_values_Fallavg, 'Slope Yearly Fall Average':slopes_Fallavg,
        'P-value Winter Average': p_values_Winteravg, 'Slope Yearly Winter Average':slopes_Winteravg,
        'P-value Spring Average': p_values_Springavg, 'Slope Yearly Spring Average':slopes_Springavg,
        'P-value Summer Average': p_values_Summeravg, 'Slope Yearly Summer Average':slopes_Summeravg} 

df_1970 = pd.DataFrame(dict)

df_1970
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Site Numbers</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>P-value Yearly Average Flow</th>
      <th>Slope Yearly Average Flow</th>
      <th>P-value Max Flow</th>
      <th>Slope Max Flow</th>
      <th>P-value 7 Day Low Flow</th>
      <th>Slope 7 Day Low Flow</th>
      <th>P-value WSCV</th>
      <th>Slope WSCV</th>
      <th>P-value Fall Average</th>
      <th>Slope Yearly Fall Average</th>
      <th>P-value Winter Average</th>
      <th>Slope Yearly Winter Average</th>
      <th>P-value Spring Average</th>
      <th>Slope Yearly Spring Average</th>
      <th>P-value Summer Average</th>
      <th>Slope Yearly Summer Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>04040500</td>
      <td>46.584106</td>
      <td>-88.575970</td>
      <td>1.238372e-01</td>
      <td>-0.649524</td>
      <td>0.599794</td>
      <td>-3.935620</td>
      <td>0.059783</td>
      <td>-0.190215</td>
      <td>0.162295</td>
      <td>-0.138212</td>
      <td>0.289686</td>
      <td>-0.836804</td>
      <td>0.792583</td>
      <td>0.086896</td>
      <td>0.226602</td>
      <td>-1.453729</td>
      <td>0.108683</td>
      <td>-0.737168</td>
    </tr>
    <tr>
      <th>1</th>
      <td>04066500</td>
      <td>45.500000</td>
      <td>-88.000000</td>
      <td>2.414495e-02</td>
      <td>3.392505</td>
      <td>0.152524</td>
      <td>9.539231</td>
      <td>0.023725</td>
      <td>1.674692</td>
      <td>0.131457</td>
      <td>-1.133846</td>
      <td>0.332455</td>
      <td>1.697717</td>
      <td>0.001316</td>
      <td>3.166834</td>
      <td>0.078319</td>
      <td>5.315140</td>
      <td>0.091305</td>
      <td>3.328921</td>
    </tr>
    <tr>
      <th>2</th>
      <td>04074950</td>
      <td>45.189970</td>
      <td>-88.733440</td>
      <td>6.642952e-01</td>
      <td>-0.373475</td>
      <td>0.877380</td>
      <td>0.559482</td>
      <td>0.259861</td>
      <td>-0.497667</td>
      <td>0.594036</td>
      <td>-0.125024</td>
      <td>0.342814</td>
      <td>-0.953179</td>
      <td>0.217537</td>
      <td>-0.720996</td>
      <td>0.906749</td>
      <td>0.198010</td>
      <td>0.423357</td>
      <td>0.844939</td>
    </tr>
    <tr>
      <th>3</th>
      <td>04124000</td>
      <td>44.436392</td>
      <td>-85.698679</td>
      <td>2.705259e-01</td>
      <td>0.838384</td>
      <td>0.238322</td>
      <td>-4.557271</td>
      <td>0.095786</td>
      <td>0.796101</td>
      <td>0.343114</td>
      <td>0.237774</td>
      <td>0.497209</td>
      <td>0.891642</td>
      <td>0.006854</td>
      <td>2.294331</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>04125460</td>
      <td>44.193340</td>
      <td>-85.769786</td>
      <td>6.450043e-01</td>
      <td>-0.118871</td>
      <td>0.683032</td>
      <td>-1.615587</td>
      <td>0.003171</td>
      <td>-0.385323</td>
      <td>0.102827</td>
      <td>-0.134696</td>
      <td>0.970094</td>
      <td>0.011597</td>
      <td>0.242333</td>
      <td>0.396432</td>
      <td>0.366599</td>
      <td>-0.519323</td>
      <td>0.186078</td>
      <td>-0.346672</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>121</th>
      <td>02KD002</td>
      <td>45.052140</td>
      <td>-77.846000</td>
      <td>3.292733e-02</td>
      <td>0.033308</td>
      <td>0.108215</td>
      <td>0.226731</td>
      <td>0.000401</td>
      <td>-0.021944</td>
      <td>0.606907</td>
      <td>-0.086201</td>
      <td>0.229698</td>
      <td>0.033173</td>
      <td>0.000845</td>
      <td>0.071858</td>
      <td>0.159497</td>
      <td>0.050871</td>
      <td>0.669489</td>
      <td>-0.007195</td>
    </tr>
    <tr>
      <th>122</th>
      <td>02LA007</td>
      <td>45.249420</td>
      <td>-75.790600</td>
      <td>3.349766e-01</td>
      <td>0.015090</td>
      <td>0.003144</td>
      <td>-0.811925</td>
      <td>0.302994</td>
      <td>0.001369</td>
      <td>0.271366</td>
      <td>-0.171505</td>
      <td>0.930340</td>
      <td>0.002206</td>
      <td>0.195012</td>
      <td>0.040006</td>
      <td>0.696700</td>
      <td>-0.016946</td>
      <td>0.152813</td>
      <td>0.028621</td>
    </tr>
    <tr>
      <th>123</th>
      <td>02LB007</td>
      <td>44.842250</td>
      <td>-75.544400</td>
      <td>4.446463e-01</td>
      <td>0.004160</td>
      <td>0.009212</td>
      <td>-0.236081</td>
      <td>0.611847</td>
      <td>-0.000169</td>
      <td>0.099736</td>
      <td>-0.156602</td>
      <td>0.478826</td>
      <td>0.005790</td>
      <td>0.001772</td>
      <td>0.033733</td>
      <td>0.074683</td>
      <td>-0.023924</td>
      <td>0.223245</td>
      <td>0.005095</td>
    </tr>
    <tr>
      <th>124</th>
      <td>02LB008</td>
      <td>45.426000</td>
      <td>-75.153200</td>
      <td>7.321793e-09</td>
      <td>-0.200448</td>
      <td>0.126566</td>
      <td>-0.441652</td>
      <td>0.000063</td>
      <td>-0.022686</td>
      <td>0.449908</td>
      <td>0.214227</td>
      <td>0.604120</td>
      <td>0.012603</td>
      <td>0.672863</td>
      <td>0.011749</td>
      <td>0.001778</td>
      <td>-0.173122</td>
      <td>0.347469</td>
      <td>0.016901</td>
    </tr>
    <tr>
      <th>125</th>
      <td>02LD005</td>
      <td>45.791670</td>
      <td>-75.091400</td>
      <td>6.115052e-01</td>
      <td>0.022265</td>
      <td>0.767696</td>
      <td>-0.079436</td>
      <td>0.595904</td>
      <td>0.007484</td>
      <td>0.031147</td>
      <td>-0.304669</td>
      <td>0.146178</td>
      <td>0.104263</td>
      <td>0.012074</td>
      <td>0.104437</td>
      <td>0.286905</td>
      <td>-0.097269</td>
      <td>0.973342</td>
      <td>-0.001896</td>
    </tr>
  </tbody>
</table>
<p>126 rows × 19 columns</p>
</div>




```python
df=df_1970

for i in range(3, 19, 2):  # Starts at index 3 (for column 4), ends at 11, step by 2
    col_P = df.iloc[:, i]     # Column i (e.g., 4)
    col_S = df.iloc[:, i + 1] # Column i+1 (e.g., 5)

    # Grab column names
    colP_name = df.columns[i]
    colS_name = df.columns[i + 1]

    # Create a GeoDataFrame with Points from Latitude and Longitude
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    
    # Set the Coordinate Reference System (CRS) to WGS 84 (EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)
    
    # Create a CartoPy map with a focus on New York State
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add the boundaries of U.S. States using Cartopy
    ax.add_feature(cfeature.STATES, edgecolor='black', zorder=1)
    
    # Zoom in to New York State
    ax.set_extent([-94, -71, 39, 50], crs=ccrs.PlateCarree())
    
    # Add water bodies (lakes, rivers, oceans) and color them
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='lightblue', zorder=0)  # Lakes
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', zorder=0)  # Rivers
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)  # Oceans
    

    for idx, row in df.iterrows():
      
        if  row[colS_name] > 0:
            # Positive slope, plot as an upwards triangle ('^')
            mark='^'
            color='blue'
            edgecolors='blue'
        else:       
            # Negative slope, plot as a downwards triangle ('^')
            mark='v'
            color='red'
            edgecolors='red'
        if row[colP_name] > 0.05:
            color='none'
        #plot each point
        sc = ax.scatter(row['Longitude'], row['Latitude'], color=color, edgecolors=edgecolors , marker=mark, s=100, transform=ccrs.PlateCarree())
    
    
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='Positive Slope', markerfacecolor='black', markersize=10),
        Line2D([0], [0], marker='v', color='w', label='Negative Slope', markerfacecolor='black', markersize=10)
    ]
    
    # Add the legend to the map
    ax.legend(handles=legend_elements, loc='upper left', title='Slope Legend')

    # Split the string into words
    words = colS_name.split()
    # Grab all but the first word
    map_type = ' '.join(words[1:])  # Join the remaining words back into a string

    
    # Set plot title
    plt.title(f'Post-1970s P-value and Slope {map_type}')
    plt.show()
```


    
![png](output_17_0.png)
    



    
![png](output_17_1.png)
    



    
![png](output_17_2.png)
    



    
![png](output_17_3.png)
    



    
![png](output_17_4.png)
    



    
![png](output_17_5.png)
    



    
![png](output_17_6.png)
    



    
![png](output_17_7.png)
    



```python

```
