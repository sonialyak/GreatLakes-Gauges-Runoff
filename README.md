# GreatLakes-Gauges-Runoff

## Introduction
This project aims to do an analysis of runoff in the Great Lakes Region in order to predict future patterns in water levels in the lakes. For more information read the final write up of this project.

## How to Use
Use these files for analyzing runoff from gauges in the US from USGS or from the Canada water office. 

The list of the natural gauges in the Great Lakes Region as used for this project can be found in [(gauge_info.xls)](https://github.com/sonialyak/GreatLakes-Gauges-Runoff/blob/main/gauge_info.xls)
For retrieving the data for the USGS sites, you can use the code as is on (GreatLakes-Runoff-AllNaturalGauges.md); however, to retrieve data for the Canadian sites, you must first download the data using Download (Canada Data.ipynb) into a folder, then access the data using the main code. 

This data analyzes by water year as defined. It finds yearly average flow, max flow, 7-day low flow, winter-spring half volume date, and averages by each season. It then graphs this data, creates a linear regression for each, and calculates significance. It also graphs the significance and positive/negative changes on a map of the Great Lakes Region.
