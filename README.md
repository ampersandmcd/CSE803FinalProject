# CSE 803 Final Project

A space for code and data associated with the final project in CSE 803 with Professor Xiaoming Liu @ MSU, Fall 2021.

#### Installation
Using conda for the larger packages
```bash
conda env create -f environment.yml
conda activate CVProj
```

If you have to make changes to the environment you can update the configuration with
```
conda env update -f environment.yml
```

#### Data

Download the `era5.nc` data file [here](https://drive.google.com/file/d/1WWSbjyY0h3MZQJI6lzvWu4bLwsP8dZUN/view?usp=sharing),
and save it into the root directory of this repository to ensure all code is able to find the data.
This file contains monthly average surface temperature (t2m) and precipitation (tp)
on a 0.1 degree grid over the United States from 1950-2020, and was extracted from the
[ERA5 Reanalysis Dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview).

#### Group Members
- Andrew McDonald
- Drew Hayward
- Tyler Lovell
- Sanjeev Thenkarai Lakshmi Narasimhan