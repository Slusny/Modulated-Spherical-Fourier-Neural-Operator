"""
    Download relative humidity data from ERA5 reanalysis dataset using the Copernicus API.
"""

import cdsapi

c = cdsapi.Client()
for level in [250]: #[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    for year in range(1979,2019):
        folder = f'./multi_pressure_level/relative_humidity/{level}/'
        file_name = f'relative_humidity_{year}_{level}_.nc'
        file = folder + file_name
        c.retrieve(
            'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': 'relative_humidity',
                    'year': [str(year)],
                    'pressure_level': level,
                    'month': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                    ],
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', 
                        '06:00',
                        '12:00',
                        '18:00', 
                    ],
                },
                file)