import pandas as pd
import argparse

def create_shp_data(date):
    shp_folder = '/home/datacube-epidemics/Documents/github/noa_epidemics_data/epidemics_data/input_data_shp/'
    output_folder = '/mnt/epidemics_storage/mosq_predictions_data/models/'
    for case in ['FR_culex', 'FR_aedes', 'GRE', 'GER', 'SER_vojvodina', 'IT_veneto', 'IT_trentino', 'IC']:
        output_filename = case+'_env_'+date[:-3].replace("-",'_')+'_2km.csv'
        data = pd.read_csv(shp_folder+case+'_shapefile_2km.csv')
        data['dt_placement'] = pd.to_datetime(date, format="%Y-%m-%d")
        data.to_csv(output_folder+output_filename, index=False)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--date',
            required=True
    )

    args = parser.parse_args()

    create_shp_data(args.date)


if __name__ == '__main__':
    main()
