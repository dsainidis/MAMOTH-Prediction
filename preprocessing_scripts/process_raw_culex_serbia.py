import pandas as pd
import argparse
import datetime

def insert_data(input_file):
    
    dataframe = pd.read_csv(input_file)
    dataframe.columns = [c.lower() for c in dataframe.columns]
    dataframe.rename(columns={'latitude':'y',
                              'longitude': 'x',
                              'sampling day':'dt_placement',
                              'cx. pipiens no of adults in trap ':'culex.spp',
                              'location':'station_id'}, inplace=True)
    dataframe['x'] = dataframe.x.apply(lambda x: round(x, 6))
    dataframe['y'] = dataframe.y.apply(lambda y: round(y, 6))
    dataframe = dataframe.dropna(how='all').reset_index(drop=True)
    dataframe['dt_placement'] = dataframe['dt_placement'].str.strip()
    dataframe['dt_placement'] = dataframe['dt_placement'].str.split(' ').apply(lambda x : pd.to_datetime(str(x[2])+'-'+str(x[1])+'-'+str(x[0].split('-')[0])))
    dataframe['week'] = dataframe['dt_placement'].dt.isocalendar().week
    dataframe['region'] = 'Vojvodina'
    
    #save csv file to be used for the database
    dataframe.to_csv('/mnt/epidemics_storage/mosq_predictions_data/db/'+input_file.split('/')[-1], index=False)
    
    #save csv file to be used for the mamoth model trainning
    dataframe['dt_placement_original'] = dataframe['dt_placement']
    dataframe['dt_placement'] = dataframe['dt_placement'] - datetime.timedelta(days=15)
    dataframe.to_csv('/mnt/epidemics_storage/mosq_predictions_data/models/'+input_file.split('/')[-1], index=False)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input-file',
            required=True
    )

    args = parser.parse_args()

    insert_data(args.input_file)


if __name__ == '__main__':
    main()
