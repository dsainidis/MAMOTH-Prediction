import pandas as pd
import argparse

def insert_data(input_file):

    dataframe = pd.read_csv(input_file, skiprows=[0], encoding="ISO-8859-1")
    dataframe['x'] = dataframe.x.apply(lambda x: round(x, 6))
    dataframe['y'] = dataframe.y.apply(lambda y: round(y, 6))
    dataframe.columns = [c.lower() for c in dataframe.columns]
    dataframe['dt_placement'] = pd.to_datetime(dataframe.dt_placement, format='%m/%d/%Y')
    dataframe = dataframe.dropna(how='all').reset_index(drop=True)
    dataframe['culex.spp'] = dataframe['culex spp.']
    dataframe['culex.pipiens'] = dataframe['culex pipiens']
    dataframe.drop(columns=['culex pipiens', 'culex spp.'], inplace=True)
    dataframe['region'] = 'Upper Rhine'

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
