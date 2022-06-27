#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import click


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)
    

def read_data(filename):
    print(f"Reading data from {filename}")
    categorical = ['PUlocationID', 'DOlocationID']
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    dicts = df[categorical].to_dict(orient='records')
    
    return df, dicts

def save_prediction(df, y_pred):
    year = df["pickup_datetime"].dt.year.astype('int').values[0]
    month = df["pickup_datetime"].dt.month.astype('int').values[0]

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    print(f"Generating dataframe from prediction with ride-id")
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predictions"] = y_pred

    taxi_type = "fvh"
    output_file = f"output/{taxi_type}-{year}-{month}_prediction.parquet"

    print(f"Saving prediction output to {output_file}")
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


@click.command()
@click.option('--year', default=2021, help='Year of the taxi data')
@click.option('--month', default=3, help='Month of the taxi data')
def predict(year, month):
    file = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'

    df, dicts = read_data(file)

    print("Making prediction")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print("Prediction Mean:", y_pred.mean())

    # save_prediction(df, y_pred)


if __name__ == "__main__":
    predict()