import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shutil
# Hourly Prediction  (Use Dataset as it is)
# Daily Prediction   (Reorder Based on Date)
# Weekly Prediction  (Reorder Based on Weeks)
# Monthly Prediction (Reorder Based on Months)
# Yearly Prediction  (Reorder Based on Years)

def Train_Test_Split_OnYear(
        path_to_file:str,
        key_to_split:str,
        value_to_split:int 
    ) -> tuple:
    data = pd.read_csv(path_to_file)
    train_set = data[data[key_to_split] < value_to_split]
    test_set  = data[data[key_to_split] >= value_to_split]
    return train_set, test_set

def Apply_Scaling(
        path_to_file: str, 
        features_to_scale: list, 
        scaler_type: str = 'standard', 
    ) -> tuple:
    
    # Load the dataset
    df = pd.read_csv(path_to_file)
    Original_order = df.columns.tolist()
    
    # Separate features to scale and other columns
    features = df[features_to_scale]
    other_columns = df.drop(columns=features_to_scale)
    
    # Apply the scaler to the features
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Convert the scaled features back to a DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=features_to_scale)
    
    # Combine the scaled features with other columns
    scaled_data = pd.concat([other_columns, scaled_features_df], axis=1)
    df_reordered = scaled_data[Original_order]

    return df_reordered, scaler

def Construct_Hourly_CSV(
        path_to_file:str,
        ctr_code:str,
        cols_to_drp:list = ['Date', 'DayOfYear', 'WeekOfYear', 'Quarter'],
        ff_format:str='%.5f'
    ) -> None:
    df = pd.read_csv(path_to_file)
    df.drop(columns=cols_to_drp, inplace=True)

    df['ID'] = range(len(df))
    df = df[['ID'] + [col for col in df.columns if col != 'ID']]
    df.to_csv(os.path.join(os.path.dirname(path_to_file), f'{ctr_code}_Hourly.csv'), index=False, float_format=ff_format)

def Construct_Daily_CSV(
        path_to_file:str,
        ctr_code:str,
        cols_to_drp:list = ['Date', 'DayOfYear', 'WeekOfYear', 'Quarter'],
        ff_format:str='%.5f'
    ) -> None:
    df = pd.read_csv(path_to_file)
    df.drop(columns=cols_to_drp, inplace=True)

    aggregated_data = df.groupby(['Year', 'Month', 'Day']).agg({
    'Weekday': 'first',
    'IsWeekend': 'first',
    'IsHoliday': 'first',
    'Temperature': 'mean',
    'Irrad_direct': 'mean',
    'Irrad_difuse': 'mean',
    'Load': 'mean'}).reset_index()

    aggregated_data['ID'] = range(len(aggregated_data))
    aggregated_data = aggregated_data[['ID'] + [col for col in aggregated_data.columns if col != 'ID']]
    aggregated_data.to_csv(os.path.join(os.path.dirname(path_to_file), f'{ctr_code}_Daily.csv'), index=False, float_format=ff_format)

def Construct_Weekly_CSV(
        path_to_file:str,
        ctr_code:str,
        cols_to_drp:list = ['Date', 'DayOfYear', 'Quarter'],
        ff_format:str='%.5f'
    ) -> None:
    df = pd.read_csv(path_to_file)
    df.drop(columns=cols_to_drp, inplace=True)

    aggregated_data = df.groupby(['Year', 'Month', 'Day', 'WeekOfYear']).agg({
    'IsWeekend': 'first',
    'IsHoliday': 'first',
    'Temperature': 'mean',
    'Irrad_direct': 'mean',
    'Irrad_difuse': 'mean',
    'Load': 'mean'}).reset_index()
    aggregated_data.drop(columns=['WeekOfYear'], inplace=True)

    aggregated_data['ID'] = range(len(aggregated_data))
    aggregated_data = aggregated_data[['ID'] + [col for col in aggregated_data.columns if col != 'ID']]
    aggregated_data.to_csv(os.path.join(os.path.dirname(path_to_file), f'{ctr_code}_Weekly.csv'), index=False, float_format=ff_format)

def Construct_Monthly_CSV(
        path_to_file:str,
        ctr_code:str,
        cols_to_drp:list = ['Date', 'DayOfYear', 'WeekOfYear', 'Quarter'],
        ff_format:str='%.5f'
    ) -> None:
    df = pd.read_csv(path_to_file)
    df.drop(columns=cols_to_drp, inplace=True)

    aggregated_data = df.groupby(['Year', 'Month']).agg({
    'IsWeekend': 'first',
    'IsHoliday': 'first',
    'Temperature': 'mean',
    'Irrad_direct': 'mean',
    'Irrad_difuse': 'mean',
    'Load': 'mean'}).reset_index()

    aggregated_data['ID'] = range(len(aggregated_data))
    aggregated_data = aggregated_data[['ID'] + [col for col in aggregated_data.columns if col != 'ID']]
    aggregated_data.to_csv(os.path.join(os.path.dirname(path_to_file), f'{ctr_code}_Monthly.csv'), index=False, float_format=ff_format)


def Copy_CSVs_For_Dataset(
        source_dir:str, 
        target_dir:str, 
    )-> None:
    for filename in os.listdir(source_dir):
        name, extension = os.path.splitext(filename)
        new_filename = f'{name}_base{extension}'
        shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, new_filename))
    print("Files Are Recieved!")

def Manage_CSVs(
        path_to_file:str,
        ctr_code: list,
        ff_format:str
) -> None:
    for str_code in ctr_code:    
        Construct_Hourly_CSV(os.path.join(path_to_file, f'{str_code}_base.csv'), str_code, ff_format=ff_format)
        Construct_Daily_CSV(os.path.join(path_to_file, f'{str_code}_base.csv'), str_code, ff_format=ff_format)
        Construct_Weekly_CSV(os.path.join(path_to_file, f'{str_code}_base.csv'), str_code, ff_format=ff_format)
        Construct_Monthly_CSV(os.path.join(path_to_file, f'{str_code}_base.csv'), str_code, ff_format=ff_format)
        os.remove(os.path.join(path_to_file, f'{str_code}_base.csv'))


if __name__ == "__main__":
    # Important Paths
    path_processed = "../../data/processed"
    path_interim = "../../data/interim"
    ctr_code = ['ES', 'PT', 'PL', 'FR', 'SE']
    features_to_scale = ['Temperature', 'Irrad_direct', 'Irrad_difuse', 'Load']
    ff_format = '%.4f'
    Copy_CSVs_For_Dataset(path_interim, path_processed)
    Manage_CSVs(path_processed, ctr_code, ff_format)



