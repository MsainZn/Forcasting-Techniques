import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shutil
import numpy as np
import pickle

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
    train_set = data[data[key_to_split] < value_to_split].copy()
    test_set  = data[data[key_to_split] >= value_to_split].copy()
    # print(f'Trainset-Length: {len(train_set)} Testset-Length:{len(test_set)}')
    train_set.drop(columns=['Year'], inplace=True)    
    test_set.drop(columns=['Year'], inplace=True)    
    return train_set, test_set

def Apply_Scaling(
        train_set: pd.DataFrame, 
        test_set: pd.DataFrame, 
        scalable_features: list, 
        scaler_type: str = 'standard', 
    ) -> tuple:

    Original_order = train_set.columns.tolist()
    # Separate features to scale and other columns
    train_set_scalable    = train_set[scalable_features]
    test_set_scalable     = test_set[scalable_features]
    train_set_categorical = train_set.drop(columns=scalable_features)
    test_set_categorical  = test_set.drop(columns=scalable_features)
    # Apply the scaler to the features
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    train_set_scaled = scaler.fit_transform(train_set_scalable)
    test_set_scaled  = scaler.transform(test_set_scalable)
    # Convert the scaled features back to a DataFrame
    train_set_scaled_features = pd.DataFrame(train_set_scaled, columns=scalable_features)
    test_set_scaled_features  = pd.DataFrame(test_set_scaled,  columns=scalable_features)
    # Combine the scaled features with other columns
    train_set_scaled_data = pd.concat([train_set_categorical.reset_index(drop=True), train_set_scaled_features.reset_index(drop=True)], axis=1)
    test_set_scaled_data = pd.concat([test_set_categorical.reset_index(drop=True),   test_set_scaled_features.reset_index(drop=True)], axis=1)

    # Reordering Dataframe as Before
    train_set_reordered = train_set_scaled_data[Original_order]
    test_set_reordered = test_set_scaled_data[Original_order]

    return train_set_reordered, test_set_reordered, scaler

def Construct_Hourly_CSV(
        path_to_file:str,
        ctr_code:str,
        cols_to_drp:list,
        ff_format:str='%.5f',
        include_ID:bool=False
    ) -> None:
    df = pd.read_csv(path_to_file)
    df.drop(columns=cols_to_drp, inplace=True)

    if include_ID:
        df['ID'] = [f'ID-{i}' for i in range(len(df))] 
        df = df[['ID'] + [col for col in df.columns if col != 'ID']]

    df.to_csv(os.path.join(os.path.dirname(path_to_file), f'{ctr_code}_Hourly.csv'), index=False, float_format=ff_format)

def Construct_Daily_CSV(
        path_to_file:str,
        ctr_code:str,
        cols_to_drp:list,
        ff_format:str='%.5f',
        include_ID:bool=False
    ) -> None:
    df = pd.read_csv(path_to_file)
    df.drop(columns=cols_to_drp, inplace=True)

    aggregated_data = df.groupby(['Year', 'Month', 'Day']).agg({
    'WeekOfMonth': 'first',
    'Weekday': 'first',
    'IsWeekend': 'first',
    'IsHoliday': 'first',
    'Temperature': 'mean',
    # 'Irrad_direct': 'mean',
    # 'Irrad_difuse': 'mean',
    'Load': 'mean'}).reset_index()

    if include_ID:
        aggregated_data['ID'] = [f'ID-{i}' for i in range(len(aggregated_data))]
        aggregated_data = aggregated_data[['ID'] + [col for col in aggregated_data.columns if col != 'ID']]

    aggregated_data.to_csv(os.path.join(os.path.dirname(path_to_file), f'{ctr_code}_Daily.csv'), index=False, float_format=ff_format)

def Construct_Weekly_CSV(
        path_to_file:str,
        ctr_code:str,
        cols_to_drp:list,
        ff_format:str='%.5f',
        include_ID:bool=False
    ) -> None:
    df = pd.read_csv(path_to_file)
    df.drop(columns=cols_to_drp, inplace=True)

    aggregated_data = df.groupby(['Year', 'Month', 'WeekOfMonth']).agg({
    'Temperature': 'mean',
    # 'Irrad_direct': 'mean',
    # 'Irrad_difuse': 'mean',
    'Load': 'mean'}).reset_index()

    if include_ID:
        aggregated_data['ID'] = [f'ID-{i}' for i in range(len(aggregated_data))]
        aggregated_data = aggregated_data[['ID'] + [col for col in aggregated_data.columns if col != 'ID']]
    
    aggregated_data.to_csv(os.path.join(os.path.dirname(path_to_file), f'{ctr_code}_Weekly.csv'), index=False, float_format=ff_format)

def Construct_Monthly_CSV(
        path_to_file:str,
        ctr_code:str,
        cols_to_drp:list,
        ff_format:str='%.5f',
        include_ID:bool=False
    ) -> None:
    df = pd.read_csv(path_to_file)
    df.drop(columns=cols_to_drp, inplace=True)

    aggregated_data = df.groupby(['Year', 'Month']).agg({
    'Temperature': 'mean',
    # 'Irrad_direct': 'mean',
    # 'Irrad_difuse': 'mean',
    'Load': 'mean'}).reset_index()

    if include_ID:
        aggregated_data['ID'] = [f'ID-{i}' for i in range(len(aggregated_data))]
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
        cols_to_drp:list,
        ff_format:str
) -> None:
    for str_code in ctr_code:    
        Construct_Hourly_CSV(os.path.join(path_to_file, f'{str_code}_base.csv'), str_code, cols_to_drp, ff_format=ff_format)
        Construct_Daily_CSV(os.path.join(path_to_file, f'{str_code}_base.csv'), str_code, cols_to_drp, ff_format=ff_format)
        Construct_Weekly_CSV(os.path.join(path_to_file, f'{str_code}_base.csv'), str_code, cols_to_drp, ff_format=ff_format)
        Construct_Monthly_CSV(os.path.join(path_to_file, f'{str_code}_base.csv'), str_code, cols_to_drp, ff_format=ff_format)
        os.remove(os.path.join(path_to_file, f'{str_code}_base.csv'))

def Create_Sequence(
        dataset: pd.DataFrame, 
        look_back:int,
        target_column:int = -1     
    ) -> tuple:

    nrows, ncols = dataset.shape
    X = []
    Y = []
    
    for i in range(look_back, nrows):
        # Get only the target column for the look-back rows
        look_back_rows = dataset.iloc[i - look_back: i, target_column].values.reshape(-1, 1)
        # Get the current row excluding the target column
        current_row_ex_target = dataset.iloc[i, dataset.columns != dataset.columns[target_column]].values
        # Combine the look-back rows and the current row (without the target column)
        combined_rows = np.hstack([current_row_ex_target, look_back_rows.flatten()])
        X.append(combined_rows)
        # Target value is the current row's target column value
        Y.append(dataset.iloc[i, target_column])

    
    return np.array(X), np.array(Y)

def Create_Dataset(
        path_to_file:str,
        key_to_split:str,
        value_to_split:int,
        features_to_scale:list,
        look_back:int
    ):
    train_set, test_set = Train_Test_Split_OnYear(path_to_file, key_to_split, value_to_split)
    train_set_scaled, test_set_scaled, scaler = Apply_Scaling(train_set, test_set, features_to_scale)
    
    train_set_X, train_set_Y = Create_Sequence(train_set_scaled, look_back)
    test_set_X, test_set_Y   = Create_Sequence(test_set_scaled, look_back)

    return train_set_X, train_set_Y, test_set_X, test_set_Y, scaler

def Manage_Datasets(
        path_to_dir_src:str,
        path_to_dir_final: str, 
        pickle_name:str,
        key_to_split:str,
        value_to_split:int,
        features_to_scale:list,
        look_back:int
    ) -> dict:
    saved_dict = {}
    filenames = os.listdir(path_to_dir_src)
    contains_name = lambda s, names: any(name in s for name in names)
    for fname in filenames:
        if contains_name(fname, ['Daily', 'Hourly', 'Weekly', 'Monthly']):
            train_set_X, train_set_Y, \
            test_set_X, test_set_Y,   \
            scaler = Create_Dataset(os.path.join(path_to_dir_src, fname),
                                    key_to_split,
                                    value_to_split,
                                    features_to_scale,
                                    look_back)
            
            saved_dict[os.path.splitext(fname)[0]] = {
                'trX': train_set_X,
                'trY': train_set_Y,
                'tsX': test_set_X,
                'tsY': test_set_Y,
                'scaler': scaler
            }
        
    with open(os.path.join(path_to_dir_final, pickle_name), 'wb') as f:
        pickle.dump(saved_dict, f)
    
    return saved_dict

def Read_Pickle(
        file_path:str
    ) -> dict:

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

if __name__ == "__main__":

    # Print Setup
    np.set_printoptions(formatter={'float': lambda x: "{:5.1f}".format(x)})

    # Default Paths
    path_processed = "../../data/processed"
    path_interim = "../../data/interim"
    path_to_dir_final = "../../data/final"
    pickle_name = 'Processed-Dataset.pickle'

    # Default Parameters
    ctr_code = ['ES', 'PT', 'PL', 'FR', 'SE']
    cols_to_drp = ['Date', 'DayOfYear', 'WeekOfYear', 'Quarter', 'Irrad_direct', 'Irrad_difuse']
    features_to_scale = ['Temperature', 
                        #  'Irrad_direct', 
                        #  'Irrad_difuse', 
                         'Load']
    ff_format = '%.4f'
    key_to_split = 'Year'
    value_to_split = 2019
    look_back=3
    
    # Default Code
    Copy_CSVs_For_Dataset(path_interim, path_processed)
    Manage_CSVs(path_processed, ctr_code, cols_to_drp, ff_format)
    mydict = Manage_Datasets(path_processed, path_to_dir_final, pickle_name, key_to_split, value_to_split, features_to_scale, look_back)
    print(f'PT_Daily:: Trainset-Length: {len(mydict["PT_Daily"]["trX"])} Testset-Length:{len(mydict["PT_Daily"]["tsX"])}')
    print(mydict["PT_Daily"]["trX"])
    print(' ')    
    print(mydict["PT_Daily"]["trY"])    
