import os
import xlwings as xw
import pandas as pd
import re
import shutil
from pathlib import Path

def Organize_Files (
        source_dir:str, 
        target_dir:str, 
        country_code:list = ['ES', 'PT', 'PL', 'FR', 'SE']
    ) -> None:
    
    # Create Directory in Inter-class
    for lang in country_code:
        lang_dir = os.path.join(target_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
    
    # Iterate through files in the source directory
    for file_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file_name)

        # Check if the file is an Excel file
        if file_name.endswith('.xlsx'):
            # Determine the language from the file name suffix
            language = file_name.split('_')[-1].split('.')[0]

            if language in country_code:
                # Copy the file to the appropriate language directory
                target_path = os.path.join(target_dir, language, file_name)
                shutil.copy(source_path, target_path)
    print(f"Files Are Transfered to Intermediate Dataset Location: {target_dir}")

def Excel_Sheet_Manager (
        path_to_excel:str, 
        excel_name:str, 
        path_to_save:str
    ) -> None:
    
    with xw.App(visible=False) as app:
        wb = app.books.open(os.path.join(path_to_excel, excel_name)) 
        for sheet in wb.sheets:
            wb_new = app.books.add()
            sheet.copy(after=wb_new.sheets[0])
            wb_new.sheets[0].delete()
           
            # wb_new.save(os.path.join(path_to_save, f'{os.path.splitext(excel_name)[0]}_{sheet.name}.xlsx'))
            wb_new.save(os.path.join(path_to_save, f'{sheet.name}.xlsx'))
            Create_CSV(os.path.join(path_to_save, f'{sheet.name}.xlsx'),os.path.join(path_to_save, f'{sheet.name}.csv'))
            wb_new.close()
            os.remove(os.path.join(path_to_save, f'{sheet.name}.xlsx'))

def Create_CSV(
        input_file:str, 
        output_file:str, 
        skip_rows:int = 5, 
        header_idx:int = 4
    )-> None:
    # Read the Excel file
    df = pd.read_excel(input_file, header=header_idx)
    # Skip the first 5 rows
    df = df.iloc[skip_rows:]
    # Write the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

def Create_CSVs(
        path_input: str, 
        pattern:str = r'^(201[5-9]\.csv)$'
    ) -> None:
    # Iterate Through all files
    file_names = os.listdir(path_input)
    for idx, name in enumerate(file_names):
        # Create Seperate Directories
        save_dir_path = os.path.join(path_input, os.path.splitext(name)[0])
        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)
        # Process xlsx files
        Excel_Sheet_Manager(path_input, name, save_dir_path)
        Merge_CSVs(save_dir_path, pattern)
        shutil.rmtree(save_dir_path)
        os.remove(os.path.join(path_input, name))
        print(f'{name} [{100 * ((idx+1)/file_names.__len__())}%]!')
    # Operation Successful    
    print(f"Taransform Completed!")

def Filter_CSVs(
        path_to_dir:str, 
        pattern:str
    )->list: 
    # Get a list of CSV file paths in the input directory that match the pattern
    csv_files = sorted([
        os.path.join(path_to_dir, f) for f in os.listdir(path_to_dir) 
        if re.match(pattern, f)
    ])
    
    return csv_files

def Merge_CSVs(
        path_to_dir:str, 
        pattern:str
    ) -> None:
    # Get a list of CSV file paths in the input directory
    csv_files = Filter_CSVs(path_to_dir, pattern)
    
    # Initialize an empty DataFrame to hold the combined data
    df_combined = pd.DataFrame()
    
    # Iterate over each CSV file path
    for i, csv_file in enumerate(csv_files):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Skip headers for all files except the first one
        if i > 0:
            df = df.iloc[1:]
        
        # Append the DataFrame to the combined DataFrame
        df_combined = pd.concat([df_combined, df], ignore_index=True)
    
    # Write the combined DataFrame to a new CSV file
    path = Path(path_to_dir)
    parent_dir = path.parent
    output_file_path = parent_dir / f"{path.name}.csv"
    df_combined.to_csv(output_file_path, index=False)
    # df_combined.to_csv(os.path.join(f"{path_to_dir}/..", f'{os.path.basename(path_to_dir)}.csv'), index=False)

def Clean_Duplicates_CSV (
        csv_path: str
    ) -> None :
    load_df = pd.read_csv(csv_path)
    # Remove Duplicate Elemnts
    load_df.drop_duplicates(subset='Date', keep='first', inplace=True)
    # Change Date String to Year
    load_df['Date'] = pd.to_datetime(load_df['Date']).dt.year
    load_df.rename(columns={'Date': 'Year'}, inplace=True)
    load_df.to_csv(csv_path, index=False)

def Clean_Dataset (
        path_to_dir:str
    ) -> None :
    # Iterate through files in the source directory
    file_names = os.listdir(path_to_dir)
    for file_name in file_names:
        # f"{path_to_dir}/{file_name}"
        Clean_Duplicates_CSV(os.path.join(path_to_dir, file_name)) 

def Join_CSVs (
        path_to_dir:str
    ) -> None :
    
    # Read each CSV file into a DataFrame
    path = Path(path_to_dir)
    holidays_df = pd.read_csv(path / f'Holidays_{path.name}.csv')
    irradiance_df = pd.read_csv(path / f'Irradiance_{path.name}.csv')
    load_df = pd.read_csv(path / f'Load_{path.name}.csv')
    temperature_df = pd.read_csv(path / f'Temperature_{path.name}.csv')

    merged_df = holidays_df
    merged_df = pd.merge(merged_df, irradiance_df, on=['Year', 'Month', 'Day', 'Hour'], how='outer')
    merged_df = pd.merge(merged_df, load_df, on=['Year', 'Month', 'Day', 'Hour'], how='outer')
    merged_df = pd.merge(merged_df, temperature_df, on=['Year', 'Month', 'Day', 'Hour'], how='outer')

    output_file_path = path.parent / f'{path.name}.csv'
    merged_df.to_csv(output_file_path, index=False)
    shutil.rmtree(path_to_dir)

def Prep_CSV (
        path_to_file:str, 
    ) -> None:
    # Remove Nan Elements
    df = pd.read_csv(path_to_file)
    df_cleaned = df.dropna()
    df_cleaned.to_csv(path_to_file, index=False)

def Sample_Manager(
        path_intermediate:str, 
        ctr_code:list,
        selected_cols:list
    ) -> None:
    # Iterate By Country
    for str_code in ctr_code:
        # Create CSVs
        Create_CSVs(os.path.join(path_intermediate, str_code), pattern)
        # Clean CSVs From Duplicate Date
        Clean_Dataset(os.path.join(path_intermediate, str_code))
        # Combine All Information
        Join_CSVs (os.path.join(path_intermediate, str_code))
        # Prep Dataset
        Prep_CSV(os.path.join(path_intermediate, f'{str_code}.csv'))
        # Reorder the columns
        Reorder_CSV(os.path.join(path_intermediate, f'{str_code}.csv'), selected_cols)

def Test_Manager(
        path_to_file:str, 
        ctr_code: list
    ) -> dict:
    status_dict = {}
    for code in ctr_code:
        print(f"For code {code}:")
        df = pd.read_csv(os.path.join(path_to_file, f'{code}.csv'))
        # Check for missing values or empty elements
        total_nan_count = df.isnull().sum().sum()
        total_empty_count = (df.astype(str) == '').sum().sum()
        print("Total count of NaN values:", total_nan_count)
        print("Total count of Empty values:", total_empty_count)

        status_dict[code] = (total_nan_count, total_empty_count)

        nan_indices = df.isnull().any(axis=1)
        nan_columns = df.columns[df.isnull().any()]

        # Print out the rows and columns where NaN values occur
        print("Rows with NaN values:")
        print(df[nan_indices])

        print("\nColumns with NaN values:")
        print(nan_columns)
    
    return status_dict 

def Reorder_CSV (
        path_to_file:str, 
        selected_columns: list
    ) -> None:
    df = pd.read_csv(path_to_file)
    df_reordered = df[selected_columns]
    df_reordered.to_csv(path_to_file, index=False)


if __name__ == "__main__":
    # Important Paths
    PATH_raw     = "../../data/raw"
    PATH_interim = "../../data/interim"
    ctr_code = ['ES', 'PT', 'PL', 'FR', 'SE']
    pattern = r'^(201[5-9]\.csv)$'
    selected_cols = ['Year', 'Month', 'Day', 'Hour', 
                     'A', 'B', 'weekday', 
                     'Temperature', 'Irrad_direct', 'Irrad_difuse', 
                     'Load_Prev', 'Load']
    Organize_Files(PATH_raw, PATH_interim, ctr_code)
    Sample_Manager(PATH_raw, PATH_interim, ctr_code, selected_cols)
    Test_Manager(PATH_interim, ctr_code)


