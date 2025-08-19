# /usr/bin/env python3
""" Data ingestion module for merging multiple CSV files into a single DataFrame.
This module defines a class `DataIngestion` that handles the ingestion of data from multiple CSV files located in a specified input folder, merges them into a single DataFrame, and saves the result to an output folder.
"""

import pandas as pd
import os


class DataIngestion():
    """
    Class to handle data ingestion and merging of multiple CSV files into a single DataFrame.
    """
    def __init__(
            self,
            input_folder_path,
            output_folder_path
    ) -> None:
        """
        Initializes the DataIngestion class with input and output folder paths.
        """
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path

    def merge_multiple_dataframe(
            self
    ) -> None:
        """
        Merges multiple CSV files from the input folder into a single DataFrame and saves it to the output folder.
        """
        # Check for datasets, compile them together, and write to an output file
        files = os.listdir(self.input_folder_path)
        # save the file names used for data ingestion to a text file
        with open(os.path.join(self.output_folder_path, 'ingestedfiles.txt'), 'w') as f:
            for file in files:
                if file.endswith('.csv'):
                    f.write(file + '\n')
        
        dataframes = []

        for file in files:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(self.input_folder_path, file))
                dataframes.append(df)

        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)
            # Ensure the dataframe has unique rows
            merged_df = merged_df.drop_duplicates(ignore_index=True)
            output_file = os.path.join(self.output_folder_path, 'finaldata.csv')
            merged_df.to_csv(output_file, index=False)
            print(f'Final data saved to {output_file}')
        else:
            print('No CSV files found to merge.')
