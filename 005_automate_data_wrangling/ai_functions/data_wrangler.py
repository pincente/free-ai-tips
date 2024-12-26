def data_wrangler(data_list):
    import pandas as pd
    import numpy as np
    '''
    Wrangle the data provided in data.
    
    data_list: A list of one or more pandas data frames containing the raw data to be wrangled.
    '''


    # Ensure data_list is a list; if not, convert it to a list
    if not isinstance(data_list, list):
        data_list = [data_list]

    # Step 1: Load the datasets (already done as dataframes are passed in)
    # Step 2: Concatenate the DataFrames into a single DataFrame
    data_combined = pd.concat(data_list, ignore_index=True)

    # Step 3: Check for duplicates and remove them if any
    data_combined = data_combined.drop_duplicates()

    # Step 4: Standardize column data types
    # Convert 'drv' to object type for consistency in categorical data representation
    data_combined['drv'] = data_combined['drv'].astype(str)

    # Step 5: Handle inconsistent values in categorical columns
    # Standardizing transmission types by removing extra characters
    data_combined['trans'] = data_combined['trans'].str.replace(r'\(.*\)', '', regex=True).str.strip()

    # Step 6: Assess missing values (although none are noted, it's good practice)
    missing_values = data_combined.isnull().sum()
    
    # Step 7: Explore unique values in categorical columns
    unique_values_summary = {
        'trans': data_combined['trans'].unique(),
        'drv': data_combined['drv'].unique(),
        'class': data_combined['class'].unique()
    }

    # Step 8: Create derived columns if needed
    # Example: Calculate combined city and highway mileage as an average
    data_combined['avg_mpg'] = (data_combined['cty'] + data_combined['hwy']) / 2

    # Step 9: Final data summary
    # Print a description of numerical columns and value counts for categorical columns
    numerical_summary = data_combined.describe()
    categorical_summary = {
        'class_counts': data_combined['class'].value_counts(),
        'trans_counts': data_combined['trans'].value_counts(),
        'drv_counts': data_combined['drv'].value_counts()
    }

    # Return the wrangled DataFrame
    return data_combined