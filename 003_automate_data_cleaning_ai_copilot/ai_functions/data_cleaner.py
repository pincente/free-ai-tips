def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer



    # Step 1: Check for Missing Values
    missing_percentage = data_raw.isnull().mean() * 100
    columns_to_remove = missing_percentage[missing_percentage > 40].index
    data_cleaned = data_raw.drop(columns=columns_to_remove)

    # Step 2: Impute Missing Values
    numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = data_cleaned.select_dtypes(exclude=[np.number]).columns

    # Impute numeric columns with mean
    imputer_numeric = SimpleImputer(strategy='mean')
    data_cleaned[numeric_cols] = imputer_numeric.fit_transform(data_cleaned[numeric_cols])

    # Impute categorical columns with mode
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    data_cleaned[categorical_cols] = imputer_categorical.fit_transform(data_cleaned[categorical_cols])

    # Step 3: Convert Data Types
    data_cleaned['TotalCharges'] = pd.to_numeric(data_cleaned['TotalCharges'], errors='coerce')

    # Step 4: Remove Duplicate Rows
    data_cleaned = data_cleaned.drop_duplicates()

    # Step 5: Remove Rows with Missing Values
    data_cleaned = data_cleaned.dropna()

    # Step 6: Analyze Data for Additional Cleaning
    # (This is a placeholder, as no specific additional cleaning was identified.)
    # For now, we will assume no further cleaning is necessary.

    # Step 7: Document Observations
    # No further actions are needed as per analysis.

    # Step 9: Final Review
    return data_cleaned