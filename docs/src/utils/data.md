Module src.utils.data
=====================

Functions
---------

    

Create a final dataset by dropping rows, columns, and rounding the values.
    
    Args:
        raw_path: The path to the raw dataset.
        final_path: The path where the final dataset will be saved.
        columns_to_drop: The columns to be dropped.
    
    Returns:
        None.

    
`make_processed_data(raw_path, processed_path, columns_to_drop)`
:   

Create a processed dataset by dropping missing values and columns.
    
    Args:
        raw_path: The path to the raw dataset.
        processed_path: The path where the processed dataset will be saved.
        columns_to_drop: The columns to be dropped.
    
    Returns:
        None.