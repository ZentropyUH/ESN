import os
from lz_utils import convert_to_numeric

def process_lz76(text):
    parsed_data = {}
    lines = text.split('\n')
    for line in lines:
        # Splitting each line at the colon
        parts = line.split(':')
        if len(parts) == 2:
            # Cleaning and storing the key-value pair
            key = parts[0].strip().strip('[]').strip()
            value = parts[1].strip().strip('[]').strip()
            # Convert to numeric if possible
            parsed_data[key] = convert_to_numeric(value)
    return parsed_data