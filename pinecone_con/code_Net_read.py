import pandas as pd
import os
def python_files_to_df():
    base_dir = '/Users/andrewyang/Desktop/workspace/turintech/LLM_BPD/Project_CodeNet_Python800'

    data = []
    index_number = 1

    # Iterate over each folder in the base directory
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith('.py'):
                    with open(file_path, 'r', encoding='utf-8') as file_content:
                        try:
                            code = file_content.read()
                            # Append the content along with the index to the data list
                            data.append({'index': index_number, 'codes': code})
                            index_number += 1
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")

    # Create a DataFrame from the data list
    df = pd.DataFrame(data, columns=['index', 'codes']).set_index('index')
    return df
