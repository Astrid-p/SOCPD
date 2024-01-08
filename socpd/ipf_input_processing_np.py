import json
from ipfn import ipfn
import numpy as np
import pandas as pd
from io import StringIO
import os 
import itertools
from collections import defaultdict
import random


def transform_to_txt(input_file, save_folder):
    """ Read excel sheets and tranform table the excel file into txt file
        And save these new txt file into a new folder
    Input: 
        - input_file: the excel file with different sheets
        - save_folder: output folder that will be used to stored the transformed data (txt)
    output: 
        Auto save files
    """
    
    # Read the Excel file with multiple sheets
    excel_data = pd.read_excel(input_file, sheet_name=None)
    # Iterate through each sheet and export to a text file
    for sheet_name, sheet_data in excel_data.items():
        # Convert the sheet data to a tab-separated text format
        text_data = sheet_data.to_csv(sep='\t', index=False, header = None)

        # Export the text data to a file
        text_file_path = f'{save_folder}/{sheet_name}.txt'
        with open(text_file_path, 'w') as text_file:
            text_file.write(text_data)
        
"""-----------------------------------
Prepare single count/proportion input
--------------------------------------"""
def get_data_singletxt(filename):
    """ Code use for one txt file: Extract group distribution in each single variable into dictionay 

    Input: file path (.txt)
    return: dictionary (key: variable name, 
                        values : single-variable distribution stored in a series)
    """
    with open(filename, 'r') as params_txt:
        data = params_txt.read()

    # Split the text into sections (required separation : ";")
    sections = data.split(';')
    out_dict = {}# Initialize a dictionary to store the data

    # Iterate through sections and parse data
    for section in sections:
        
        lines = section.strip().lower().split('\n')

        # Check if there are enough lines to proceed
        if len(lines) < 2:
            continue

        category = lines[0].strip().split("\t")[0].lower()  # Use lowercase as keys in JSON
        _, *values = lines[0:]  # Skip the first line (header)

        # Extract values and convert to appropriate data types
        keys = []
        data_values = []

        for row in values:
            if row.strip():
                row_data = list(map(lambda x: x.strip(), row.lower().split('\t'))) #split group name and value
                keys.append(row_data[0])
                data_values.append(float(row_data[1]))

        # Create the dictionary entry
        out_dict[category] = [keys, [i for i in data_values]]
    return out_dict

def processing_single_data(xls_file, out_txt_folder, params_folder):
    """
    input: An Excel file with 2 sheets:
            - "params_individual" tab stored proportional distribution of groups in each variable
            - "groups_count" tab stored frequency of each group in each variable
    Return:
            - Save proportional distribution (json) in parameters folder for model run
            - Return a dictionary, including 2 dictionary dictionary of group counts and dictionary of group proportions in all variables
    """

    # tranform excel form into separate txt files, save txt files in out put folder for single distribution
    transform_to_txt(xls_file, out_txt_folder)
    
    # processing text file
    filenames = [i for i in os.listdir(out_txt_folder)]
    
    single_distribution = {}
    
    for f in filenames:
        
        output = get_data_singletxt(f"{out_txt_folder}/{f}")
        file_dict = {}
        if f == "cat_proportion.txt":
            output = {k: [_,[round(n/100,4) for n in v2]] for k, [_, v2] in output.items()} # transform percentage into proportion
            json_output = json.dumps(output, indent=2)
            # Write the JSON to a file
            with open(f'{params_folder}/params_individual.json', 'w') as json_file:
                json_file.write(json_output)
            
        # Transform values into series with index is the corresponding groups
        
        for k in output.keys():
            v_list = output[k]
            distribution = pd.Series(v_list[1], index = v_list[0], name = 'total')
            distribution =distribution.rename_axis(k)
            file_dict[k] = distribution.sort_index() # sorting group names in each variable
        
        single_distribution[f[:-4]] = file_dict
    return single_distribution


"""-----------------------------------
Prepare crosstab count/proportion input
--------------------------------------"""
def cross_check_groupname(single_distribution, crosstab_table_dict):
    # check if all group names of variable in single distribution data and crosstab table are all aligned

    group_dict = single_distribution["cat_proportion"]
    group_keymap = {k: list(v.index) for k, v in group_dict.items()}

    for k in crosstab_table_dict.keys():
        if list(crosstab_table_dict[k].index )==group_keymap[k[0]] and list(crosstab_table_dict[k].columns)==group_keymap[k[1]]:
            continue
        else:
            raise ValueError(f"group names of in single dist file don't match with crosstab table {k}")


def get_data_crosstabtxt(filename):
    """processing txt data from one txt file
    input: One txt-file path (.txt)
    Return: 
        Dictionary: 
         + key: name of variable pair
         + value: "crosstab_count_df"
    """
    
    # Read the text data from a file
    with open(filename, 'r') as textfile:
        input_text = textfile.read()
    sections = input_text.split(';')
    crosstab_count_df = {}
    
    for sec in sections: 
        # Read the text data from a file
        sec = sec.strip()
        lines = sec.split('\n')
        
        second_index = lines[0].strip().lower()
        data_text = '\n'.join([line.strip() for line in lines[1:] if line.strip()]).lower() # merging data, using strip to remove empty row 
        
        # Read the text into a DataFrame
        df = pd.read_csv(StringIO(data_text), sep='\t', thousands=',')
        
        first_index = df.columns[0]
        # Set 'age_cat' as the index
        df.set_index(first_index, inplace=True)

        # Drop the 'Total' row and column
        if 'total' in df.index:
            df = df.drop('total', axis=0)
        if 'total' in df.columns:
            df = df.drop('total', axis=1)
        df = df.dropna(axis = 1) # remove empty columns
        
        # save dictionary of crosstab-count df by keys - pair of corresponding variable
        # sorting index and columns by alphabet order
        crosstab_count_df[first_index, second_index] = df.sort_index(axis=0).sort_index(axis=1)
        
    return crosstab_count_df
    


def processing_crosstab_data(cross_tab_xls, out_txt_folder, single_distribution ):
    """ From excel file stored cross-tab distribution, return crostab count and proportion under dictionary 
        Cross-check if crosstab categorical name aligned with categorical name in single distribution file

    Input: 
         - Excel files with different sheets stored cross-tab distribution 
         - Output folder save txt file
    Return: 
         - save txt files under cross-tab data folder 
         - dictionary:
             key1: crosstab-count dictionary with sub keys are pair name by order
             key 1: crosstab-proportion dictionary with sub keys are pair name by order
    Collect crosstab data from all the txt files and add them up into one dictionary"""
    transform_to_txt(cross_tab_xls, out_txt_folder)
    filenames = [i for i in os.listdir(out_txt_folder)]
    crosstab_count_df = {} 
    crosstab_proportion_matrices = {}
    
    for f in filenames:
        o = get_data_crosstabtxt(f"{out_txt_folder}/{f}")
        crosstab_count_df.update(o)
    
     #cross-check if groups name of variable in all crosstab table are aligned  
    cross_check_groupname(single_distribution, crosstab_count_df)

    # transform into numpy array, compute proportional distribution
    crosstab_proportion_matrices = {k: np.array(df)/np.array(df).sum()  for k, df in crosstab_count_df.items()}

 
    return {"crosstab_count_df": crosstab_count_df, "crosstab_proportion_matrices":crosstab_proportion_matrices}


"""-------------------------------------
    Create count inputs for ipf
    -------------------------------------
"""


# Relable crosstab matrix to get index in combination
def relabel_keys(obj, label_mapping):
    if isinstance(obj, tuple):
        return tuple(label_mapping.get(item, item) for item in obj)
    elif isinstance(obj, dict):
        return {relabel_keys(key,label_mapping): value for key, value in obj.items()}
    else:
        return obj
    
# flatten feature combinations 
def flatten(d):
    for i in d:
        yield from [i] if not isinstance(i, tuple) else flatten(i)



def get_ifp_input(single_proportions, crosstab_proportion_matrices):
    
    # Create matrix with uniformed count (=1) on all combination of features(=1)
    m_shape = [len(i) for i in single_proportions.values()]
    m = np.ones(m_shape)

    
    # save keymap: maping position of variables in the matrix
    # And will be used to transform dimension indices
    features = list(single_proportions.keys())
    keymap = {k:v for v, k in enumerate(features)}

    # cat_keymap: maping the position/indices of categorical levels/ group-names in each varaibles in the matrix 
    cat_keymap = {k: list(v.index) for k, v in single_proportions.items()}
    cat_keymap = {k: {ind: cat for ind, cat in enumerate(cat_list)} for k, cat_list in cat_keymap.items()}


    # collect distribution
    aggregates = []
    dimensions = []
    for k,v in single_proportions.items():
        dimensions.append([keymap[k]])
        aggregates.append(np.array(v)*100)
    relabeled_crosstab_matrices = relabel_keys(crosstab_proportion_matrices, keymap)
    for k,v in relabeled_crosstab_matrices.items():
        dimensions.append(list(k))
        aggregates.append(v*100)

    return keymap,cat_keymap, m, aggregates, dimensions

        
"""______________________________________
REVERSE results
___________________________________________
"""

