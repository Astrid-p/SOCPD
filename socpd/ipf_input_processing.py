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
    """extract data from txt file that stored count or distribution of groups in each variable
    input: file path (.txt)
    return: dictionary (key: variable name, 
                        values : data stored in a series)
    """
    with open(filename, 'r') as params_txt:
        data = params_txt.read()

    # Split the text into sections based on ','
    sections = data.split(';')

    # Initialize a dictionary to store the data
    out_dict = {}

    # Iterate through sections and parse data
    for section in sections:
        
        lines = section.strip().lower().split('\n')

        # Check if there are enough lines to proceed
        if len(lines) < 2:
            continue

        category = lines[0].strip().split("\t")[0].lower()  # Use lowercase as keys in JSON
        header, *values = lines[1:]  # Skip the first line (header)

        # Extract values and convert to appropriate data types
        keys = []
        data_values = []

        for row in values:
            if row.strip():
                row_data = list(map(lambda x: x.strip(), row.lower().split('\t')))
                keys.append(row_data[0])
                data_values.append(float(row_data[1]))

        # Create the dictionary entry
        out_dict[category] = [keys, [i for i in data_values]]
    return out_dict

def processing_single_data(xls_file, out_txt_folder, params_folder):
    """
    input: Excel file with:
            - "params_individual" tab stored proportional distribution of groups in each variable
            - "groups_count" tab stored frequency of each group in each variable
    Return:
            - Save proportional distribution (json) in parameters folder for model run
            - Return a dictionary, including 2 dictionary dictionary of group counts and dictionary of group proportions in all variables
            """
    # tranform excel form into separate txt files
    transform_to_txt(xls_file, out_txt_folder)
    
    # processing text file
    filenames = [i for i in os.listdir(out_txt_folder)]
    single_distribution = {}
    for f in filenames:
        
        output = get_data_singletxt(f"{out_txt_folder}/{f}")
        file_dict = {}
        if f == "params_individual.txt":
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
            file_dict[k] = distribution
        
        single_distribution[f] = file_dict
    return single_distribution


"""-----------------------------------
Prepare crosstab count/proportion input
--------------------------------------"""


def get_data_crosstabtxt(filename):
    """processing txt data
    input: file path (.txt)
    Return: 
        Dictionary of "crosstab_count_matrices" and "crosstab_proportion_series"
        - dictionary  of crosstab frequency (matrice) of each pair of variables in the txt file, keys are the name of the variable pairs
        - dictionary of crosstab proportions 
    """
    
    # Read the text data from a file
    with open(filename, 'r') as textfile:
        input_text = textfile.read()
    sections = input_text.split(';')
    crosstab_count_matrices = {}
    
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
        
        #save crosstab matrices
        crosstab_count_matrices[first_index, second_index] = df
    
        # transform the contingency table into series
    # tranform series into the right ipf algo format, with index name (column)
    crosstab_proportion_series = {k: v.stack().rename_axis(list(k)).rename('total') for k, v in crosstab_count_matrices.items()}
    # Apply the function to the series
    crosstab_proportion_series = {k: v / v.sum() for k, v in crosstab_proportion_series.items()}
        
    return {"crosstab_count_matrices": crosstab_count_matrices, "crosstab_proportion_series": crosstab_proportion_series}
    
    #    # Calculate percentages of each combination pair over the whole column
    #    df_pct = df.div(df.sum(axis=0), axis=1) 

        # Stack the DataFrame to create a Series with MultiIndex
    #    series = df_pct.stack()

        # Convert the series index to a MultiIndex and set the second level name
    #    series.index = pd.MultiIndex.from_tuples([(ind1, ind2) for ind1, ind2 in series.index], names=[first_index, second_index])

        # Rename the series
    #    series.name = 'total'

        #sav crosstab proportions - series form
    #    crosstab_proportion_series[first_index, second_index] =  series
 


def processing_crosstab_data(cross_tab_xls, out_txt_folder):
    """
    Collect crosstab data from all the txt files and add them up into one dictionary"""
    transform_to_txt(cross_tab_xls, out_txt_folder)
    filenames = [i for i in os.listdir(out_txt_folder)]
    crosstab_count_matrices = {} 
    crosstab_proportion_series = {}
    
    for f in filenames:
        o = get_data_crosstabtxt(f"{out_txt_folder}/{f}")
        crosstab_count_matrices.update(o['crosstab_count_matrices'])
        crosstab_proportion_series.update(o['crosstab_proportion_series'])
 
    return {"crosstab_count_matrices": crosstab_count_matrices, "crosstab_proportion_series":crosstab_proportion_series}


"""-------------------------------------
    Create count inputs for ipf
    - seed_data: count of joint frequency accross all varaibles
    - maginal input (crosstab)
    -------------------------------------
"""

#____________ get seed_data

# Relable crosstab matrix to get index in combination
def relabel_keys(obj, label_mapping):
    if isinstance(obj, tuple):
        return tuple(label_mapping.get(item, item) for item in obj)
    elif isinstance(obj, dict):
        return {relabel_keys(key,label_mapping): relabel_keys(value,label_mapping) for key, value in obj.items()}
    else:
        return obj
    
# flatten feature combinations 
def flatten(d):
    for i in d:
        yield from [i] if not isinstance(i, tuple) else flatten(i)
        
def generate_shared_features_count(group_counts, crosstab_count_matrices, iteration, n_sample, seed = None):
    
    # Create a list of all combinations of 
    features = list(group_counts.keys())
    combinations = group_counts[features[0]].index
    for f in features[1:]:
        combinations = list(itertools.product(combinations,group_counts[f].index )) # combination of groups from features
        combinations = [tuple(flatten(i)) for i in combinations]
 
    # Relabel key for crosstab_matric, into number to trace down the location of crosstab values from group name in the corresponding variable value
    
    keymap = {k:v for v, k in enumerate(features)}
    relabeled_crosstab_matrices = relabel_keys(crosstab_count_matrices, keymap)
    
    # randomly sample list of 50 combinations
    if seed == None: 
        seed = 42
    random.seed(seed)
    
        # Create a DataFrame to store the results
    result_df = pd.DataFrame(columns= features + ['total'])

    min_count = 1000000
    
    for i in range(iteration):
        if i%10==0:
            print(f"----iteration {i}-----")
        sample = random.sample(combinations, n_sample)
        combinations = list(set(combinations)-set(sample)) # remove the chosen combinations
        
        newrows = defaultdict(list)
        
        
        
        for com in sample:
            # single varaible counts
            #shared_count_single= [group_counts[f].loc[group] for f, group in zip(features, com)]
            #shared_count = shared_count_single+shared_count_combo
            #crosstab count
            shared_count = [relabeled_crosstab_matrices[k].loc[com[k[0]], com[k[1]]] for k in relabeled_crosstab_matrices.keys()]
            
            row = {f: group for f, group in zip(features, com)}
            row['total'] = min(shared_count)
            if min_count > min(shared_count):
                min_count =  min(shared_count)
            for k, v in row.items():
                newrows[k].append(v)
        newrows_df = pd.DataFrame(newrows)
        result_df = pd.concat([result_df, newrows_df], ignore_index=True)

        
    
    # Assign remaining combinations with the smallest count 
    df_remained = pd.DataFrame(combinations, columns= features )
    df_remained['total'] = min_count
    result_df = pd.concat([result_df, df_remained], ignore_index=True)
  
    return result_df

























    
def trf_params_individual(input_file, txt_params_folder, params_folder):
    
    transform_to_txt(input_file,txt_params_folder)
    # Read the text file
    with open(f'{txt_params_folder}/params_individual.txt', 'r') as params_txt:
        data = params_txt.read()

    # Split the text into sections based on ','
    sections = data.split(';')

    # Initialize a dictionary to store the data
    json_data = {}

    # Iterate through sections and parse data
    for section in sections:
        
        lines = section.strip().split('\n')

        # Check if there are enough lines to proceed
        if len(lines) < 2:
            continue

        category = lines[0].strip().split("\t")[0].lower()  # Use lowercase as keys in JSON
        header, *values = lines[1:]  # Skip the first line (header)

        # Extract values and convert to appropriate data types
        keys = []
        data_values = []

        for row in values:
            if row.strip():
                row_data = list(map(lambda x: x.strip(), row.lower().split('\t')))
                keys.append(row_data[0])
                data_values.append(float(row_data[1]))

        # Create the dictionary entry
        json_data[category] = [keys, [round(n/100,3) for n in data_values]]

    # Convert the dictionary to JSON
    json_output = json.dumps(json_data, indent=2)
    # Write the JSON to a file
    with open(f'{params_folder}/params_individual.json', 'w') as json_file:
        json_file.write(json_output)
        
def get_single_distribution(jsonfile):
    f = open(jsonfile, "r")
    # Reading from file
    data = json.loads(f.read())
    single_dist_dict = {}
    for k in data.keys():
        v = data[k]
        distribution = pd.Series(v[1], index = v[0], name = 'prob')
        distribution =distribution.rename_axis(k)
        single_dist_dict[k] = distribution
    return single_dist_dict

