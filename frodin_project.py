import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import math

failures = {}
functions = {}
tests = {}
cols = ['Test', 'Function', 'Failed', 'Passed']
data = []
test_names = []
total_fails = 0
total_pass = 0

def output_data(data_frame, type_of_data):
    '''Generates a CSV for each of the types of SBFL records. These are in CSV files and sorted by suspiciousness level according to the chosen formula.'''
    try:
        data_frame.to_csv(f"output/{type_of_data}.csv", index=False)
    except:
        print(f"{type_of_data} didn't work - some DF issue to look at.")

def clean(data_frame):
    '''Cleans data by counting number of times a function appears, then counting how many times it is in a failed test and in a passed test. 
    Finally, duplicate records of the same function are dropped, keeping the totals intact, and the unique function name rows.'''
    # first extract how many times a function occurs in the data
    function_counts = data_frame['Function'].value_counts()
    # then, find out how many times that function failed and add that as a column
    failure_counts = data_frame[data_frame['Failed'] == True]['Function'].value_counts()
    # then, find out how many passed and add that as a column - this column + failed should equal occurrences
    success_counts = data_frame[data_frame['Passed'] == True]['Function'].value_counts()
    # map the number of occurrences for easy reference
    data_frame['Occurrence_Count'] = data_frame['Function'].map(function_counts)
    # map the failure counts for easy reference
    data_frame['Failure_Count'] = data_frame['Function'].map(failure_counts)
    # map the success counts for easy reference
    data_frame['Success_Count'] = data_frame['Function'].map(success_counts)
    # fill NaN values with 0 where applicable
    data_frame.fillna(0, inplace = True)
    # remove all duplicate occurrences of any function so that only the first test to log the function will show that test.
    data_frame = data_frame.drop_duplicates(subset=['Function'])
    # reset the index of the dataframe so that it can be referenced if needed
    data_frame.reset_index(drop=True, inplace=True)
    return data_frame

def calculate_tarantula(data_frame):
    '''The Tarantula formula is Suspicious (s) = (fails / totalfail) / ((fails / totalfail) + (pass / totalpass)) where 
    fails are the tests that covered the function and failed, passes are the tests that covered the test and passed, and totalfail/totalpass 
    are the grand totals of all failed/passed tests.'''
    t_cols = ['Function', 'Occurrence_Count', 'Failure_Count', 'Success_Count', 'Suspiciousness_Score']
    global total_fails, total_pass
    local_df = data_frame.drop(['Failed', 'Passed', 'Test'], axis=1)
    t_data = []
    for index, value in local_df.iterrows():
        # store the failures for easy reference
        fails = value['Failure_Count']
        # store the passes for easy reference
        passes = value['Success_Count']
        # calculate tarantula score
        t_score = fails / total_fails
        t_score /= ((fails / total_fails) + (passes / total_pass))
        t_data.append((value['Function'], value['Occurrence_Count'], fails, passes, t_score))
    score_list = [item[4] for item in t_data]
    data_frame['Tarantula_Score'] = score_list
    tarantula_df = pd.DataFrame(columns=t_cols, data=t_data)
    tarantula_df = tarantula_df.sort_values(by='Suspiciousness_Score', ascending=False)
    #tarantula_df.reset_index(drop=True, inplace=True)
    output_data(tarantula_df, "Tarantula")
    return tarantula_df, data_frame

def calculate_sbi(data_frame):
    '''The SBI formula is Suspicious (s) = fails / (fails + passes) where 
    fails are the tests that covered the function and failed, and passes are the tests that covered the test and passed.'''
    sbi_cols = ['Function', 'Occurrence_Count', 'Failure_Count', 'Success_Count', 'Suspiciousness_Score']
    global total_fails, total_pass
    local_df = data_frame.drop(['Failed', 'Passed', 'Test'], axis=1)
    sbi_data = []
    for index, value in local_df.iterrows():
        # store the failures for easy reference
        fails = value['Failure_Count']
        # store the passes for easy reference
        passes = value['Success_Count']
        # calculate sbi score
        sbi_score = fails / (passes + fails)
        sbi_data.append((value['Function'], value['Occurrence_Count'], fails, passes, sbi_score))
    score_list = [item[4] for item in sbi_data]
    data_frame['SBI_Score'] = score_list
    sbi_df = pd.DataFrame(columns=sbi_cols, data=sbi_data)
    sbi_df = sbi_df.sort_values(by='Suspiciousness_Score', ascending=False)
    #sbi_df.reset_index(drop=True, inplace=True)
    output_data(sbi_df, "SBI")
    return sbi_df, data_frame
    

def calculate_jaccard(data_frame):
    '''The Jaccard formula is Suspicious (s) = fails / (totalfailed + passes) where 
    fails are the tests that covered the function and failed, passes are the tests that covered the test and passed, and totalfailed is 
    the grand total of all failed tests.'''
    j_cols = ['Function', 'Occurrence_Count', 'Failure_Count', 'Success_Count', 'Suspiciousness_Score']
    global total_fails, total_pass
    local_df = data_frame.drop(['Failed', 'Passed', 'Test'], axis=1)
    j_data = []
    for index, value in local_df.iterrows():
        # store the failures for easy reference
        fails = value['Failure_Count']
        # store the passes for easy reference
        passes = value['Success_Count']
        # calculate jaccard score
        j_score = fails / (total_fails + passes)
        j_data.append((value['Function'], value['Occurrence_Count'], fails, passes, j_score))
    score_list = [item[4] for item in j_data]
    data_frame['Jaccard_Score'] = score_list
    j_df = pd.DataFrame(columns=j_cols, data=j_data)
    j_df = j_df.sort_values(by='Suspiciousness_Score', ascending=False)
    #j_df.reset_index(drop=True, inplace=True)
    output_data(j_df, "Jaccard")
    return j_df, data_frame

def calculate_ochiai(data_frame):
    '''The Ochiai formula is Suspicious (s) = fails / sqrt(totalfailed * (passes + fails)) where 
    fails are the tests that covered the function and failed, passes are the tests that covered the test and passed, and totalfailed is 
    the grand total of all failed tests.'''
    o_cols = ['Function', 'Occurrence_Count', 'Failure_Count', 'Success_Count', 'Suspiciousness_Score']
    global total_fails, total_pass
    local_df = data_frame.drop(['Failed', 'Passed', 'Test'], axis=1)
    o_data = []
    for index, value in local_df.iterrows():
        # store the failures for easy reference
        fails = value['Failure_Count']
        # store the passes for easy reference
        passes = value['Success_Count']
        # calculate the ochiai score
        o_score = fails / math.sqrt(total_fails * (passes + fails))
        o_data.append((value['Function'], value['Occurrence_Count'], fails, passes, o_score))
    score_list = [item[4] for item in o_data]
    data_frame['Ochiai_Score'] = score_list
    o_df = pd.DataFrame(columns=o_cols, data=o_data)
    o_df = o_df.sort_values(by='Suspiciousness_Score', ascending=False)
    #o_df.reset_index(drop=True, inplace=True)
    output_data(o_df, "Ochiai")
    return o_df, data_frame

def generate_charts(tarantula, sbi, jaccard, ochiai):
    '''This is not strictly part of the class project objectives, but I thought it helpful compared to looking at giant CSV files. This just generates a 3D
    scatter plot for each type, mapping the success, failure, and suspiciousness of each formula. Then, a bar chart with the max of each is generated into the 
    output directory where the CSVs are stored.'''
    # these files are the same so they can be looped
    out_files = ['tarantula_plot.png', 'sbi_plot.png', 'jaccard_plot.png', 'ochiai_plot.png']
    for item in out_files:
        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")
        ax.set_xlabel('Success Count')
        ax.set_ylabel('Failure Count')
        ax.set_zlabel('Suspiciousness Score')
        if item.find('tarantula') != -1:
            ax.set_title('Tarantula Scatter Plot')
            ax.scatter(tarantula['Success_Count'], tarantula['Failure_Count'], tarantula['Suspiciousness_Score'])
        elif item.find('sbi') != -1:
            ax.set_title('SBI Scatter Plot')
            ax.scatter(sbi['Success_Count'], sbi['Failure_Count'], sbi['Suspiciousness_Score'])
        elif item.find('jaccard') != -1:
            ax.set_title('Jaccard Scatter Plot')
            ax.scatter(jaccard['Success_Count'], jaccard['Failure_Count'], jaccard['Suspiciousness_Score'])
        else:
            ax.set_title('Ochiai Scatter Plot')
            ax.scatter(ochiai['Success_Count'], ochiai['Failure_Count'], ochiai['Suspiciousness_Score'])
        ax.view_init(elev=26, azim=157)
        plt.savefig(f'output/{item}')
        plt.close()
    # this last file is different so it is done separately
    image_cols = ['Tarantula', 'SBI', 'Jaccard', 'Ochiai']
    max_data = [tarantula['Suspiciousness_Score'].max(), sbi['Suspiciousness_Score'].max(), jaccard['Suspiciousness_Score'].max(), ochiai['Suspiciousness_Score'].max()]
    plt.bar(image_cols, max_data)
    plt.xlabel('SBFL Formula')
    plt.ylabel('Suspiciousness Score')
    plt.title('Max Suspiciousness Found')
    plt.savefig('output/max_suspiciousness.png')
    plt.close()


def read_files():
    '''This function reads the files and stores some of the relevant data. Finally, this generates the dataframe used by all other elements of the project.'''
    files = []
    global functions, failures, cols, test_names, total_fails, total_pass
    path = os.path.join(os.getcwd(), "CoverageData/NewCoverageData")
    for filename in os.listdir(path):
        # init passed to false for each file
        passed = False
        test_name = ""
        with open(os.path.join(path, filename), 'r') as f:
            for line in f:
                # obtain totalfail and totalpass
                if line.find("true") != -1 or line.find("false") != -1:
                    if line.find("true") != -1:
                        # set file level pass/fail to True
                        passed = True
                        total_pass += 1
                    else:
                        total_fails += 1
                    # extract the test name
                    line_list = line.split('.')
                    # add test name to global list
                    test_names.append(line_list[-1].split(' ')[0])
                    test_name = line_list[-1].split(' ')[0]
                if line.find("true") == -1 and line.find("false") == -1:
                    if line not in functions:
                        functions[line] = 1
                    else:
                        functions[line] += 1
                    if passed:
                        data.append((test_name, line.strip(), False, True))
                    else:
                        data.append((test_name, line.strip(), True, False))
    tests = pd.DataFrame(data, index = range(len(data)), columns = cols)
    return tests

def main():
    data_frame = read_files()
    clean_data = clean(data_frame)
    tarantula, clean_data = calculate_tarantula(clean_data)
    sbi, clean_data = calculate_sbi(clean_data)
    jaccard, clean_data = calculate_jaccard(clean_data)
    ochiai, clean_data = calculate_ochiai(clean_data)
    # create an average to sort by
    avg_colums = ['Tarantula_Score', 'SBI_Score', 'Jaccard_Score', 'Ochiai_Score']
    clean_data['Average_Score'] = clean_data[avg_colums].mean(axis=1)
    # sort by the average
    clean_data = clean_data.sort_values(by="Average_Score", ascending=False)
    output_data(clean_data, "Composite")
    #print(clean_data)
    generate_charts(tarantula, sbi, jaccard, ochiai)
    return 0

if __name__ == "__main__":
    main()