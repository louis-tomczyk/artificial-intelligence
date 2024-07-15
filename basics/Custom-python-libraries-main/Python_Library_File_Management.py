# ============================================================================#
# author        :   louis TOMCZYK
# goal          :   Definition of personalized Files Management functions
# ============================================================================#
# version       :   0.0.6 - 2021 09 29  - cd
#                                       - clc
#                                       - get_path_and_name_of_file
#                                       - Import_Files_In_Out
#                                       - Import_File
#                                       - ls
#                                       - Read_Matrix
# ------------------------------------
# version       :   0.0.7 - 2022 03 03
#                                       - write_matrix_in_file
#                                       -
#                                       -
#                                       -
#                                       -
#                                       -
#                                       -
#                                       -
#                                       -
# ============================================================================#

import os
import numpy as pd
from tkinter import *

def cd(path):
    return os.chdir(path)

            # ================================================#
            # ================================================#
            # ================================================#

def clc():
    return os.system('clear')

            # ================================================#
            # ================================================#
            # ================================================#

# import several files

def get_path_and_name_of_file(file_name):
    
    # From the complete path of a file, return the path and the name of the
    # file
    
    file_name_split     = file_name.split("/")
    name                = file_name_split[-1]            # name of the file
    file_path           = file_name[:-len(name)]    # path of the file
    
    return[file_path, name]

            # ================================================#
            # ================================================#
            # ================================================#

# import only 2 files : input and output spectrums for NF calculations

def Import_Files_In_Out():
    
    # Import two files (spectrum in and spectrum out), return the complete
    # path of the two files (path+name) and the path of the first file.

    
    root = Tk()     # open window
    # names of the files to open
    file_name_in            = print("\n \t File IN : ")
    file_name_in            = filedialog.askopenfilename(initialdir="~/Bureau/",title="Select a file",filetypes=([("txt files","*.txt")]))

    matches_in              = re.finditer("/",file_name_in)
    matches_out             = re.finditer(".",file_name_in)

    matches_in_positions    = [match.start() for match in matches_in]
    matches_out_positions   = [match.start() for match in matches_out]

    # indexes of the specified characters
    index_start             = matches_in_positions[-1]+1
    index_end               = matches_out_positions[-1]+1

    path_of_spectrums       = file_name_in[0:index_start]
    print("\n\t\t {} \n".format(file_name_in[index_start:index_end]))

    file_name_out           = print("\t File OUT : ")
    file_name_out           = filedialog.askopenfilename(initialdir=path_of_spectrums,title="Select a file",filetypes=([("txt files","*.txt")]))

    matches_in              = re.finditer("/",file_name_out)
    matches_out             = re.finditer(".",file_name_out)

    matches_in_positions    = [match.start() for match in matches_in]
    matches_out_positions   = [match.start() for match in matches_out]

    # indexes of the specified characters
    index_start             = matches_in_positions[-1]+1
    index_end               = matches_out_positions[-1]+1

    print("\n\t\t {} \n".format(file_name_out[index_start:index_end]))

    # close window
    root.destroy()
    
    return [file_name_in, file_name_out ,path_of_spectrums]

            # ================================================#
            # ================================================#
            # ================================================#

def Import_File():
    
    # Import two files (spectrum in and spectrum out), return the complete
    # path of the two files (path+name) and the path of the first file.

    
    root = Tk()     # open window
    # names of the files to open
    file_name_in            = print("\n \t File IN : ")
    file_name_in            = filedialog.askopenfilename(initialdir="~/",title="Select a file",filetypes=([("all files",".*")]))

    matches_in              = re.finditer("/",file_name_in)
    matches_out             = re.finditer(".",file_name_in)

    matches_in_positions    = [match.start() for match in matches_in]
    matches_out_positions   = [match.start() for match in matches_out]

    # indexes of the specified characters
    index_start             = matches_in_positions[-1]+1
    index_end               = matches_out_positions[-1]+1

    path_of_spectrums       = file_name_in[0:index_start]

    # close window
    root.destroy()
    
    return [file_name,path__of_file]

            # ================================================#
            # ================================================#
            # ================================================#

def ls():
    return os.system('ls')

            # ================================================#
            # ================================================#
            # ================================================#

def Read_Matrix(Root,Delimiter):
    
    # Import two files (spectrum in and spectrum out), return the complete
    # path of the two files (path+name) and the path of the first file.


    root = Tk()     # open window
    # names of the files to open
    Root                    = input("\n\t What is the root path?\t ")
    file_name               = filedialog.askopenfilename(initialdir=Root,title="Select a file",filetypes=([("txt or csv files","*.txt *.csv")]))
    Delimiter               = input("\n \t What is the Delimiter?\t ")
    matches_in              = re.finditer("/",file_name)
    matches_out             = re.finditer(".",file_name)

    matches_in_positions    = [match.start() for match in matches_in]
    matches_out_positions   = [match.start() for match in matches_out]

    # indexes of the specified characters
    index_start             = matches_in_positions[-1]+1
    index_end               = matches_out_positions[-1]+1

    path_of_file            = file_name[0:index_start]

    # close window
    root.destroy()
    
    file_id              = open(file_name, 'r')
    DATA                 = genfromtxt(file_name,delimiter=Delimiter)
 
    return [file_name,path_of_file,DATA]


            # ================================================#
            # ================================================#
            # ================================================#

# https://stackoverflow.com/questions/22118648/how-to-write-a-numpy-matrix-in-a-text-file-python
def write_matrix_in_file(matrix,filename):

    df = pd.DataFrame(data=matrix.astype(float))
    df.to_csv(filename, sep=',', header=False, float_format='%.2f', index=False)
