# ============================================================================#
# author        :   louis TOMCZYK
# goal          :   Definition of personalized Lists and Matrix functions
# ============================================================================#
# version       :   0.0.1 - 2021-09-27  - array2list
#                                       - column
#                                       - find_all_indexes
#                                       - find_nearest_element
#                                       - list2array
#                                       - ones
#                                       - negation
#                                       - remove_using_element
#                                       - remove_using_index
#                                       - remove_duplicated_values
#                                       - zeros
# ---------------
# version       :   0.0.2 - 2021-03-01  - custom sort
#                                       - flatten_list_of_list
# ============================================================================#

# convert an array type to list type
def array2list(Array):
    return Array.tolist()

            # ================================================#
            # ================================================#
            # ================================================#

# return the i_column of a matrix
def column(matrix, i):
    return [row[i] for row in matrix]

            # ================================================#
            # ================================================#
            # ================================================#
           
# find all indexes of an element in a list or array
def find_all_indexes(Element,List_or_Array):
    if type(List_or_Array) == list:
        Type = 'Liste'
    else:
        Type = 'Array'
    List_or_Array = list(List_or_Array)
    result = []
    offset = -1
    while True:
        try:
            offset = List_or_Array.index(Element, offset+1)
        except ValueError:
            if Type != "Liste":
                List_or_Array = np.array(List_or_Array)
            return result, List_or_Array
        result.append(offset)

            # ================================================#
            # ================================================#
            # ================================================#
        
def find_nearest_element(Element,List_or_Array):
       
    if type(List_or_Array)==list:
        List_or_Array = np.array(List_or_Array)
        
    tmp = np.abs(List_or_Array-Element)
    Difference_with_nearest_element = min(tmp)
    [Nearest_Element,tmp] = find_all_indexes(Difference_with_nearest_element, tmp)
    del(tmp)

    return List_or_Array[Nearest_Element]
    
            # ================================================#
            # ================================================#
            # ================================================#

# convert an list type to array type
def list2array(List):
    return np.array(List)

            # ================================================#
            # ================================================#
            # ================================================#

def ones(N_rows,N_columns,N_deep):
    return np.ones((N_rows,N_columns,N_deep))

            # ================================================#
            # ================================================#
            # ================================================#

def negation(List):
    List = [-List[k] for k in range(len(List))]
    return(List)
 
            # ================================================#
            # ================================================#
            # ================================================#

# https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list
    
def remove_using_element(Element,List_or_Array):
    List_or_Array = list(List_or_Array)
    while Element in List_or_Array:
        List_or_Array.remove(Element)
    return List_or_Array

            # ================================================#
            # ================================================#
            # ================================================#
            
#   https://www.kite.com/python/answers/how-to-remove-an-element-from-an-array-in-python
def remove_using_index(All_Indexes,List_or_Array):
    
    if type(List_or_Array) == list:
        Type = 'Liste'
    else:
        Type = 'Array'
    List_or_Array = list(List_or_Array)
    
    if type(List_or_Array) == list:
        N_Steps = len(All_Indexes)
        for k in range(N_Steps):
            tmp_index = All_Indexes[k]-k
            del(List_or_Array[tmp_index])
    if Type != "Liste":
        List_or_Array = np.array(List_or_Array)
        
    return List_or_Array

            # ================================================#
            # ================================================#
            # ================================================#
            
# https://waytolearnx.com/2019/04/supprimer-les-doublons-dune-liste-en-python.html
    
def remove_duplicated_values(List_or_Array):
    if type(List_or_Array) == list:
        Type = 'Liste'
    else:
        Type = 'Array'
    List_or_Array = list(List_or_Array)
    
    List_or_Array=list(set(List_or_Array))
    List_or_Array.sort()
    if Type != "Liste":
        List_or_Array = np.array(List_or_Array)
    return List_or_Array

            # ================================================#
            # ================================================#
            # ================================================#

def zeros(N_rows,N_columns,N_deep):
    return np.zeros((N_rows,N_columns,N_deep))

            # ================================================#
            # ================================================#
            # ================================================#

# https://www.kite.com/python/answers/how-to-sort-the-rows-of-a-numpy-array-by-a-column-in-python

def custom_sort(In,k):
    
    In_t    = In.transpose()
    Out_t   = In_t[np.argsort(In_t[:, ,k])]
    Out     = Out_t.transpose()
    
    return Out

            # ================================================#
            # ================================================#
            # ================================================#

# custom flatten
# -------
def flatten_list_of_list(list_in):
    
    list_out = []
    for k in range(len(list_in)):
        for j in range(len(list_in[k])):
            list_out.append(list_in[k][j])

    return list_out
# -------
