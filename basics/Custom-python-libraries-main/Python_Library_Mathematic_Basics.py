# ============================================================================#
# author        :   louis TOMCZYK
# goal          :   Definition of personalized Mathematics Basics functions
# ============================================================================#
# version       :   0.0.1 - 2021 09 27  - derivative
#                                       - average
#                                       - moving_average
#                                       - basic_uncertainty
# ---------------
# version       :   0.0.2 - 2021 03 01  - num2bin
# ============================================================================#

def derivative(Y,dx):
    dYdx = []
    for k in range(len(Y)-1):
        dYdx.append((Y[k+1]-Y[k])/dx)
    return(dYdx)

            # ================================================#
            # ================================================#
            # ================================================#

def average(arr, n):
    end = n*int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)

            # ================================================#
            # ================================================#
            # ================================================#


def moving_average(List,N_samples):
    return np.convolve(List,np.ones(N_samples),'valid')/N_samples

            # ================================================#
            # ================================================#
            # ================================================#

# Uncertainty of data
def basic_uncertainty(data):  
    std_deviation = np.std(data)
    nb_data = len(data)
    b_uncertainty = std_deviation/np.sqrt(nb_data)
    return b_uncertainty

            # ================================================#
            # ================================================#
            # ================================================#

# convert number to binary
# -------
def num2bin(num):
    
    if num == 0:
        return [0]
    else:
        pow_max = int(np.floor(np.log2(num)))
        pow_list= [k for k  in range(pow_max+1)]
        bin_list= []
        num_tmp = num
        
        for k in range(1,len(pow_list)+1):
            
            pow_tmp = 2**pow_list[-k]
            
            diff    = num_tmp-pow_tmp
            
            if diff >= 0:
                num_tmp = diff
                bin_list.append(1)
            else:
                bin_list.append(0)
                
        return bin_list
# -------
