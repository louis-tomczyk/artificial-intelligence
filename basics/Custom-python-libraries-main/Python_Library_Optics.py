# ============================================================================#
# author        :   louis TOMCZYK
# goal          :   Definition of personalized Optical functions
# ============================================================================#
# version       :   0.0.1 - 2021 09 27  - mW2dBm 
#                                       - dBm2mW
#                                       - NF_calculation_High_Input_Power
#                                       - NF_calculation_Low_Input_Power
#                                       - linear_spectrum
# ============================================================================#

# convert power from mW to dBm
def mW2dBm(value_mW):  
    return 10*np.log10(value_mW/1)

            # ================================================#
            # ================================================#
            # ================================================#
 
# convert power from dBm to mW
def dBm2mW(value_dBm):  
    return 10**(value_dBm/10)

            # ================================================#
            # ================================================#
            # ================================================#

def NF_calculation_High_Input_Power(Power_ase_out_dBm,Gain_dB,Wavelength):

    c               = 299792458
    h               = 6.62607004e-34
    Wavelength      = Wavelength*1e-9
    Frequency       = c/Wavelength
    Resolution_OSA  = 0.2*1e-9
    dFrequency      = c*Resolution_OSA/(Wavelength**2)
    Power_ase_out_W = dBm2mW(Power_ase_out_dBm)/1000   
    gain_lin        = dBm2mW(Gain_dB)
    NF_High_Input_Power          = Power_ase_out_W/(h*Frequency*dFrequency*gain_lin)-1/gain_lin
    
    return 10*np.log10(NF_High_Input_Power)
   
            # ================================================#
            # ================================================#
            # ================================================# 
   
def NF_calculation_Low_Input_Power(Power_ase_in_dBm,Power_ase_out_dBm,Gain_dB,Wavelength):

    c               = 299792458
    h               = 6.62607004e-34
    Wavelength      = Wavelength*1e-9
    Frequency       = c/Wavelength
    Resolution_OSA  = 0.2*1e-9
    dFrequency      = c*Resolution_OSA/(Wavelength**2)
    Power_ase_in_W  = dBm2mW(Power_ase_in_dBm)/1000
    Power_ase_out_W = dBm2mW(Power_ase_out_dBm)/1000   
    gain_lin        = dBm2mW(Gain_dB)
    NF_Low_Input_Power          = (Power_ase_out_W-gain_lin*Power_ase_in_W)/(h*Frequency*dFrequency*gain_lin)+1/gain_lin
    
    return 10*np.log10(NF_Low_Input_Power)

            # ================================================#
            # ================================================#
            # ================================================# 

def linear_spectrum(data, time):
    samples     = len(data)
    fs          = samples/(time[-1]-time[0])
    amp_spec    = np.abs(np.fft.fft(data)/samples*(time[-1]-time[0]))
    amp_spec[0] = amp_spec[0]/2
    f           = np.fft.fftfreq(samples, 1/fs)
    return amp_spec[:int(len(f)/2)], f[:int(len(f)/2)]
