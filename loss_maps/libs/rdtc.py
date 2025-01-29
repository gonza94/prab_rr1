# RDTC

# Resonance Driving Terms Coefficients


import numpy as np
import pandas as pd

def hline_to_coeff(df10, dfv, dfhline, j = 3, k = 0, l = 0, m = 0):
	"""
    This function returns a dataframe with the corresponding generating function
    coeffcients and Hamiltonian coeffcients as calculated from theory of
    resonance driving terms.    
    
    Parameters
    ----------
    df10 : pandas Dataframe
    	Dataframe containing ['tune','line','amplitude','phase','error'] for the 
    	horizontal tune line as imported from a SUSSIX output file
    	
    dfv : pandas Dataframe
    	Dataframe containing ['bpm','tune','amplitude','phase','error'] for the 
    	vertical tune line as imported from a SUSSIX output file
    	
    dfhline : pandas Dataframe
    	Dataframe containing ['tune','line','amplitude','phase','error'] for the 
    	horizontal line corresponding to the (1-j+k, m-l) spectral line
    	as imported from a SUSSIX output file
    	
    j,k,l,m : int
    	Indices defining the coefficients
    
    Output
    ----------
    Returns dataframe with the generating function coeffcients as well as 
    the Hamiltonian coeffcients with the following columns:
    	'sext': Label for any powered sextupole
    	'sext-current': Current for powered sextupole if any
    	'tune-x' : Horizontal tune
    	'tune-y' : Vertical tune
    	'Ix' and 'Iy' : Invariants of motion in each direction
    	'abs(f_jklm)' and 'phi_jklm': Amplitude and phase for the generating function coefficient
    	'abs(h_jklm)' and 'psi_jklm': Amplitude and phase for the Hamiltonian coefficient
    	'error': Error of resonance lines
    	'bpm': Label for the correspoding BPM
    	'frame': Frame number inside tbt file
    	'int_avg' : Average intensity data from intensity bpm
    	'int-std' : Standard Deviation intensity data from intensity bpm
    """   
	
	dfjklm = pd.DataFrame(columns = ['sext', 'tune-x', 'tune-y', 'Ix', 'Iy', 'phihat', 'abs(f_jklm)', 'phi_jklm', 'abs(h_jklm)', 'psi_jklm', 'error', 'plane', 'bpm', 'frame', 'sext-current','int_avg','int_std'])
	
	dfjklm['sext'] = df10['sext']
	dfjklm['sext-current'] = df10['sext-current']
	
	dfjklm['int_avg'] = df10['int_avg']
	dfjklm['int_std'] = df10['int_std']
	
	dfjklm['tune-x'] = df10['tune']
	dfjklm['tune-y'] = dfv['tune']
	
	dfjklm['Ix'] = np.square(df10['amplitude'])*0.5
	dfjklm['Iy'] = np.square(dfv['amplitude'])*0.5
	
	dfjklm['phihat'] = np.pi*((j-k)*dfjklm['tune-x']+(l-m)*dfjklm['tune-y'])
	
	deno = 2.0*j*np.multiply(np.power(df10['amplitude'], j+k-1), np.power(dfv['amplitude'], l+m))
	num = np.abs(dfhline['amplitude'])
	
	dfjklm['abs(f_jklm)'] = np.divide(num, deno)
	dfjklm['phi_jklm'] = (dfhline['phase']*np.pi/180.)-((1-j+k)*df10['phase']*np.pi/180.)+((l-m)*dfv['phase']*np.pi/180.)+np.pi/2.0
	
	mult1 = np.multiply(dfjklm['abs(f_jklm)'], np.exp(1.j*dfjklm['phi_jklm']))
	mult2 = 1.0-np.exp(2*np.pi*1.j*((j-k)*dfjklm['tune-x']+(l-m)*dfjklm['tune-y']))
	
	hjklm = np.multiply(mult1, mult2)
	
	dfjklm['abs(h_jklm)'] = np.abs(hjklm)
	dfjklm['psi_jklm'] = np.angle(hjklm)
	
	dfjklm['error'] = df10['error'] + dfhline['error'] + dfv['error']
	dfjklm['bpm'] = df10['bpm']
	
	funxx = lambda s: 'H' if s[2] == 'H' else 'V'
	dfjklm['plane'] = [funxx(bi) for bi in np.array(dfjklm['bpm'])]
	dfjklm['frame'] = df10['frame']
	
	return dfjklm

def hline_to_coeff2(df10, dfv, dfhline, j = 3, k = 0, l = 0, m = 0):
	"""
    This function returns a dataframe with the corresponding generating function
    coefficients and Hamiltonian coeffcients as calculated from theory of
    resonance driving terms.    
    
    Parameters
    ----------
    df10 : pandas Dataframe
    	Dataframe containing ['tune','line','amplitude','phase','error'] for the 
    	horizontal tune line as imported from a SUSSIX output file
    	
    dfv : pandas Dataframe
    	Dataframe containing ['tune','amplitude','phase','error'] for the 
    	vertical tune line as imported from a SUSSIX output file
    	
    dfhline : pandas Dataframe
    	Dataframe containing ['tune','line','amplitude','phase','error'] for the 
    	horizontal line corresponding to the (1-j+k, m-l) spectral line
    	as imported from a SUSSIX output file
    	
    j,k,l,m : int
    	Indices defining the coefficients
    
    Output
    ----------
    Returns dataframe with the generating function coeffcients as well as 
    the Hamiltonian coeffcients with the following columns:
    	'tune-x' : Horizontal tune
    	'tune-y' : Vertical tune
    	'Ix' and 'Iy' : Invariants of motion in each direction
    	'abs(f_jklm)' and 'phi_jklm': Amplitude and phase for the generating function coefficient
    	'abs(h_jklm)' and 'psi_jklm': Amplitude and phase for the Hamiltonian coefficient
    	'error': Error of resonance lines
    	'bpm': Label for the correspoding BPM
    	'frame': Frame number inside tbt file 
    	'isc220', 'isc222', 'isc319', 'isc321': Currents for powered sextupole 
    """   
	
	dfjklm = pd.DataFrame(columns = ['tune-x', 'tune-y', 'Ix', 'Iy', 'phihat', 'abs(f_jklm)', 'phi_jklm', 'abs(h_jklm)', 'psi_jklm', 'error', 'plane', 'bpm', 'frame', 'isc220', 'isc222', 'isc319', 'isc321'])
	
	# These currents should be specified by user
	dfjklm['isc220'] = df10['isc220']
	dfjklm['isc222'] = df10['isc222']
	dfjklm['isc319'] = df10['isc319']
	dfjklm['isc321'] = df10['isc321']	
	
	dfjklm['tune-x'] = df10['tune']
	dfjklm['tune-y'] = dfv['tune']
	
	dfjklm['Ix'] = np.square(df10['amplitude'])*0.5
	dfjklm['Iy'] = np.square(dfv['amplitude'])*0.5
	
	dfjklm['phihat'] = np.pi*((j-k)*dfjklm['tune-x']+(l-m)*dfjklm['tune-y'])
	
	deno = 2.0*j*np.multiply(np.power(df10['amplitude'], j+k-1), np.power(dfv['amplitude'], l+m))
	num = np.abs(dfhline['amplitude'])
	
	dfjklm['abs(f_jklm)'] = np.divide(num, deno)
	dfjklm['phi_jklm'] = (dfhline['phase']*np.pi/180.)-((1-j+k)*df10['phase']*np.pi/180.)+((l-m)*dfv['phase']*np.pi/180.)+np.pi/2.0
	
	mult1 = np.multiply(dfjklm['abs(f_jklm)'], np.exp(1.j*dfjklm['phi_jklm']))
	mult2 = 1.0-np.exp(2*np.pi*1.j*((j-k)*dfjklm['tune-x']+(l-m)*dfjklm['tune-y']))
	
	hjklm = np.multiply(mult1, mult2)
	
	dfjklm['abs(h_jklm)'] = np.abs(hjklm)
	dfjklm['psi_jklm'] = np.angle(hjklm)
	
	dfjklm['error'] = df10['error'] + dfhline['error'] + dfv['error']
	dfjklm['bpm'] = df10['bpm']
	
	funxx = lambda s: 'H' if s[2] == 'H' else 'V'
	dfjklm['plane'] = [funxx(bi) for bi in np.array(dfjklm['bpm'])]
	dfjklm['frame'] = df10['frame']
	
	return dfjklm 
	
def hline_to_coeff3(df10, dfv, dfhline, j = 3, k = 0, l = 0, m = 0):
	"""
    This function returns a dataframe with the corresponding generating function
    coefficients and Hamiltonian coeffcients as calculated from theory of
    resonance driving terms.    
    
    Parameters
    ----------
    df10 : pandas Dataframe
    	Dataframe containing ['tune','line','amplitude','phase','error'] for the 
    	horizontal tune line as imported from a SUSSIX output file
    	
    dfv : pandas Dataframe
    	Dataframe containing ['tune','amplitude','phase','error'] for the 
    	vertical tune line as imported from a NAFF output file
    	
    dfhline : pandas Dataframe
    	Dataframe containing ['tune','line','amplitude','phase','error'] for the 
    	horizontal line corresponding to the (1-j+k, m-l) spectral line
    	as imported from a SUSSIX output file
    	
    j,k,l,m : int
    	Indices defining the coefficients
    
    Output
    ----------
    Returns dataframe with the generating function coeffcients as well as 
    the Hamiltonian coeffcients with the following columns:
    	'tune-x' : Horizontal tune
    	'tune-y' : Vertical tune
    	'Ix' and 'Iy' : Invariants of motion in each direction
    	'abs(f_jklm)' and 'phi_jklm': Amplitude and phase for the generating function coefficient
    	'abs(h_jklm)' and 'psi_jklm': Amplitude and phase for the Hamiltonian coefficient
    	'error': Error of resonance lines
    	'bpm': Label for the correspoding BPM
    	'frame': Frame number inside tbt file 
    """   
	
	dfjklm = pd.DataFrame(columns = ['tune-x', 'tune-y', 'Ix', 'Iy', 'phihat', 'abs(f_jklm)', 'phi_jklm', 'abs(h_jklm)', 'psi_jklm', 'error', 'plane', 'bpm', 'frame'])
	
	dfjklm['tune-x'] = df10['tune']
	dfjklm['tune-y'] = dfv['tune']
	
	dfjklm['Ix'] = np.square(df10['amplitude'])*0.5
	dfjklm['Iy'] = np.square(dfv['amplitude'])*0.5
	
	dfjklm['phihat'] = np.pi*((j-k)*dfjklm['tune-x']+(l-m)*dfjklm['tune-y'])
	
	deno = 2.0*j*np.multiply(np.power(df10['amplitude'], j+k-1), np.power(dfv['amplitude'], l+m))
	num = np.abs(dfhline['amplitude'])
	
	dfjklm['abs(f_jklm)'] = np.divide(num, deno)
	dfjklm['phi_jklm'] = (dfhline['phase']*np.pi/180.)-((1-j+k)*df10['phase']*np.pi/180.)+((l-m)*dfv['phase']*np.pi/180.)+np.pi/2.0
	
	mult1 = np.multiply(dfjklm['abs(f_jklm)'], np.exp(1.j*dfjklm['phi_jklm']))
	mult2 = 1.0-np.exp(2*np.pi*1.j*((j-k)*dfjklm['tune-x']+(l-m)*dfjklm['tune-y']))
	
	hjklm = np.multiply(mult1, mult2)
	
	dfjklm['abs(h_jklm)'] = np.abs(hjklm)
	dfjklm['psi_jklm'] = np.angle(hjklm)
	
	dfjklm['error'] = df10['error'] + dfhline['error'] + dfv['error']
	dfjklm['bpm'] = df10['bpm']
	
	funxx = lambda s: 'H' if s[2] == 'H' else 'V'
	dfjklm['plane'] = [funxx(bi) for bi in np.array(dfjklm['bpm'])]
	dfjklm['frame'] = df10['frame']
	
	return dfjklm 	

def vline_to_coeff(dfh, df01, dfvline, j = 0, k = 0, l = 3, m = 0):
	"""
    This function returns a dataframe with the corresponding generating function
    coeffcients and Hamiltonian coeffcients as calculated from theory of
    resonance driving terms.    
    
    Parameters
    ----------
    dfh : pandas Dataframe
    	Dataframe containing ['tune','amplitude','phase','error'] for the 
    	horizontal tune line as imported from a NAFF output file
    	
    df01 : pandas Dataframe
    	Dataframe containing ['tune','line','amplitude','phase','error'] for the 
    	vertical tune line as imported from a SUSSIX output file
    	
    dfvline : pandas Dataframe
    	Dataframe containing ['tune','line','amplitude','phase','error'] for the 
    	vertical line corresponding to the (1-j+k, m-l) spectral line
    	as imported from a SUSSIX output file
    	
    j,k,l,m : int
    	Indices defining the coefficients
    
    Output
    ----------
    Returns dataframe with the generating function coeffcients as well as 
    the Hamiltonian coeffcients with the following columns:
    	'sext': Label for any powered sextupole
    	'sext-current': Current for powered sextupole if any
    	'tune-x' : Horizontal tune
    	'tune-y' : Vertical tune
    	'Ix' and 'Iy' : Invariants of motion in each direction
    	'abs(f_jklm)' and 'phi_jklm': Amplitude and phase for the generating function coefficient
    	'abs(h_jklm)' and 'psi_jklm': Amplitude and phase for the Hamiltonian coefficient
    	'error': Error of resonance lines
    	'bpm': Label for the correspoding BPM
    	'frame': Frame number inside tbt file 
   		'int_avg' : Average intensity data from intensity bpm
    	'int-std' : Standard Deviation intensity data from intensity bpm
    """   
	
	dfjklm = pd.DataFrame(columns = ['sext', 'tune-x', 'tune-y', 'Ix', 'Iy', 'phihat', 'abs(f_jklm)', 'phi_jklm', 'abs(h_jklm)', 'psi_jklm', 'error', 'plane', 'bpm', 'frame', 'sext-current','int_avg','int_std'])
	
	dfjklm['sext'] = df01['sext']
	dfjklm['sext-current'] = df01['sext-current']
	
	dfjklm['tune-x'] = dfh['tune']
	dfjklm['tune-y'] = df01['tune']
	
	dfjklm['int_avg'] = df01['int_avg']
	dfjklm['int_std'] = df01['int_std']
	
	dfjklm['Ix'] = np.square(dfh['amplitude'])*0.5
	dfjklm['Iy'] = np.square(df01['amplitude'])*0.5
	
	dfjklm['phihat'] = np.pi*((j-k)*dfjklm['tune-x']+(l-m)*dfjklm['tune-y'])
	
	deno = 2.0*l*np.multiply(np.power(dfh['amplitude'], j+k), np.power(df01['amplitude'], l+m-1))
	num = np.abs(dfvline['amplitude'])
	
	dfjklm['abs(f_jklm)'] = np.divide(num, deno)
	dfjklm['phi_jklm'] = (dfvline['phase']*np.pi/180.)+((j-k)*dfh['phase']*np.pi/180.)-((1-l+m)*df01['phase']*np.pi/180.)+np.pi/2.0
	
	mult1 = np.multiply(dfjklm['abs(f_jklm)'], np.exp(1.j*dfjklm['phi_jklm']))
	mult2 = 1-np.exp(2*np.pi*1.j*((j-k)*dfjklm['tune-x']+(l-m)*dfjklm['tune-y']))
	
	hjklm = np.multiply(mult1, mult2)
		
	dfjklm['abs(h_jklm)'] = np.abs(hjklm)
	dfjklm['psi_jklm'] = np.angle(hjklm)
	
	dfjklm['error'] = dfh['error'] + dfvline['error'] + df01['error']
	dfjklm['bpm'] = df01['bpm']
	
	funxx = lambda s: 'H' if s[2] == 'H' else 'V'
	dfjklm['plane'] = [funxx(bi) for bi in np.array(dfjklm['bpm'])]
	dfjklm['frame'] = df01['frame']
	
	return dfjklm
 	
 	
 	
 	
                                     
    
                                        
    

