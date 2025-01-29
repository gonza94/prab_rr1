import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,a,x0,sigma):
    """
    This function returns the Gaussian function at x with amplitude (a), mean (x0) and standard deviation (sigma)           
    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def tune_tbt(bpm_data,ignore_terms=50):
    """
    This function returns the tune for a given measurement of a BPM based on the tbt data structure based on a FFT
    
    Parameters
    ----------
    bpm_data : float
        vector with the BPM reading for a given measurement     
    ignore_terms : int
        Integer that tells the code how many low-order frequency terms to drop in order to calculate tune 
    """    

    # Get the raw FFT from the BPM measurement, as well as raw tune space
    spect_raw=np.fft.fft(bpm_data)
    tunes_raw=np.arange(0,len(spect_raw))/(len(spect_raw))

    # Slice the FFT spectrum to drop the symmetric part and cancel the first low frequency terms
    spect=np.absolute(spect_raw[0:int(len(spect_raw)/2)])
    spect[0:ignore_terms]=0
    
    # Create the corresponding tune space
    tunes=tunes_raw[0:int(len(spect_raw)/2)]

    # Fit a Gaussian function to the data with some initial guess, assuming spectrum is centrally distributed
    # If Gaussian fit fails calculate tune just from maximum value in spectrum
    try:
        a0=1000
        tune0=tunes[np.argmax(spect)]
        s0=0.01
        popt,pcov = curve_fit(gaussian,tunes,spect,p0=[a0,tune0,s0])
        avg_tune=popt[1]
        
        return(avg_tune)
    
    except Exception:
        
        avg_tune=tunes[np.argmax(spect)]
        return(avg_tune)
        
def get_window(pos_array, threshold2 = 0.4, std_width = 2):
    """
    This function gets the initial and final turn in order to window the data and write SUSSIX files 
    It uses CUSUM (CUmulative SUM Statistics) of the Standard Deviation with weights to capture time series data
    
    Parameters
    ----------
    bpm_data : float
        Position array centered at 0 with the BPM reading for a given measurement     
      
    """   
    n1 = 200
    foundn1 = False
    n2 = 1000
    foundn2 = False
    
    cusumvec = []
    
    for i in np.arange(5, len(pos_array)):
        
        weights = np.logspace(0.01, 10, i)
        average = np.mean(pos_array[:i])
        sigmak = std_width*np.sqrt(np.average((pos_array[:i]-average)**2, weights=weights))
        
        if (sigmak > np.amax(pos_array)) & (not foundn1):
            n1 = i
            #print(n1)
            foundn1 = True
            
        if (sigmak < threshold2*np.amax(pos_array)) & (foundn1):
            n2 = i
            #print(n2)
            foundn2 = True
            break
    
        cusumvec = np.append(cusumvec, sigmak)
    return n1+5, n2, cusumvec

def plot_bpm_reading(bpm_data, title = 'BPM Reading'):
    """
    This function plots the BPM reading for one measurement
    
    Parameters
    ----------
    bpm_data : float
        vector with the BPM reading for a given measurement     
      
    """   

    fig,ax=plt.subplots(1,1,figsize=(12,6))

    ax.plot(bpm_data)

    ax.set_xlabel('Number of turns', fontsize=30)
    ax.set_ylabel('BPM Reading [mm]', fontsize=30)
    ax.set_title(title, fontsize=32)

    plt.tight_layout()
    
    return fig, ax

def plot_fft_bpm(bpm_data, tuneline = True, ignore_terms=50, plane = 'x'):
    """
    This function plots the FFT spectrum for a BPM reading for one measurement
    Returs the figure with the fractional tune vs amplitude of the FFT
    Parameters
    ----------
    bpm_data : float
        vector with the BPM reading for a given measurement
    tuneline : boolean
        Boolean telling matplotlib to plot tune line
    ignore_terms : int
        Integer that tells the code how many low-order frequency terms to drop in order to calculate tune       
    """
    # Get the raw FFT from the BPM measurement, as well as raw tune space
    spect_raw=np.fft.fft(bpm_data)
    tunes_raw=np.arange(0,len(spect_raw))/(len(spect_raw))

    # Slice the FFT spectrum to drop the symmetric part and cancel the 0-th term
    spect=np.absolute(spect_raw[0:int(len(spect_raw)/2)])
    spect[0:ignore_terms]=0
    
    # Create the corresponding tune space
    tunes=tunes_raw[0:int(len(spect_raw)/2)]
    
    fig,ax3=plt.subplots(1,1,figsize=(12,6))

    ax3.plot(tunes,spect)
    
    tu=tune_tbt(bpm_data,ignore_terms)
    
    if tuneline:
        ax3.axvline(tu,c='r',linestyle='dashed',label=r'$\nu_%s=$%.3f'%(plane,tu) )
        ax3.legend(fontsize=26)

    ax3.set_ylabel('Amplitude', fontsize=30)
    ax3.set_xlabel('Fractional Tune', fontsize=30)

    plt.grid()
    plt.tight_layout()
    
    return fig,ax3


def ellipse_parametric(beta=1,alpha=0,emit=1):
    '''
    Returns (z,zp) for an ellipse with the given beta, alpha and emittance using parametric definition
    For more info see:
    https://arxiv.org/pdf/1101.3649.pdf
    https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotatio    
    '''
        
    # Calculate third Twiss parameter, tilt angle and radii (major and minor for ellipse)
    gamma=(1+alpha**2)/beta
    if beta!=1:
        angle=0.5*np.arctan(2*alpha/(gamma-beta))
    else:
        angle=0
    
    cc=np.cos(angle)
    ss=np.sin(angle)
    rx=np.sqrt(emit*beta)
    rxp=np.sqrt(emit/beta)
    
    # Create parametric variable
    t=np.linspace(0,2*np.pi,200)
    
    # Create z and zp data for ellipse
    z=rx*cc*np.cos(t)-rxp*ss*np.sin(t)
    zp=rx*ss*np.cos(t)+rxp*cc*np.sin(t)
    
    return z,zp
    
def plot_phase_space(dataind,window_data,plane='H'):
    '''
    Returns figure with plot for the x-xprime phase space.
    It uses previously defined dict entries to call ellipse funtion plot
    ---------                      
    dataind: dict                                                            
    Dictionary with tbtfit data for one BPM, e.g.,dataind=frame_h.bpm_fit['R:HP414']
    
    window_data: array,float                                                   
    2D array with initial and final frame used to fit lattice functions
    '''
    fig,ax1=plt.subplots(1,1,figsize=(8,8))

    # Create the ellipse from the fitted parameters                                
    xfit,xpfit=ellipse_parametric(beta=dataind['beta'],
                                  alpha=dataind['alpha'],
                                  emit=dataind['emit'])
    if plane=='H':
        labelfit=r"""$\beta_x$=%.2f [m]
$\alpha_x$=%.2f
$\epsilon_x$=%.2f [$\pi$ mm mrad]"""%(dataind['beta'],dataind['alpha'],dataind['emit'])

        ax1.plot(xfit,xpfit,c='r',label=labelfit)
        ax1.scatter(dataind['x'][window_data[0]:window_data[1]],
                dataind['xp'][window_data[0]:window_data[1]],label='Data')

        ax1.set_xlabel('$x$ [mm]',fontsize=26)
        ax1.set_ylabel('$x^{\prime}$ [mrad]',fontsize=26)


    elif plane=='V':
        labelfit=r"""$\beta_y$=%.2f [m]
$\alpha_y$=%.2f
$\epsilon_y$=%.2f [$\pi$ mm mrad]"""%(dataind['beta'],dataind['alpha'],dataind['emit'])
        ax1.plot(xfit,xpfit,c='r',label=labelfit)
        ax1.scatter(dataind['y'][window_data[0]:window_data[1]],
                dataind['yp'][window_data[0]:window_data[1]],label='Data')

        ax1.set_xlabel('$y$ [mm]',fontsize=26)
        ax1.set_ylabel('$y^{\prime}$ [mrad]',fontsize=26)


    ax1.legend(fontsize=16)

    plt.tight_layout()

    return fig



def plot_both_norm(dataind,window_data, planee = 'H'):
    '''
    Returns figure with two plots, one for the regular phase space x and xprime,
    and one for the normalized phase space, windowed by window_data. It uses 
    the previously defined dict from tbtfit to call ellipse function plot.
    ----------
    dataind: dict
    Dictionary with tbtfit data for one BPM, e.g.,dataind=frame_h.bpm_fit['R:HP414']

    window_data: array,float
    2D array with initial and final frame used to fit lattice functions
    '''

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6))

    # Create the ellipse from the fitted parameters 
    xfit,xpfit=ellipse_parametric(beta=dataind['beta'],
                                  alpha=dataind['alpha'],
                                  emit=dataind['emit'])
    
    if planee == 'H':
        labelfit=r"""$\beta_x$=%.2f [m]
$\alpha_x$=%.2f 
$\epsilon_x$=%.2f [$\pi$ mm mrad]"""%(dataind['beta'],dataind['alpha'],dataind['emit'])
        
        ax1.plot(xfit,xpfit,c='r',label=labelfit)
        
        ax1.scatter(dataind['x'][window_data[0]:window_data[1]],
                    dataind['xp'][window_data[0]:window_data[1]],label='Data')

        ax1.tick_params(labelsize=18)
        ax1.set_xlabel('$x$ [mm]',fontsize=26)
        ax1.set_ylabel('$x^{\prime}$ [mrad]',fontsize=26)

    elif planee == 'V':
        labelfit=r"""$\beta_y$=%.2f [m]
$\alpha_y$=%.2f 
$\epsilon_y$=%.2f [$\pi$ mm mrad]"""%(dataind['beta'],dataind['alpha'],dataind['emit'])
        
        ax1.plot(xfit,xpfit,c='r',label=labelfit)
        
        ax1.scatter(dataind['y'][window_data[0]:window_data[1]],
                    dataind['yp'][window_data[0]:window_data[1]],label='Data')

        ax1.tick_params(labelsize=18)
        ax1.set_xlabel('$y$ [mm]',fontsize=26)
        ax1.set_ylabel('$y^{\prime}$ [mrad]',fontsize=26)

    ax1.legend(fontsize=18)

    if planee == 'H':
        ax2.scatter(dataind['x_norm'][window_data[0]:window_data[1]]*0.001,
                    dataind['xp_norm'][window_data[0]:window_data[1]]*0.001,label='Data')

        ax2.tick_params(labelsize=18)
        ax2.set_xlabel('$x_N$ [m$^{1/2}$]',fontsize=26)
        ax2.set_ylabel('$x^{\prime}_N$ [m$^{1/2}$]',fontsize=26)

    elif planee == 'V':
        ax2.scatter(dataind['y_norm'][window_data[0]:window_data[1]]*0.001,
                    dataind['yp_norm'][window_data[0]:window_data[1]]*0.001,label='Data')
        
        ax2.tick_params(labelsize=18)
        ax2.set_xlabel('$y_N$ [m$^{1/2}$]',fontsize=26)
        ax2.set_ylabel('$y^{\prime}_N$ [m$^{1/2}$]',fontsize=26)

    # Create the circle for the normalized data
    xnfit,xpnfit=ellipse_parametric(beta=1,alpha=0,emit=dataind['emit'])
    #ax2.plot(xnfit*0.001,xpnfit*0.001,c='r',label= r'Normalized fit'%dataind['emit'])


    ax2.legend(fontsize=18)

    #ax2.set_xlim(-1.1*max(xnfit*0.001),1.1*max(xnfit*0.001))
    #ax2.set_ylim(-1.1*max(xpnfit*0.001),1.1*max(xpnfit*0.001))

    plt.tight_layout()

    return fig
