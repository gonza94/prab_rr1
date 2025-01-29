import numpy as np
import h5py
import os

np.seterr(invalid = 'raise')

class h5fit_single(object):
    """
    This class imports h5 files from synergia simulations with a single observation point.
    Further methods allow for Twiss parameter calculation and normalized coordinates generation. 


    """

    def __init__(self, h5file, which_plane = 'x'):
        # Import data from h5 file and save onto dict
        h5data = h5py.File(h5file,'r')
        self.data = dict({'x' : np.array(h5data['mean'][:,0]),
                          'xp': np.array(h5data['mean'][:,1]),
                          'y' : np.array(h5data['mean'][:,2]),
                          'yp': np.array(h5data['mean'][:,3])})
        self.frame = 0 
        if which_plane in ['x','y','both']:
            self.which_plane = which_plane
        else:
            raise ValueError('Must specify which plane to be analyzed, can be x, y or both, i.e. which_plane="x"')
        

    def zeroAverage(self, wzero = [500,1500]):
        # Average out to zero all coordinates
        self.data_avg = dict({'x' : self.data['x']-np.mean(self.data['x'][wzero[0]:wzero[1]]),
                              'xp': self.data['xp']-np.mean(self.data['xp'][wzero[0]:wzero[1]]),
                              'y' : self.data['y']-np.mean(self.data['y'][wzero[0]:wzero[1]]),
                              'yp': self.data['yp']-np.mean(self.data['yp'][wzero[0]:wzero[1]])})
        pass
    
        
    def calculate_twiss(self,window = [50,30]):
        # Calculate Twiss parameters for both transverse coordinates at the observation point
        self.goodfit = True
        if self.which_plane == 'x':
            try:
                alphax,betax,emitx = fitEllipseMJ(self.data['x'], self.data['xp'], window)
                self.datanorm = dict({'x_n' : self.data['x']/np.sqrt(betax),
                                      'xp_n': self.data['x']*(alphax/np.sqrt(betax))+np.sqrt(betax)*self.data['xp']})

                self.twiss = dict({'betax' : betax, 'alphax' : alphax, 'emitx' : emitx})
                
            except ValueError:
                self.goodfit = False
                print('Bad fit')
                raise

        elif self.which_plane == 'y':
            try:
                alphay,betay,emity = fitEllipseMJ(self.data['y'], self.data['yp'], window)
                self.datanorm = dict({'y_n' : self.data['y']/np.sqrt(betay),
                                      'yp_n': self.data['y']*(alphay/np.sqrt(betay))+np.sqrt(betay)*self.data['yp']})

                self.twiss = dict({'betay' : betax, 'alphay' : alphay, 'emity' : emity})
                
            except:
                self.goodfit = False
                print('Bad fit')
                raise

        elif self.which_plane == 'both':        
            try:
        
                alphax,betax,emitx = fitEllipseMJ(self.data['x'], self.data['xp'], window)
                alphay,betay,emity = fitEllipseMJ(self.data['y'], self.data['yp'], window)

                self.datanorm = dict({'x_n' : self.data['x']/np.sqrt(betax),
                                      'xp_n': self.data['x']*(alphax/np.sqrt(betax))+np.sqrt(betax)*self.data['xp'],
                                      'y_n' : self.data['y']/np.sqrt(betay),
                                      'yp_n': self.data['y']*(alphay/np.sqrt(betay))+np.sqrt(betay)*self.data['yp']})

                self.twiss = dict({'betax' : betax, 'alphax' : alphax, 'emitx' : emitx, 'betay' : betax, 'alphay' : alphay, 'emity' : emity} )
            
            except:
                self.goodfit = False
                print('Bad fit')
                raise

        pass
            
    def writefiles4sussix(self, working_dir, twodims = False, onlyhori = True, onlyvert = False):        
        # Write files for SUSSIX analysis
        output_dir=working_dir+'/frame'+str(self.frame)

        if np.sum([twodims,onlyhori,onlyvert])>1:
            raise ValueError('Can only set one option True. Choose between onlyhori, onlyvert or twodims')
        
        try:
            os.mkdir(output_dir)
        except:
            pass

        # If fit was good, create SUSSIX file for analysis depending on case
        if self.goodfit:
            i = 1
            if onlyhori:
                to_write = np.vstack([self.datanorm['x_n'],self.datanorm['xp_n']]).T
                filname = output_dir+'/bpm.%.3i' %i
                np.savetxt(filname,to_write)

            elif onlyvert:
                to_write = np.vstack([self.datanorm['y_n'],self.datanorm['yp_n']]).T
                filname = output_dir+'/bpm.%.3i' %i
                np.savetxt(filname,to_write)

            elif twodims:
                to_write = np.vstack([self.datanorm['x_n'],self.datanorm['xp_n'],self.datanorm['y_n'],self.datanorm['yp_n']]).T
                filname = output_dir+'/bpm.%.3i' %i
                np.savetxt(filname,to_write)
            
        

def fitEllipseMJ(x,xp,window=[50,30]):
    # Function that fits ellipse to phase space data
    # Helps retrieve Twiss parameters
    
    x=x[window[0]:window[0]+window[1]]
    xp=xp[window[0]:window[0]+window[1]]
    A=np.zeros((3,3))
    U = np.zeros(3)
    A[0,0]=np.sum((x**2)*(x**2))
    A[0,1]=np.sum(x*xp*(x**2))
    A[0,2]=np.sum(x**2)
    A[1,0]=np.sum((x**2)*x*xp)
    A[1,1]=np.sum(x*xp*x*xp)
    A[1,2]=np.sum(x*xp)
    A[2,0]=np.sum(x**2)
    A[2,1]=np.sum(x*xp)
    A[2,2]=np.sum(np.ones(x.size))
    U[0]=np.sum(xp**2 * x**2)
    U[1]=np.sum(xp**2 * x* xp)
    U[2]=np.sum(xp**2 )
    A,B,C=np.dot(np.linalg.pinv(A),U)
    beta=1/np.sqrt(-B**2/4 -  A)
    alpha=beta*-B/2
    emit=C*beta
    return alpha, beta, emit

