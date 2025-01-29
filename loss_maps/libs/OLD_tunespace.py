import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy import signal
from scipy.ndimage.filters import gaussian_filter

class scan_data(object):
    def __init__(self, filename, normalize = 'False', filter_type='None'):
        f=open(filename,'rb')
        data=pickle.load(f)
        f.close()
        self.data = data
        self.get_tunes()
        self.interpolate_data()
        self.find_beam_data()
        fil_data=self.filter_data(self.beam_data,filter_type)
        self.diff_data(fil_data, normalize)
        #self.build_contour_data()

    def get_tunes(self):
        htune = np.round(self.data[0]['htune'],3)
        vtune = np.round(self.data[0]['vtune'],3)
        if htune[0]==htune[-1]:
            self.scan_type='H'
        elif vtune[0]==vtune[-1]:
            self.scan_type='V'
        htune,vtune=[],[]
        for i in range(len(self.data.keys())):
            htune.append(np.round(self.data[i]['htune'],3))
            vtune.append(np.round(self.data[i]['vtune'],3))
        self.htune=np.array(htune)
        self.vtune=np.array(vtune)
        if self.scan_type=='H':
            self.scan_size=np.flatnonzero(self.htune[:,0] == self.htune[0,0]).size
        elif self.scan_type=='V':
            self.scan_size=np.flatnonzero(self.vtune[:,0] == self.vtune[0,0]).size

    def find_start_and_end(self,t,beam,time=[0.035,0.8]):
        start = np.argmin(np.abs(t-time[0]))
        end = np.argmin(np.abs(t-time[1]))
        d2y = np.gradient(np.gradient(beam))
        win=5
        start_of_beam = np.argmin(d2y[start-win:start+win]) + start - win+1
        end_of_beam = np.argmin(d2y[end-win:end+win]) + end - win -1
        return start_of_beam, end_of_beam

    def plot_window(self,t,beam,time=[0.035,0.8]):
        s,e = self.find_start_and_end(t,beam)
        plt.plot(beam)
        plt.plot([s]*2,[0,3],'--k')
        plt.plot([e]*2,[0,3],'--k')
        plt.show()

    def interpolate_data(self):
        interp_data=[]
        new_time = np.linspace(0.01,1.25,1000)
        for i in range(len(self.data.keys())):
            f=interp1d(self.data[i]['time'],self.data[i]['beam'])
            interp_data.append(f(new_time))
        self.new_time=new_time
        self.interp_data=np.array(interp_data)

    def find_beam_data(self):
        beam_data=[]
        start,end = self.find_start_and_end(self.new_time,self.interp_data[0])
        for i in range(self.interp_data.shape[0]):
            beam_data.append(self.interp_data[i][start:end+1])
        self.beam_data=np.array(beam_data)

    def diff_data(self,beam_data,normalise='False'):
        d_data=[]
        for i in range(beam_data.shape[0]):
            if normalise=='byI(t)':
                grad = np.gradient(beam_data[i])
                beam = beam_data[i]
                beam[np.array(beam)<0.01*np.max(beam)] = 0 
                diffnorm = np.divide(grad, beam, out=np.zeros_like(grad), where=beam!=0)
                #diffnorm = np.divide(grad, beam)
                #diffnorm[np.isnan(diffnorm)] = 0
                #d_data.append(np.gradient(beam_data[i]/np.max(beam_data[i])))
                d_data.append(diffnorm)
            elif normalise=='byI0':
                d_data.append(np.gradient(beam_data[i]/np.max(beam_data[i])))
            elif normalise=='False':
                d_data.append(np.gradient(beam_data[i]))
        self.d_data=np.array(d_data)

    def filter_data(self,data,filter_type='None'):
        fil_data=[]
        if filter_type=='None':
            fil_data=data
        elif filter_type=='gaussian':
            width=4
            for i in range(data.shape[0]):
                fil_data.append(gaussian_filter(data[i],width))
        elif filter_type=='butter':
            b, a = signal.butter(1, 0.03, analog=False)
            for i in range(data.shape[0]):
                fil_data.append(signal.filtfilt(b,a,data[i]))
        else:
            pass
        return np.array(fil_data)

    def build_contour_data(self):
        self.combined_data=np.zeros((int(self.d_data.shape[0]/self.scan_size),self.d_data.shape[1]))
        for i in range(self.scan_size):
            self.combined_data+=self.d_data[i::self.scan_size,:]
        if self.scan_type=='H':
            self.htune_scan = self.htune[::self.scan_size,0]
            self.vtune_scan = np.linspace(self.vtune[0][0],self.vtune[0][1],self.d_data.shape[1])
            self.contour_interp=interp2d(self.htune_scan,self.vtune_scan,self.combined_data.T)
        if self.scan_type=='V':
            self.vtune_scan = self.vtune[::self.scan_size,0]
            self.htune_scan = np.linspace(self.htune[0][0],self.htune[0][1],self.d_data.shape[1])
            self.contour_interp=interp2d(self.htune_scan,self.vtune_scan,self.combined_data)

    def build_intensity_data(self,size):
        self.combined_data=np.zeros((int(self.d_data.shape[0]/size),self.d_data.shape[1]))
        turns=np.linspace(2,14,13)
        for i in range(size):
            self.combined_data+=self.d_data[i::size,:]
        self.vtune_scan = np.linspace(self.vtune[0][0],self.vtune[0][1],self.d_data.shape[1])
        self.contour_interp=interp2d(turns,self.vtune_scan,self.combined_data.T)
