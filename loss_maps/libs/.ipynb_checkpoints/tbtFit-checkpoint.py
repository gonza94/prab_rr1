import numpy as np
from numpy.linalg import eig, inv
from tbt import tbtReader
import svd_clean
import os
import pandas as pd
import pickle
import NAFFlib

def txt2array(matFile,machine,just_bpms=True,old=False):
    '''Extract transfer matrix from txt file copied from acnet console'''
    bpm_machine_dict = {'mi':['I:HP','I:VP'],'rr':['R:HP','R:VP']}
    hbn = bpm_machine_dict[machine][0]
    vbn = bpm_machine_dict[machine][1]
    f=open(matFile,'r')
    lines=f.readlines()
    f.close()
    newlines=[]
    names=[]
    for line in lines[1:]:
        if len(line.split())==8:
            newlines.append(line.split()[1:])
            names.append(line.split()[0])
        elif len(line.split())==7:
            newlines.append(line.split())
    mat=np.array(newlines,dtype=float).reshape((len(names),7,7))
    names_b, mat_b = [],[]
    mat_dict = {}
    if just_bpms==True:
        for i in range(len(names)):
            if (hbn in names[i] or vbn in names[i]):
                names_b.append(names[i])
                mat_b.append(mat[i,:,:])
                mat_dict[names[i]]=mat[i,:,:]
        mat=np.array(mat_b)
        names=names_b
    if old==True:
        return mat, names
    else:
        return mat_dict 

def transferMat(matrixfile,dim='x'):
    mat, names= txt2array(matrixfile)
    T = np.zeros((mat.shape[0],3,3))
    T[:,:2,:2] = mat[:,:2,:2]
    T[:,:2,2] = mat[:,:2,5]
    T[:,2,2]=1.0
    return T

class tbtFit(object):
    def __init__(self, latticeFile, matrixFile, frame=10, plane='H'):
        self.data = tbtReader(latticeFile)
        self.ignore_list=[]
        if self.data.header.machine==2:
            self.machine='mi'
            self.mi_ignore_offplane()
        elif self.data.header.machine==3:
            self.machine='rr'
        self.mat=txt2array(matrixFile,self.machine)
        self.frame=frame
        self.plane=plane
        self.averaged=False
        self.buildMats()

    def check_bad_bpms(self,badlim=5):
        # Should be called before LatticeScan
        # badlim should be set to error limit wanted
        
        for key in self.data.bpm.keys():
            badcount=np.sum(self.data.bpm[key][self.frame] == 999.)
            if badcount>badlim:
                self.ignore_list.append(key)
                print('Removing bad BPM at:')
                print(self.frame,key)

            
    def mi_ignore_offplane(self):
        offplane = ['I:VP222','I:VP402','I:VP522','I:VP608','I:VP620','I:HP101','I:HP321']
        for obpm in offplane:
            self.ignore_list.append(obpm)

    def cleanData(self,num):
        ##RR only atm##
        hmat=np.zeros((104,2048))
        vmat=np.zeros((104,2048))
        i=0
        for hbpm in self.data.Hmap:
            hmat[i,:]=self.data.bpm[hbpm][self.frame,:]
            i+=1
        i=0
        for vbpm in self.data.Vmap:
            vmat[i,:]=self.data.bpm[vbpm][self.frame,:]
            i+=1
        clean_hbpm = svd_clean.svd_decomposition(hmat,num)
        clean_vbpm = svd_clean.svd_decomposition(vmat,num)
        i=0
        for hbpm in self.data.Hmap:
            self.data.bpm[hbpm][self.frame,:]=clean_hbpm[i] 
            i+=1
        i=0
        for vbpm in self.data.Vmap:
            self.data.bpm[vbpm][self.frame,:]=clean_vbpm[i] 
            i+=1

    def buildMats(self):
        self.Hmat={}
        self.Vmat={}
        for bpm in self.mat.keys():
            hbpm=np.zeros((3,3))
            vbpm=np.zeros((2,2))
            hbpm[0,:2]=self.mat[bpm][0,:2]
            hbpm[0,2]=self.mat[bpm][0,5]
            hbpm[1,:2]=self.mat[bpm][1,:2]
            hbpm[1,2]=self.mat[bpm][1,5]
            hbpm[2,2]=1
            vbpm[0,:] = self.mat[bpm][2,2:4]
            vbpm[1,:] = self.mat[bpm][3,2:4]
            self.Hmat[bpm]=hbpm
            self.Vmat[bpm]=vbpm

    def bpms2use(self, bpm, window=10):
        start,end=0,2048
        self.bpm_in_use=bpm
        bpms_in_range=False
        valid_bpm=False 
        bpm_list=[]
        Hmap = self.data.Hmap
        Vmap = self.data.Vmap
        Hmap = [bpm for bpm in Hmap if bpm not in self.ignore_list]
        Vmap = [bpm for bpm in Vmap if bpm not in self.ignore_list]
        if self.machine=='rr':
            Hmap = reorderBpms(Hmap, 'R:HP602')
            Vmap = reorderBpms(Vmap, 'R:VP601')
        elif self.machine=='mi':
            Hmap = reorderBpms(Hmap, 'I:HP624')
            Vmap = reorderBpms(Vmap, 'I:VP625')
        if self.plane=='H':
            bpm_map = Hmap
        elif self.plane=='V':
            bpm_map = Vmap
        try:
            if 'HP' in bpm:
                start=Hmap.index(bpm)
            elif 'VP' in bpm:
                start=Vmap.index(bpm)
            valid_bpm=True
        except ValueError:
            print(bpm, 'is not a valid bpm')
        if valid_bpm==True:
            if (start-window)<0:
                #print('Window too close to boundary, shifting window')
                dif = start-window
                self.bpm_list=bpm_map[(start-window-dif):(start+window-dif)]
            elif (start+window)>len(bpm_map):
                #print('Window too close to boundary, shifting window')
                dif = len(bpm_map)-(start+window)
                self.bpm_list=bpm_map[(start-window+dif):(start+window+dif)]
            else:
                self.bpm_list = bpm_map[start-window:start+window]
            #print('Using bpms', self.bpm_list)

    def zerosAverage(self, turnStart, numberOfTurns):
        self.bpm_av ={}
        self.averages={}
        for key in self.data.bpm.keys():
            self.averages[key]=np.mean(self.data.bpm[key][self.frame,turnStart:turnStart+numberOfTurns])
            self.bpm_av[key] = self.data.bpm[key][self.frame,:]-self.averages[key]
        self.averaged=True
        if self.averaged==False:
            pass
        #else:
            #print('Averaging for frame '+str(self.frame))

    def tfit(self):
        BPM = np.zeros((2048, len(self.bpm_list)))
        T = np.zeros((len(self.bpm_list),3))
        i=0
        for bpm  in self.bpm_list:
            BPM[:,i]=self.bpm_av[bpm]
            T[i] = self.Hmat[bpm][0]
            i+=1
        A = np.zeros((3,3))
        A[0,0] = np.sum(T[:,0]*T[:,0])
        A[0,1] = np.sum(T[:,1]*T[:,0])
        A[0,2] = np.sum(T[:,2]*T[:,0])
        A[1,0] = np.sum(T[:,0]*T[:,1])
        A[1,1] = np.sum(T[:,1]*T[:,1])
        A[1,2] = np.sum(T[:,2]*T[:,1])
        A[2,0] = np.sum(T[:,0]*T[:,2])
        A[2,1] = np.sum(T[:,1]*T[:,2])
        A[2,2] = np.sum(T[:,2]*T[:,2])
        AI = np.linalg.pinv(A)
        U = np.dot(BPM,T)
        x=np.dot(AI,U.T)
        error=np.std(BPM-np.dot(T, x).T,axis=1)
        return x,error

    def tfit_y(self):
        BPM = np.zeros((2048, len(self.bpm_list)))
        T = np.zeros((len(self.bpm_list),2))
        i=0
        for bpm  in self.bpm_list:
            BPM[:,i]=self.bpm_av[bpm]
            T[i] = self.Vmat[bpm][0]
            i+=1
        A = np.zeros((2,2))
        A[0,0] = np.sum(T[:,0]*T[:,0])
        A[0,1] = np.sum(T[:,1]*T[:,0])
        A[1,0] = np.sum(T[:,0]*T[:,1])
        A[1,1] = np.sum(T[:,1]*T[:,1])
        AI = np.linalg.pinv(A)
        U = np.dot(BPM,T)
        x=np.dot(AI,U.T)
        error=np.std(BPM-np.dot(T, x).T,axis=1)
        return x,error

    def latticeScan(self,window=[100,25]):
        self.bpm_fit={}
        bpms2fit = [bpm for bpm in self.mat.keys() if bpm not in self.ignore_list]
        if self.averaged==False:
            print('You need to average!')
        else:
            for bpm in bpms2fit:
                #print(bpm)
                self.bpms2use(bpm)
                if self.plane=='H':
                    x0,error=self.tfit()
                    x,xp,delta=np.dot(self.Hmat[bpm],x0)
                    alpha,beta,emit=fitEllipseMJ(x,xp,window)
                elif self.plane=='V':
                    y0,error=self.tfit_y()
                    y,yp=np.dot(self.Vmat[bpm],y0)
                    alpha,beta,emit=fitEllipseMJ(y,yp,window)
                self.bpm_fit[bpm]={}
                if self.plane=='H':
                    self.bpm_fit[bpm]['x']=np.array(x)
                    self.bpm_fit[bpm]['xp']=np.array(xp)
                    self.bpm_fit[bpm]['delta']=np.array(delta)
                    self.bpm_fit[bpm]['x_norm']=np.array(x/np.sqrt(beta))
                    self.bpm_fit[bpm]['xp_norm']=np.array(alpha/np.sqrt(beta)*x + xp*np.sqrt(beta))
                if self.plane=='V':
                    self.bpm_fit[bpm]['y']=np.array(y)
                    self.bpm_fit[bpm]['yp']=np.array(yp)
                    self.bpm_fit[bpm]['y_norm']=np.array(y/np.sqrt(beta))
                    self.bpm_fit[bpm]['yp_norm']=np.array(alpha/np.sqrt(beta)*y + yp*np.sqrt(beta))
                self.bpm_fit[bpm]['error']=np.array(error)
                self.bpm_fit[bpm]['alpha']=alpha
                self.bpm_fit[bpm]['beta']=beta
                self.bpm_fit[bpm]['emit']=emit
                self.bpm_fit[bpm]['goodfit']=True
                self.bpm_fit[bpm]['goodbpm']=True

    def writebpm4sussix(self, working_dir, onlyhori = False, onlyvert = False, offset_i = 0, twodims = False):
        output_dir=working_dir+'/frame'+str(self.frame)
        try:
            os.mkdir(output_dir)
            #print('Made new directory')
        except:
            pass
        
        if self.bpm_fit:
            i = 1 + offset_i
            num_to_keys = {}
            for key in self.bpm_fit.keys():

                if onlyhori:
                    if (key[2] == 'H') and (self.bpm_fit[key]['goodfit']) and (self.bpm_fit[key]['goodbpm']):
                        if twodims:
                             zerotofill = np.zeros( len( self.bpm_fit[key]['x_norm']) )
                             data = np.vstack([self.bpm_fit[key]['x_norm'], self.bpm_fit[key]['xp_norm'], zerotofill, zerotofill]).T
                        
                        else:    
                            data = np.vstack([self.bpm_fit[key]['x_norm'],self.bpm_fit[key]['xp_norm']]).T
                        
                        name = output_dir+'/bpm.%.4i' %i
                        np.savetxt(name,data)

                        num_to_keys['bpm.%.4i'%i] = key
                        i+=1
                        
                elif onlyvert:
                    if (key[2] == 'V') and (self.bpm_fit[key]['goodfit']) and (self.bpm_fit[key]['goodbpm']):
                        if twodims:
                             zerotofill = np.zeros( len( self.bpm_fit[key]['y_norm']) )
                             data = np.vstack([zerotofill, zerotofill, self.bpm_fit[key]['y_norm'], self.bpm_fit[key]['yp_norm']]).T
                        
                        else:    
                            data = np.vstack([self.bpm_fit[key]['y_norm'],self.bpm_fit[key]['yp_norm']]).T
                        
                        name = output_dir+'/bpm.%.4i' %i
                        print(name)
                        np.savetxt(name,data)

                        num_to_keys['bpm.%.4i'%i] = key
                        i+=1

                else:
                #print(key)
                    if key[2] == 'H':
                        zerotofill = np.zeros( len( self.bpm_fit[key]['x_norm']))
                        data = np.vstack([self.bpm_fit[key]['x_norm'], self.bpm_fit[key]['xp_norm'], zerotofill, zerotofill]).T
                        name = output_dir+'/bpm.%.4i' %i
                        np.savetxt(name,data)

                    elif key[2] == 'V':
                        zerotofill = np.zeros( len( self.bpm_fit[key]['x_norm']))
                        data = np.vstack([zerotofill, zerotofill, self.bpm_fit[key]['x_norm'],self.bpm_fit[key]['xp_norm']]).T
                        name = output_dir+'/bpm.%.4i' %i
                        np.savetxt(name,data)

                    num_to_keys['bpm.%.4i'%i] = key  
                    i+=1
    
            print(len(num_to_keys))
            f = open(output_dir+"/num-to-keys.pkl","wb")
            pickle.dump(num_to_keys,f)
            f.close()
	
    def mask_bad_fits(self,errlimit=1000):
        # Should be called after LatticeScan
        # Errlimit should be set to error limit wanted
        
        for ibpm in self.bpm_fit: 
            sumerror=np.sum(self.bpm_fit[ibpm]['error'])
            if sumerror>errlimit or np.isnan(self.bpm_fit[ibpm]['beta']):
                self.bpm_fit[ibpm]['goodfit'] = False
                print('Bad fit at:')
                print(self.frame,ibpm)

    def write_NAFF(self, working_dir, whichbpms = 'V'):
        # Write NAFF files with initial estimates for vertical or horizontal NAFF
        # Useful to estimate Ix or Iy and initial phases
        
        dfnaff = pd.DataFrame(columns = ['bpm','tune','amplitude','phase','error'])
        
        if self.bpm_fit:
            for key in self.bpm_fit.keys():
                
                coord = list(self.bpm_fit[key].keys())[0]
                
                if key[2] == whichbpms:
                    naffi = NAFFlib.get_tunes(self.bpm_fit[key][coord])
                    
                    dftoapp = pd.DataFrame({
                        'bpm': [key],
                        'tune': [float(naffi[0])],
                        'amplitude': [np.abs(naffi[1][0])],
                        'phase': [np.angle(naffi[1][0],deg='True')],
                        'error': [1.0/len(self.bpm_fit[key][coord])]
                    })
                    
                    dfnaff = pd.concat([dfnaff, dftoapp], ignore_index = True)
        
        output_dir=working_dir+'/frame'+str(self.frame)       
        dfnaff.to_csv(output_dir+'/dfnaff'+whichbpms+'.csv')

def writeHVfiles4sussix(frameh, framev, working_dir):
    # frameh corresponds to frame with horizontal data
    # framev correpsonds to frame with vertical data
    output_dir=working_dir+'/frame'+str(frameh.frame)
    try:
        os.mkdir(output_dir)
        print('Made new directory')
    except:
        pass
    if frameh.bpm_fit and framev.bpm_fit:
            i = 1
            
            keyss = np.unique(np.append([],np.append(np.array(list(frameh.bpm_fit.keys())), np.array(list(framev.bpm_fit.keys())))))
            
            num_to_keys = {}
            
            for key in keyss:
                
                data = np.vstack([frameh.bpm_fit[key]['x_norm'], frameh.bpm_fit[key]['xp_norm'], framev.bpm_fit[key]['y_norm'], framev.bpm_fit[key]['yp_norm']]).T
                
                name = output_dir+'/bpm.%.3i' %i
                np.savetxt(name,data)
                
                num_to_keys['bpm.%.3i'%i] = key  
                i+=1
                                
            f = open(output_dir+"/num-to-keys.pkl","wb")
            pickle.dump(num_to_keys,f)
            f.close()
    

def tbtFitFn(latticeFile, matrixFile, frame=0,avg=[1000,500]):
    data = tbtReader(latticeFile)
    mat, names = txt2array(matrixFile,old=True) 
    Hnames=[(n,i) for i,n in enumerate(names) if 'HP' in n]
    Hnames=Hnames[:10]
    print('Using BPMs:', Hnames)
    BPM = np.zeros((2048,len(Hnames)))
    T = np.zeros((len(Hnames),3))
    for i in range(len(Hnames)):
        BPM[:,i]=data.bpm[Hnames[i][0]][frame]
        print(np.mean(data.bpm[Hnames[i][0]][frame,avg[0]:avg[0]+avg[1]]))
        BPM[:,i]-=np.mean(data.bpm[Hnames[i][0]][frame,avg[0]:avg[0]+avg[1]])
        T[i,:2]=mat[Hnames[i][1]][0,:2]
        T[i,2]=mat[Hnames[i][1]][0,5]
    A = np.zeros((3,3))
    A[0,0] = np.sum(T[:,0]*T[:,0])
    A[0,1] = np.sum(T[:,1]*T[:,0])
    A[0,2] = np.sum(T[:,2]*T[:,0])
    A[1,0] = np.sum(T[:,0]*T[:,1])
    A[1,1] = np.sum(T[:,1]*T[:,1])
    A[1,2] = np.sum(T[:,2]*T[:,1])
    A[2,0] = np.sum(T[:,0]*T[:,2])
    A[2,1] = np.sum(T[:,1]*T[:,2])
    A[2,2] = np.sum(T[:,2]*T[:,2])
    AI = np.linalg.pinv(A)
    U = np.dot(BPM,T)
    x=np.dot(AI,U.T)
    return x 

def fitEllipseMJ(x,xp,window=[1000,500]):
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
    
def reorderBpms(bpmlist, bpmstart):
    i = bpmlist.index(bpmstart)
    newlist = bpmlist[i:] + bpmlist[:i]
    return newlist
