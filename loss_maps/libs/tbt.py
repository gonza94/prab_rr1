from ctypes import *
import time
import numpy as np

class device_type(Structure):
    _pack_ = 2
    _fields_ = [('house',c_short),('channel',c_short),('name',c_byte*10),('tbt_name',c_byte*10),('assoc',c_short),('di',c_int),('flag',c_short)]

class bpm_trigger_type(Structure):
    _pack_ = 2
    _fields_ = [('armEvent',c_short),('triggerEvent',c_short),('bsDA_delay',c_float),('bpm_state',c_short)]

class header_type(Structure):
     _pack_ = 2
     _fields_ = [ ('version',c_short), ('machine',c_short), ('comment',c_byte*50), ('date',c_byte*18), ('header_size',c_int),('bpm_ts',c_byte*18), ('turn_start',c_short), ('turns_total',c_short), ('intensity',device_type),('num_house',c_short),('num_bpm',c_short*2),('live',c_short),('scaled_data',c_short),('trigger',bpm_trigger_type),('tbt_frames',c_short),('dummy',c_byte*384) ]

class acnet_device(Structure):
    _pack_ = 2
    _fields_ = [('name',c_byte*10),('di',c_int)]

class rr_house_type(Structure):
    _pack_ = 2
    _fields_ = [('name',c_byte*4),('cmd',acnet_device),('arm_delay',acnet_device),('code',c_short),('flag',c_short),('dummy',acnet_device*5),('bpm_idx',c_short*80)]

class rr_house_data_type(Structure):
    _pack_ = 1
    _fields_ = [('name',c_byte*5),('delay',c_float),('code',c_short),('offset',c_short),('dummy',c_short*9),('dev',device_type*80),('map',c_short*80)]

class tbt_arch_type(Structure):
    #_pack_ = 3
    _fields_ = [('ncase',c_short),('frame',c_short),('data',c_byte*4),('comment',c_byte*50),('trigger',bpm_trigger_type),('bpm_ts',c_byte*18),('dummy',c_short*5)]

class house_data_type(Structure):
   #old datatype likely
   _pack_ = 1
   _fields_ = [('name',c_byte*5),('cycle',c_int),('delay',c_float),('last_turn',c_short),('circular',c_short),('state',c_short),('t_stamp',acnet_device*2),('dev',device_type*12),('map',c_short*12),('offset',c_short),('code',c_short),('dummy',c_short*8)]

class data(Structure):
    _pack_ = 1
    _fields_ = [('data',c_float*2048*209)]

class tbtReader(object):
    '''tbtReader reads a .tbt file and creates a dictionary bpm which for each beam has an n * m array where n
       is the number of frames and m is the number of turns'''
    def __init__(self, file):
        f = open(file,'rb')
        self.header = header_type()
        f.readinto(self.header)
        if self.header.tbt_frames==0:
            self.header.tbt_frames=1
        self.rr_house_type = (rr_house_data_type*self.header.num_house)()
        f.readinto(self.rr_house_type)
        self.arch_type=(tbt_arch_type*self.header.tbt_frames)()
        #foo=f.read(self.header.tbt_frames*96) ### need to modify such that arch_type gives 96 bytes
        f.readinto(self.arch_type)
        self.data=(data*self.header.tbt_frames)()
        f.readinto(self.data)
        f.close()
        self.makePositionArray()
        self.makeBpmMap()
        self.Hmap()
        self.Vmap()

    def makePositionArray(self):
        self.pos=np.zeros((self.header.tbt_frames,(1+sum(self.header.num_bpm)),self.header.turns_total))
        for i in range(self.header.tbt_frames):
            for j in range((1+sum(self.header.num_bpm))):
                self.pos[i,j,:] = np.array(self.data[i].data[:][j][:])

    def makeBpmMap(self):
        self.map = {}
        for i in range(self.header.num_house):
            for j in range(80):
                name = "".join([chr(k) for k in self.rr_house_type[i].dev[j].name[:] if k!=0 if chr(k)!=' '])
                if len(name)!=0:
                    ind = self.rr_house_type[i].dev[j].assoc*self.header.num_bpm[0] + 1 + self.rr_house_type[i].map[j]
                    self.map[name]=ind
        self.map["".join([chr(k) for k in self.header.intensity.name[:] if k!=0 if chr(k)!=' '])]=0
        self.bpm = {}
        for key in self.map.keys():
            self.bpm[key] = self.pos[:,self.map[key],:]

    def Hmap(self):
        self.Hmap=[key for key in self.map.keys() if 'HP' in key]
        self.Hmap.sort(key=lambda s: int(s[4:]))
        Hind=np.array([self.map[i] for i in self.Hmap])
        self.Hpos = self.pos[:,Hind,:]

    def Vmap(self):
        self.Vmap=[key for key in self.map.keys() if 'VP' in key]
        self.Vmap.sort(key=lambda s: int(s[4:]))
        Vind=np.array([self.map[i] for i in self.Vmap])
        self.Vpos = self.pos[:,Vind,:]

def timeStampConv(date):
    months = {'JAN': '01', 'FEB':'02', 'MAR':'03','APR':'04' ,'MAY':'05','JUN':'06','JUL':'07','AUG':'08','SEP':'09','OCT':'10','NOV':'11','DEC':'12'}
    d=date.split()[0][:2]+'.'+months[date.split()[0][2:5]]+'.20'+date.split()[0][5:]+' ' + date.split()[1][:8]
    return time.mktime(time.strptime(d, "%d.%m.%Y %H:%M:%S"))
