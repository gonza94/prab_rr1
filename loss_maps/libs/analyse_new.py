import tunespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


#hscan = tunespace.scan_data('tunescan_202220_hscan_nosq.dat','butter'p)
#hscan = tunespace.scan_data('tunescan_122320_hscan_10turns_sqon.dat','butter')
#hscan = tunespace.scan_data('tuneScan_290921_allOff.dat','butter')
hscan = tunespace.scan_data('tunescan_2turns_alloff_092921_scanh.dat','butter')
#vscan = tunespace.scan_data('tunescan_122220_vscan_nosq.dat','butter')
#vscan = tunespace.scan_data('tunescan_122320_vscan_10turns_sqon.dat','butter')
vscan = tunespace.scan_data('tunescan_2turns_alloff_092921_scanv.dat','butter')

hscan.build_contour_data()
vscan.build_contour_data()

xnew = np.arange(25.3, 25.46, 3e-4)
ynew = np.arange(24.3, 24.46, 3e-4)
fullinterp = hscan.contour_interp(xnew,ynew) + vscan.contour_interp(xnew,ynew)
#plt.contourf(-fullinterp,extent=(25.3,25.46,24.3,24.46),vmin=0.002,vmax=0.05)
#plt.contourf(-fullinterp,extent=(25.3,25.46,24.3,24.46))
#plt.figure(figsize=(4,4))
for i in range(fullinterp.shape[0]):
    fullinterp[i,fullinterp[i,:]>0]=-1e-5


levs=np.array([2.5e-6,5e-6,7.5e-6,1e-5,2.5e-5,5e-5,7.5e-5,1e-4,2.5e-4,5e-4,7.5e-4,1e-3,2.5e-3,5e-3,7.5e-3,1e-2,2.5e-2,5e-2,1e-1,2.5e-1,5e-1,7.5e-1])
plt.contourf(-fullinterp,extent=(25.3,25.46,24.3,24.46),locator=ticker.LogLocator(numticks=10),levels=levs)


#plt.contourf(-fullinterp,extent=(25.3,25.46,24.3,24.46),locator=ticker.LogLocator(numticks=15))
plt.plot([25+1/3.,25+1/3.],[24.3,24.46],'k')
plt.plot([25.3,25.46],[24+1/3.,24+1/3.],'k--')
#plt.plot([25.3,25.46],[24.3,24.46],'k')
plt.plot([25.3,25.35],[24.4,24.3],'k--')
plt.plot([25.3,25.4],[24.35,24.3],'k')
#plt.plot([25.385,25.425],[24.46,24.3],'k')
plt.plot([25.4,25.4],[24.46,24.3],'k')
plt.plot([25.3,25.46],[24.425,24.385],'k--')
plt.plot([25.0,25.5],[24.+1/3.,24.5],'k--')
plt.plot([25.3,25.46],[24.3,24.46],'k--')
#plt.plot([25.4,25.4],[24.3,24.46],'k')
#plt.plot([25.3,25.46],[24.4,24.4],'k')
plt.text(25.325,24.45,'3Qx=76',rotation=90,fontsize=15)
plt.text(25.4,24.325,'3Qy=75',fontsize=15)
plt.text(25.40,24.37,'4Qx + Qy =126',rotation=-65,fontsize=15)
plt.text(25.37,24.4,'5Qx  =127',rotation=0,fontsize=15)
#plt.text(25.37,24.4,'Qx + 4Qy =123',rotation=-8,fontsize=15)
plt.text(25.36,24.455,'-Qx + 3Qy =48',rotation=12,fontsize=15)
#plt.text(25.4,24.43,'Qx-Qy=1',rotation=38,fontsize=15)
plt.text(25.35,24.327,'Qx+2Qy=74',rotation=-18,fontsize=15)
plt.text(25.31,24.37,'2Qx+Qy=75',rotation=-48,fontsize=15)
plt.xlim((25.3,25.46))
plt.ylim((24.3,24.46))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('$Q_x$',fontsize=40)
plt.ylabel('$Q_y$',fontsize=40)
ax=plt.gca()
ax.set_aspect('equal', 'box')
plt.show()
