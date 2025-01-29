import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import multiprocessing

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo
import xpart as xp

from scipy import constants

def run(n_turns=72800, qx = 25.365, qy = 24.44, nemx = 15, nemy = 15, sigma_z = 0.6, rf_voltage = 80e3, harmonic = 588, ppb = 0.5e10, n_part = int(1e4)):

    context = xo.ContextCpu() 

    mad = Madx(stdout = False)

    mad.input('''
            !beam,    sequence=RING605_FODO, particle="proton",energy= 8.88459;
            E0 :=8.88459;
            beam, particle=proton,energy= E0;

            call, file="markers.def";
            call, file="multipoles_madx.def";
            !call, file="madx_types.def";
            call, file="madx_types-apertures.def";
            call, file="values.dat";
            call, file="madx.seq";

            use, sequence=RING605_FODO;
            
            SC220A,K2 = -0.475948;
            SC220B,K2 = -0.475948;
              
            SC222A,K2 = 0.472948;
            SC222B,K2 = 0.472948;
              
            SC319A,K2 = 0.480748;
            SC319B,K2 = 0.480748;
              
            SC321A,K2 = -0.483748;
            SC321B,K2 = -0.483748;

            twiss,chrom,sequence=RING605_FODO,file="twiss_cpymad.tfs";

            MATCH, SEQUENCE=RING605_FODO;
            GLOBAL, Q1=%f;
            GLOBAL, Q2=%f;
            VARY, NAME= k1_qt601a, STEP=0.00001;
            VARY, NAME= k1_qt601b, STEP=0.00001;
            VARY, NAME= k1_qt601c, STEP=0.00001;
            VARY, NAME= k1_qt601d, STEP=0.00001;
            VARY, NAME= k1_qt602a, STEP=0.00001;
            VARY, NAME= k1_qt602b, STEP=0.00001;
            VARY, NAME= k1_qt602c, STEP=0.00001;
            VARY, NAME= k1_qt602d, STEP=0.00001;
            VARY, NAME= k1_qt603a, STEP=0.00001;
            VARY, NAME= k1_qt603b, STEP=0.00001;
            VARY, NAME= k1_qt603c, STEP=0.00001;
            VARY, NAME= k1_qt603d, STEP=0.00001;
            VARY, NAME= k1_qt604a, STEP=0.00001;
            VARY, NAME= k1_qt604b, STEP=0.00001;
            VARY, NAME= k1_qt604c, STEP=0.00001;
            VARY, NAME= k1_qt604d, STEP=0.00001;
            VARY, NAME= k1_qt605a, STEP=0.00001;
            VARY, NAME= k1_qt605b, STEP=0.00001;
            VARY, NAME= k1_qt605c, STEP=0.00001;
            VARY, NAME= k1_qt605d, STEP=0.00001;
            VARY, NAME= k1_qt606a, STEP=0.00001;
            VARY, NAME= k1_qt606b, STEP=0.00001;
            VARY, NAME= k1_qt606c, STEP=0.00001;
            VARY, NAME= k1_qt606d, STEP=0.00001;
            VARY, NAME= k1_qt607a, STEP=0.00001;
            VARY, NAME= k1_qt607b, STEP=0.00001;
            VARY, NAME= k1_qt607c, STEP=0.00001;
            VARY, NAME= k1_qt607d, STEP=0.00001;
            VARY, NAME= k1_qt608a, STEP=0.00001;
            VARY, NAME= k1_qt608b, STEP=0.00001;
            VARY, NAME= k1_qt608c, STEP=0.00001;
            VARY, NAME= k1_qt608d, STEP=0.00001;
            VARY, NAME= k1_qt609a, STEP=0.00001;
            VARY, NAME= k1_qt609b, STEP=0.00001;
            VARY, NAME= k1_qt609c, STEP=0.00001;
            VARY, NAME= k1_qt609d, STEP=0.00001;
            JACOBIAN, CALLS=200, TOLERANCE=1e-5;
            ENDMATCH;

            twiss,chrom,sequence=RING605_FODO,file="twiss_tuned_cpymad.tfs";

            '''%(qx,qy))
    
    line = xt.Line.from_madx_sequence(sequence=mad.sequence.ring605_fodo, 
                                      allow_thick=True, 
                                      enable_align_errors=True,
                                      deferred_expressions=True,
                                      install_apertures=True)

    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,gamma0=mad.sequence.ring605_fodo.beam.gamma)

    tw = line.twiss(method='4d')

    c = constants.value('speed of light in vacuum')

    nemitt_x = nemx*1e-6/6
    nemitt_y = nemy*1e-6/6

    # Turn on RF cavities
    rf_freq = harmonic * tw.beta0 * c/line.get_length()

    t_rev = tw.T_rev0
    f_rev = 1/t_rev

    line['rfcav53mhz'].voltage = rf_voltage
    line['rfcav53mhz'].frequency = rf_freq

    line.discard_tracker() 

    # Insert beam size monitor at IPM locations
    ipmh = xt.BeamSizeMonitor(start_at_turn=0, stop_at_turn=n_turns,frev=f_rev,sampling_frequency=f_rev)
    line.insert_element(at='ipmh', element = ipmh, name = 'my_ipm')

    # Create particles and track them
    line.build_tracker(_context=context)

    particles = xp.generate_matched_gaussian_bunch(_context=context,
                                                   num_particles=n_part, total_intensity_particles=ppb,
                                                   nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
                                                   particle_ref=line.particle_ref, line=line)
    
    line.track(particles, num_turns=n_turns, turn_by_turn_monitor=False, with_progress = True)

    # Calculate survival ratio
    ntotal = len(particles.x)
    particles.hide_lost_particles()
    nsurv = len(particles.x)
    particles.unhide_lost_particles()

    surv = nsurv/ntotal

    np.save('72800turns/qxtuned_%.3f.npy'%qx,mad.table.summ['q1'][0])
    np.save('72800turns/qytuned_%.3f.npy'%qx,mad.table.summ['q2'][0])
    np.save('72800turns/surv_%.3f.npy'%qx,surv)
    np.save('72800turns/count_%.3f.npy'%qx,ipmh.count)
    np.save('72800turns/x_std_%.3f.npy'%qx,ipmh.x_std)
    np.save('72800turns/y_std_%.3f.npy'%qx,ipmh.y_std)

    return

def task(argslice):

    qxgrid = np.arange(25.300,25.380,0.002)

    qxtask = np.split(qxgrid.flatten(),8)[argslice]

    for qxtaski in qxtask:
        run(qx = qxtaski, n_turns = 72800, nemx = 12, nemy = 12)

if __name__ == '__main__':
    # create all tasks
    processes = [multiprocessing.Process(target=task, args=(i,)) for i in range(8)]
    # start all processes
    for process in processes:
        process.start()
    # wait for all processes to complete
    for process in processes:
        process.join()
    # report that all tasks are completed
    print('Done', flush=True)










