import synergia
from synergia.lattice import Lattice_element
import syn_lib
import numpy as np
import sys, os

def getSextupoles(name, orig_length, sex_length = 0.1524):
    """
    Function that takes any drift element length and name and creates two sextupoles(sa1/sb1)-drift(d)-two sextupoles(sa2/sb2)
    of that same length.

    Parameters
    ----------
    name: str (String with name to insert into every element's name)
    orig_length: float (Float representing the length of the original drift object)
    sex_length: float (Float representing the sextupole length of the new sextupoles to introduce)
    :return: sa,sb,o,sa2,sb2
    sa: synergia.lattice.Lattice_element (New sextupole to insert)
    sb: synergia.lattice.Lattice_element (New sextupole to insert)
    o: synergia.lattice.Lattice_element (New drift between sextupoles to insert)
    sa2: synergia.lattice.Lattice_element (New sextupole to insert)
    sb2: synergia.lattice.Lattice_element (New sextupole to insert) 
    """

    sa = Lattice_element("sextupole", name+'a1')
    sa.set_double_attribute("l", sex_length)
    sa.set_double_attribute("k2", 0)
    
    sb = Lattice_element("sextupole", name+'b1')
    sb.set_double_attribute("l", sex_length)
    sb.set_double_attribute("k2", 0)
    
    o = Lattice_element("drift", name+"d")
    o.set_double_attribute("l", orig_length-4*sex_length)
    
    sa2 = Lattice_element("sextupole", name+'a2')
    sa2.set_double_attribute("l", sex_length)
    sa2.set_double_attribute("k2", 0)
    
    sb2 = Lattice_element("sextupole", name+'b2')
    sb2.set_double_attribute("l", sex_length)
    sb2.set_double_attribute("k2", 0)

    #print(orig_length,orig_length-2*sex_length)
    #print(name)
    return sa,sb,o,sa2,sb2

def intro_620_sexts(lattice):
    """
    Function that takes the RR lattice and introduces the two new 620 sextupoles

    Parameters
    ----------
    lattice : synergia.lattice Object
    :return: new_lattice
    new_lattice2: synergia.lattice.Lattice (Lattice object with the two new 620 sextupoles)
    """
    # Calculate lattice functions and 
    synergia.simulation.Lattice_simulator.CourantSnyderLatticeFunctions(lattice)
    synergia.simulation.Lattice_simulator.set_closed_orbit_tolerance(1e-12)
    synergia.simulation.Lattice_simulator.calculate_closed_orbit(lattice)
    synergia.simulation.Lattice_simulator.calc_dispersions(lattice)

    # Create first new lattice
    new_lattice = synergia.lattice.Lattice("h3000_h1020_comp")
    new_lattice.set_reference_particle(lattice.get_reference_particle())

    # Insert sextupoles drift spaces in 620 sextupoles
    i = 25
    scsexts = []

    for elm in lattice.get_elements():
        if (elm.get_length() > 1.0) & (elm.get_type_name()=='drift') & (elm.lf.dispersion.hor < 0.1):

            if elm.get_name()=='dqq':
                if elm.get_ancestors()[-1]=='mid620':
                    csa,csb,csd,csa2,csb2 = getSextupoles('cs'+str(i),orig_length = elm.get_length(), sex_length = 0.1524)
                    new_lattice.append(csa)
                    scsexts.append(csa.get_name())

                    new_lattice.append(csb)
                    scsexts.append(csb.get_name())

                    new_lattice.append(csd)

                    new_lattice.append(csa2)
                    scsexts.append(csa2.get_name())

                    new_lattice.append(csb2)
                    scsexts.append(csb2.get_name())

                    i+=1
                else:
                    new_lattice.append(elm)
            else:
                new_lattice.append(elm)
        else:
            new_lattice.append(elm)

    print('Original lattice length: %.10f'%lattice.get_length())
    print('First new lattice length: %.10f'%new_lattice.get_length())

    # Keep only 620 sextupoles that we want
    cskeep = np.array(['cs25b1','cs26b2'],dtype=object)

    # Get new lattice only with wanted 620 candidates
    new_lattice2 = synergia.lattice.Lattice("third_comp")
    new_lattice2.set_reference_particle(lattice.get_reference_particle())

    for elem in new_lattice.get_elements():
        if elem.get_name()[:2]!='cs':
            new_lattice2.append(elem)
        elif elem.get_name() in cskeep:
            if elem.get_name()=='cs25b1':
                elem620 = Lattice_element("sextupole", 'sc620a')
                elem620.copy_attributes_from(elem)
            elif elem.get_name()=='cs26b2':
                elem620 = Lattice_element("sextupole", 'sc620b')
                elem620.copy_attributes_from(elem)
            print(elem)
            print(elem620)
            new_lattice2.append(elem620)
        elif len(elem.get_name())<4:
            new_lattice2.append(elem)
        else:
            o = Lattice_element("drift", "csd")
            o.set_double_attribute("l", elem.get_length())
            #print('Removing: ',elem.get_name())
            new_lattice2.append(o)

    print('Original lattice length: %.10f'%lattice.get_length())
    print('New lattice with 620 sextupoles length: %.10f'%new_lattice2.get_length())
    
    return new_lattice2


def comp_h3000(lattice):
    """
    Function to cancel out the h3000 term in the SYNERGIA lattice using the 4 sextupoles: sc220,sc222,sc319,sc321

    Parameters
    ----------
    lattice : synergia.lattice Object
    :return: lattice object with h3000 cancelled out
    """
    h3000s,s1,elemsh3000 = syn_lib.get_hjklm(lattice,j=3,k=0,l=0,m=0)
    h3000bare = np.sum(h3000s)
    print('Bare Machine h3000 RDT:')
    print(h3000bare)

    msc220a = syn_lib.get_mjklm_elm(lattice,'sc220a',j=3,k=0,l=0,m=0)
    msc220b = syn_lib.get_mjklm_elm(lattice,'sc220b',j=3,k=0,l=0,m=0)

    msc222a = syn_lib.get_mjklm_elm(lattice,'sc222a',j=3,k=0,l=0,m=0)
    msc222b = syn_lib.get_mjklm_elm(lattice,'sc222b',j=3,k=0,l=0,m=0)

    msc319a = syn_lib.get_mjklm_elm(lattice,'sc319a',j=3,k=0,l=0,m=0)
    msc319b = syn_lib.get_mjklm_elm(lattice,'sc319b',j=3,k=0,l=0,m=0)

    msc321a = syn_lib.get_mjklm_elm(lattice,'sc321a',j=3,k=0,l=0,m=0)
    msc321b = syn_lib.get_mjklm_elm(lattice,'sc321b',j=3,k=0,l=0,m=0)

    hbarevec = [-np.real(h3000bare),-np.imag(h3000bare), 0, 0, 0, 0, 0, 0]

    mmatrix = np.array([[float(msc220a[0]),float(msc220b[0]),float(msc222a[0]),float(msc222b[0]),float(msc319a[0]),float(msc319b[0]),float(msc321a[0]),float(msc321b[0])],
                        [float(msc220a[1]),float(msc220b[1]),float(msc222a[1]),float(msc222b[1]),float(msc319a[1]),float(msc319b[1]),float(msc321a[1]),float(msc321b[1])],
                        [1.0, -1.0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1.0, -1.0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1.0, -1.0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1.0, -1.0],
                        [1, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 1, 0],])

    kcompi = np.linalg.inv(mmatrix).dot(hbarevec)

    # Setting sextupoles to compensate h3000
    print('Setting sextupoles to compensate h3000')
    for elm in lattice.get_elements():
        if elm.get_name() == 'sc220a':
            elm.set_double_attribute('k2',kcompi[0])
            print(elm.get_name())
            print('k2=%.8f'%elm.get_double_attribute('k2'))
            
        elif elm.get_name() == 'sc220b':
            elm.set_double_attribute('k2',kcompi[1])
            print(elm.get_name())
            print('k2=%.8f'%elm.get_double_attribute('k2'))
        
        elif elm.get_name() == 'sc222a':
            elm.set_double_attribute('k2',kcompi[2])
            print(elm.get_name())
            print('k2=%.8f'%elm.get_double_attribute('k2'))
        
        elif elm.get_name() == 'sc222b':
            elm.set_double_attribute('k2',kcompi[3])
            print(elm.get_name())
            print('k2=%.8f'%elm.get_double_attribute('k2'))
        
        elif elm.get_name() == 'sc319a':
            elm.set_double_attribute('k2',kcompi[4])
            print(elm.get_name())
            print('k2=%.8f'%elm.get_double_attribute('k2'))
        
        elif elm.get_name() == 'sc319b':
            elm.set_double_attribute('k2',kcompi[5])
            print(elm.get_name())
            print('k2=%.8f'%elm.get_double_attribute('k2'))
        
        elif elm.get_name() == 'sc321a':
            elm.set_double_attribute('k2',kcompi[6])
            print(elm.get_name())
            print('k2=%.8f'%elm.get_double_attribute('k2'))
        
        elif elm.get_name() == 'sc321b':
            elm.set_double_attribute('k2',kcompi[7])
            print(elm.get_name())
            print('k2=%.8f'%elm.get_double_attribute('k2'))

    h3000s,s1,elemsh3000 = syn_lib.get_hjklm(lattice,j=3,k=0,l=0,m=0)
    h3000comp = np.sum(h3000s)

    print('Compensated h3000 RDT:')
    print(h3000comp)

    return lattice

def comp_h3000_with620(lattice):
     """
    Function to cancel out the h3000 term in the SYNERGIA lattice using the 4 sextupoles: sc220,sc222,sc319,sc321

    Parameters
    ----------
    lattice : synergia.lattice Object
    :return: lattice object with h3000 cancelled out including 620 sextupoles
    """
    h3000s,s1,elemsh3000 = syn_lib.get_hjklm(lattice,j=3,k=0,l=0,m=0)
    h3000bare = np.sum(h3000s)
    print('Bare Machine h3000 RDT:')
    print(h3000bare)

    h1020s,s2,elemsh1020 = syn_lib.get_hjklm(lattice,j=1,k=0,l=2,m=0)
    h1020bare = np.sum(h1020s)
    print('Bare Machine h1020 RDT:')
    print(h1020bare)

    # Calculate m_3000 values for each sextupole
    mh3000sc220a = syn_lib.get_mjklm_elm(new_lattice2,'sc220a',j=3,k=0,l=0,m=0)
    mh3000sc220b = syn_lib.get_mjklm_elm(new_lattice2,'sc220b',j=3,k=0,l=0,m=0)

    mh3000sc222a = syn_lib.get_mjklm_elm(new_lattice2,'sc222a',j=3,k=0,l=0,m=0)
    mh3000sc222b = syn_lib.get_mjklm_elm(new_lattice2,'sc222b',j=3,k=0,l=0,m=0)

    mh3000sc319a = syn_lib.get_mjklm_elm(new_lattice2,'sc319a',j=3,k=0,l=0,m=0)
    mh3000sc319b = syn_lib.get_mjklm_elm(new_lattice2,'sc319b',j=3,k=0,l=0,m=0)

    mh3000sc321a = syn_lib.get_mjklm_elm(new_lattice2,'sc321a',j=3,k=0,l=0,m=0)
    mh3000sc321b = syn_lib.get_mjklm_elm(new_lattice2,'sc321b',j=3,k=0,l=0,m=0)

    mh3000sc620a = syn_lib.get_mjklm_elm(new_lattice2,'sc620a',j=3,k=0,l=0,m=0)
    mh3000sc620b = syn_lib.get_mjklm_elm(new_lattice2,'sc620b',j=3,k=0,l=0,m=0)

    # Calculate m_1020 values for each sextupole
    mh1020sc220a = syn_lib.get_mjklm_elm(new_lattice2,'sc220a',j=1,k=0,l=2,m=0)
    mh1020sc220b = syn_lib.get_mjklm_elm(new_lattice2,'sc220b',j=1,k=0,l=2,m=0)

    mh1020sc222a = syn_lib.get_mjklm_elm(new_lattice2,'sc222a',j=1,k=0,l=2,m=0)
    mh1020sc222b = syn_lib.get_mjklm_elm(new_lattice2,'sc222b',j=1,k=0,l=2,m=0)

    mh1020sc319a = syn_lib.get_mjklm_elm(new_lattice2,'sc319a',j=1,k=0,l=2,m=0)
    mh1020sc319b = syn_lib.get_mjklm_elm(new_lattice2,'sc319b',j=1,k=0,l=2,m=0)

    mh1020sc321a = syn_lib.get_mjklm_elm(new_lattice2,'sc321a',j=1,k=0,l=2,m=0)
    mh1020sc321b = syn_lib.get_mjklm_elm(new_lattice2,'sc321b',j=1,k=0,l=2,m=0)

    mh1020sc620a = syn_lib.get_mjklm_elm(new_lattice2,'sc620a',j=1,k=0,l=2,m=0)
    mh1020sc620b = syn_lib.get_mjklm_elm(new_lattice2,'sc620b',j=1,k=0,l=2,m=0) 

    hbarevec = [-np.real(h3000bare),-np.imag(h3000bare), -np.real(h1020bare),-np.imag(h1020bare), 0, 0, 0, 0]

    mmatrix = np.array([[float(mh3000sc220a[0]),float(mh3000sc220b[0]),float(mh3000sc222a[0]),float(mh3000sc222b[0]),float(mh3000sc319a[0]),float(mh3000sc319b[0]),float(mh3000sc321a[0]),float(mh3000sc321b[0]),float(mh3000sc620a[0]),float(mh3000sc620b[0])],
                        [float(mh3000sc220a[1]),float(mh3000sc220b[1]),float(mh3000sc222a[1]),float(mh3000sc222b[1]),float(mh3000sc319a[1]),float(mh3000sc319b[1]),float(mh3000sc321a[1]),float(mh3000sc321b[1]),float(mh3000sc620a[1]),float(mh3000sc620b[1])],
                        [float(mh1020sc220a[0]),float(mh1020sc220b[0]),float(mh1020sc222a[0]),float(mh1020sc222b[0]),float(mh1020sc319a[0]),float(mh1020sc319b[0]),float(mh1020sc321a[0]),float(mh1020sc321b[0]),float(mh1020sc620a[0]),float(mh1020sc620b[0])],
                        [float(mh1020sc220a[1]),float(mh1020sc220b[1]),float(mh1020sc222a[1]),float(mh1020sc222b[1]),float(mh1020sc319a[1]),float(mh1020sc319b[1]),float(mh1020sc321a[1]),float(mh1020sc321b[1]),float(mh1020sc620a[1]),float(mh1020sc620b[1])],
                        [1.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1.0, -1.0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1.0, -1.0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1.0, -1.0, 0, 0]])

    kcompi = np.linalg.pinv(mmatrix).dot(hbarevec)

    for elm in new_lattice2.get_elements():
        if elm.get_name() == 'sc220a':
            elm.set_double_attribute('k2',kcompi[0])
            print(elm)
            
        elif elm.get_name() == 'sc220b':
            elm.set_double_attribute('k2',kcompi[1])
            print(elm)
        
        elif elm.get_name() == 'sc222a':
            elm.set_double_attribute('k2',kcompi[2])
            print(elm)
        
        elif elm.get_name() == 'sc222b':
            elm.set_double_attribute('k2',kcompi[3])
            print(elm)
        
        elif elm.get_name() == 'sc319a':
            elm.set_double_attribute('k2',kcompi[4])
            print(elm)
        
        elif elm.get_name() == 'sc319b':
            elm.set_double_attribute('k2',kcompi[5])
            print(elm)
        
        elif elm.get_name() == 'sc321a':
            elm.set_double_attribute('k2',kcompi[6])
            print(elm)
        
        elif elm.get_name() == 'sc321b':
            elm.set_double_attribute('k2',kcompi[7])
            print(elm)
            
        elif elm.get_name() == 'sc620a':
            elm.set_double_attribute('k2',kcompi[8])
            print(elm)
            
        elif elm.get_name() == 'sc620b':
            elm.set_double_attribute('k2',kcompi[9])
            print(elm)

    comph3000s,comps1,compelemsh3000 = syn_lib.get_hjklm(new_lattice2,j=3,k=0,l=0,m=0)
    comph3000bare = np.sum(comph3000s)
    print('Compensated h3000 RDT:')
    print(comph3000bare)

    comph1020s,comps2,compelemsh1020 = syn_lib.get_hjklm(new_lattice2,j=1,k=0,l=2,m=0)
    comph1020bare = np.sum(comph1020s)
    print('Compensated h1020 RDT:')
    print(comph1020bare)

