import numpy as np
import synergia

def convert_rbends_to_sbends(orig_lattice):
    lattice = synergia.lattice.Lattice("rrnova")
    for elem in orig_lattice.get_elements():
        if elem.get_type_name() == "rbend":
            new_elem = synergia.lattice.Lattice_element("sbend", elem.get_name())
            s_attributes = elem.get_string_attributes()
            d_attributes = elem.get_double_attributes()
            for s in s_attributes.keys():
                new_elem.set_string_attribute(s, s_attributes[s])
            for d in d_attributes.keys():
                new_elem.set_double_attribute(d, d_attributes[d])
            ang = elem.get_double_attribute("angle")
            length = elem.get_double_attribute("l")
            arclength = ang*length/(2.0*np.sin(ang/2.0))
            new_elem.set_double_attribute("l", arclength)
            new_elem.set_double_attribute("e1", ang/2.0)
            new_elem.set_double_attribute("e2", ang/2.0)
            lattice.append(new_elem)
        else:
            lattice.append(elem)
    lattice.set_reference_particle(orig_lattice.get_reference_particle())
    return lattice
    
    