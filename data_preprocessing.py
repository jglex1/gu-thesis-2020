"""
Author: Alex GU (44287207)
Date: Wed Jul 22 2020

# --- Modifications / Revisions --- #
+ [Jul 22 2020] Created to replace the deprecated data preprocessing module
  "prepro.py" which was used for semester 1 work.
+ [Jul 29 2020] "Data" class methods were generalised to work for the entire
  mendeleev dataset.

# --- Description --- #
Module to handle all data preprocessing steps. Includes features such as: (1)
retrieve the mendeleev dataset and save it as a csv file, (2) 'Data' class for
preprocessing the mendeleev data.
"""

# Import statements

import pandas as pd
import numpy as np
import mendeleev
from datetime import date
from sklearn.preprocessing import MinMaxScaler


#------------------------EXPORT MENDELEEV PROPERTIES--------------------------#

def mendeleev_to_csv(save_dir='..\\datasets_original\\'):
    """Retrieves the mendeleev "elements" dataset, removes unimportant features
    and saves as csv file "atom_characteristics_{currentDate}".
    
    Data from https://mendeleev.readthedocs.io/en/stable/data.html
    
    Parameters:
        save_dir (str) : directory prefix string
    """
    
    # Get mendeleev data table "elements"
    table = mendeleev.get_table('elements')
    
    # Drop some unimportant properties
    table = table.drop(['annotation','jmol_color','cpk_color',
                        'discoverers','discovery_year','discovery_location',
                        'name_origin','sources','uses'],axis=1)
    
    # Save to csv with current date appended to file name
    table.to_csv(save_dir + 'atom_characteristics_{}_{}.csv'.format(date.today().month,date.today().day), index=False)


def make_atom_char_dict(location='../datasets_original/atom_characteristics_7_30.csv'):
    """Creates a dictionary of dictionaries of atom characteristics based on the
    supplied csv of data. Structure: {'chemSymbol':{'feature':...},...}

    Parameters:
        location (str) : locaton of the dataset
    
    Returns:
        atom_char_dict (dict) : atom characteristic dictionary of dictionaries
        with the chemical symbol as the key and features as the subsequent keys.
    """
    
    atom_char = pd.read_csv(location).set_index('symbol')
    atom_char_dict = atom_char.to_dict('index')
    
    return atom_char_dict


#--------------------------- DATA HANDLING CLASS -----------------------------#

class Data():
    
    def __init__(self, directory):
        
        self._df = pd.read_csv(directory)
        self._date_created = (date.today().month,date.today().day)
        
    def clean(self):
        
        df = self._df
        
        df = df.replace('-',np.NaN)
        df = df.dropna()
        df = df.apply(pd.to_numeric, errors='ignore')
        
        self._df = df
        
    def include_mendeleev_characteristic(self, chars):
        
        # e.g. chars = ['atomic_weight']
        
        df = self._df
        
        for char in chars:
            for feature in ['A','B']:
                convert = df[feature].values
                converted = [elements[x][char] for x in convert]
                char_title_form = char.replace("_", " ")
                df["{} {}".format(feature, char_title_form)] = converted
        
        self._df = df
    
    def drop(self, todrop):
        """
        todrop -> a list of strings (column names)
        """
        df = self._df
        df = df.drop(todrop ,axis=1)
        self._df = df
        
    def bulk_normalise_features(self, lst):
        
        df = self._df
    
        scalers = []
        for group in lst:
            scaler = MinMaxScaler().fit(df[group].values.reshape(-1, 1))
            scalers.append(scaler)
            df[group] = scaler.transform(df[group])
        
        self._norm_scalers = scalers
        self._df = df
    
    def saveas(self, name, save_dir='..\\datasets_generated\\'):

        save_string = save_dir + '{}_{}_{}.csv'.format(name, date.today().month, date.today().day)
        self._df.to_csv(save_string, index=False)
       

#----------------------------------- MAIN ------------------------------------#

# Make Mendeleev dictionary
elements = make_atom_char_dict()


# Semester 1
if 0:
    df = Data("../datasets_original/database_perovskite.csv")
    df.drop(['Chemical formula', 'Crystal Type', 'Crystal Type.1',
             'Magnetic moment [mu_B]','Stability [eV/atom]',
             'Volume per atom [A^3/atom]','Vacancy energy [eV/O atom]'])
    df.clean()
    df.include_mendeleev_characteristic(['atomic_weight'])
    df.drop(['A','B']) # always after mendeleev
    df.bulk_normalise_features([['Valence A', 'Valence B'],
                                ['Radius A [ang]', 'Radius B [ang]'],
                                ['a [ang]','b [ang]', 'c [ang]'],
                                ['alpha [deg]', 'beta [deg]', 'gamma [deg]'],
                                ['Band gap [eV]'],
                                ['Formation energy [eV/atom]'],
                                ['A atomic weight','B atomic weight']])
    
    df.saveas('semester1_norm')
    
# Semester 2 (no-norm)
if 0:
    df = Data("../datasets_original/database_perovskite.csv")
    df.drop(['Chemical formula', 'Crystal Type', 'Crystal Type.1',
             'Magnetic moment [mu_B]','Stability [eV/atom]',
             'Volume per atom [A^3/atom]','Vacancy energy [eV/O atom]'])
    df.clean()
    df.include_mendeleev_characteristic(['atomic_number', 'atomic_radius',
        'atomic_volume', 'boiling_point', 'density', 'dipole_polarizability',
        'electron_affinity', 'evaporation_heat', 'fusion_heat', 'lattice_constant',
        'melting_point', 'specific_heat', 'thermal_conductivity', 'vdw_radius',
        'covalent_radius_cordero', 'covalent_radius_pyykko', 'en_pauling',
        'heat_of_formation', 'covalent_radius_slater', 'vdw_radius_uff',
        'vdw_radius_mm3', 'abundance_crust', 'abundance_sea', 'en_ghosh',
        'vdw_radius_alvarez', 'c6_gb', 'atomic_weight', 'atomic_radius_rahm',
        'goldschmidt_class',  'covalent_radius_pyykko_double', 'mendeleev_number',
        'pettifor_number', 'glawe_number'])
    # Immediate drop (temp)
    df.drop(['A goldschmidt class','B goldschmidt class',
             'A abundance crust', 'A abundance sea',
             'B abundance crust', 'B abundance sea'])
    df.drop(['A','B']) # always after mendeleev
    df.clean()
    df.saveas('db-sem2-v2')
    
if 0:
    df = Data("../datasets_original/database_perovskite.csv")
    df.drop(['Chemical formula', 'Crystal Type', 'Crystal Type.1',
             'Magnetic moment [mu_B]','Stability [eV/atom]',
             'Volume per atom [A^3/atom]','Vacancy energy [eV/O atom]'])
    df.include_mendeleev_characteristic(['atomic_number', 'atomic_radius',
        'atomic_volume', 'boiling_point', 'density', 'dipole_polarizability',
        'electron_affinity', 'evaporation_heat', 'fusion_heat', 'lattice_constant',
        'melting_point', 'specific_heat', 'thermal_conductivity', 'vdw_radius',
        'covalent_radius_cordero', 'covalent_radius_pyykko', 'en_pauling',
        'heat_of_formation', 'covalent_radius_slater', 'vdw_radius_uff',
        'vdw_radius_mm3', 'abundance_crust', 'abundance_sea', 'en_ghosh',
        'vdw_radius_alvarez', 'c6_gb', 'atomic_weight', 'atomic_radius_rahm',
        'goldschmidt_class',  'covalent_radius_pyykko_double', 'mendeleev_number',
        'pettifor_number', 'glawe_number'])
    # Immediate drop (temp)
    df.saveas('raw')
    
if 0:
    ##### MAIN MAIN MAIN MAIN MAIN ######
    df = Data("../datasets_original/database_perovskite.csv")
    df.drop(['Chemical formula', 'Crystal Type', 'Crystal Type.1',
             'Magnetic moment [mu_B]','Stability [eV/atom]',
             'Volume per atom [A^3/atom]','Vacancy energy [eV/O atom]'])
    df.clean()
    df.include_mendeleev_characteristic(['atomic_number',
                                         'atomic_radius',
                                         'atomic_volume',
                                         'boiling_point',
                                         'density',
                                         'dipole_polarizability',
                                         #'electron_affinity',
                                         #'evaporation_heat',
                                         #'fusion_heat',
                                         'lattice_constant',
                                         'melting_point',
                                         'specific_heat',
                                         #'thermal_conductivity',
                                         'vdw_radius',
                                         'covalent_radius_cordero',
                                         'covalent_radius_pyykko',
                                         'en_pauling',
                                         'heat_of_formation',
                                         'covalent_radius_slater',
                                         'vdw_radius_uff',
                                         'vdw_radius_mm3', 
                                         'abundance_crust',
                                         #'abundance_sea',
                                         'en_ghosh',
                                         'vdw_radius_alvarez',
                                         'c6_gb',
                                         'atomic_weight',
                                         'atomic_radius_rahm',
                                         #'goldschmidt_class',
                                         'covalent_radius_pyykko_double',
                                         'mendeleev_number',
                                         'pettifor_number',
                                         'glawe_number'])
    # Immediate drop (temp)
    df.drop(['A','B']) # always after mendeleev
    df.clean()
    
    df.bulk_normalise_features([['Valence A', 'Valence B'],
            ['Radius A [ang]', 'Radius B [ang]'],
            ['a [ang]','b [ang]', 'c [ang]'],
            ['alpha [deg]', 'beta [deg]', 'gamma [deg]'],
            ['Band gap [eV]'],
            ['Formation energy [eV/atom]'],
            ['A atomic number', 'B atomic number'],
            ['A atomic radius', 'B atomic radius'],
            ['A atomic volume', 'B atomic volume'],
            ['A boiling point', 'B boiling point'],
            ['A density', 'B density'],
            ['A dipole polarizability', 'B dipole polarizability'],
            #['A electron affinity', 'B electron affinity'],
            #['A evaporation heat', 'B evaporation heat'],
            #['A fusion heat', 'B fusion heat'],
            ['A lattice constant', 'B lattice constant'],
            ['A melting point', 'B melting point'],
            ['A specific heat', 'B specific heat'],
            #['A thermal conductivity', 'B thermal conductivity'],
            ['A vdw radius', 'B vdw radius'],
            ['A covalent radius cordero', 'B covalent radius cordero'],
            ['A covalent radius pyykko', 'B covalent radius pyykko'],
            ['A en pauling', 'B en pauling'],
            ['A heat of formation', 'B heat of formation'],
            ['A covalent radius slater', 'B covalent radius slater'],
            ['A vdw radius uff', 'B vdw radius uff'],
            ['A vdw radius mm3', 'B vdw radius mm3'],
            ['A abundance crust', 'B abundance crust'],
            #['A abundance sea', 'B abundance sea'],
            ['A en ghosh', 'B en ghosh'],
            ['A vdw radius alvarez', 'B vdw radius alvarez'],
            ['A c6 gb', 'B c6 gb'],
            ['A atomic weight','B atomic weight'],
            ['A atomic radius rahm', 'B atomic radius rahm'],
            ['A covalent radius pyykko double', 'B covalent radius pyykko double'],
            ['A mendeleev number', 'B mendeleev number'],
            ['A pettifor number', 'B pettifor number'],
            ['A glawe number', 'B glawe number']
            ])
    
    df.saveas('db-sem2-norm-v2')    
    

if 1:
    ## MAIN ALTERED FOR CULLED FEATURES
    df = Data("../datasets_original/database_perovskite.csv")
    df.drop(['Chemical formula', 'Crystal Type', 'Crystal Type.1',
             'Magnetic moment [mu_B]','Stability [eV/atom]',
             'Volume per atom [A^3/atom]','Vacancy energy [eV/O atom]'])
    df.clean()
    df.include_mendeleev_characteristic(['atomic_number',
                                         'atomic_radius',
                                         'atomic_volume',
                                         'boiling_point',
                                         'density',
                                         'dipole_polarizability',
                                         #'electron_affinity',
                                         #'evaporation_heat',
                                         #'fusion_heat',
                                         'lattice_constant',
                                         'melting_point',
                                         'specific_heat',
                                         #'thermal_conductivity',
                                         'vdw_radius',
                                         'covalent_radius_cordero',
                                         'covalent_radius_pyykko',
                                         'en_pauling',
                                         'heat_of_formation',
                                         'covalent_radius_slater',
                                         'vdw_radius_uff',
                                         'vdw_radius_mm3', 
                                         'abundance_crust',
                                         #'abundance_sea',
                                         'en_ghosh',
                                         'vdw_radius_alvarez',
                                         'c6_gb',
                                         'atomic_weight',
                                         'atomic_radius_rahm',
                                         #'goldschmidt_class',
                                         'covalent_radius_pyykko_double',
                                         'mendeleev_number',
                                         'pettifor_number',
                                         'glawe_number'])
    # Immediate drop (temp)
    df.drop(['A','B']) # always after mendeleev
    
    df.clean()
    
    df.bulk_normalise_features([['Valence A', 'Valence B'],
            ['Radius A [ang]', 'Radius B [ang]'],
            ['a [ang]','b [ang]', 'c [ang]'],
            ['alpha [deg]', 'beta [deg]', 'gamma [deg]'],
            ['Band gap [eV]'],
            ['Formation energy [eV/atom]'],
            ['A atomic number', 'B atomic number'],
            ['A atomic radius', 'B atomic radius'],
            ['A atomic volume', 'B atomic volume'],
            ['A boiling point', 'B boiling point'],
            ['A density', 'B density'],
            ['A dipole polarizability', 'B dipole polarizability'],
            #['A electron affinity', 'B electron affinity'],
            #['A evaporation heat', 'B evaporation heat'],
            #['A fusion heat', 'B fusion heat'],
            ['A lattice constant', 'B lattice constant'],
            ['A melting point', 'B melting point'],
            ['A specific heat', 'B specific heat'],
            #['A thermal conductivity', 'B thermal conductivity'],
            ['A vdw radius', 'B vdw radius'],
            ['A covalent radius cordero', 'B covalent radius cordero'],
            ['A covalent radius pyykko', 'B covalent radius pyykko'],
            ['A en pauling', 'B en pauling'],
            ['A heat of formation', 'B heat of formation'],
            ['A covalent radius slater', 'B covalent radius slater'],
            ['A vdw radius uff', 'B vdw radius uff'],
            ['A vdw radius mm3', 'B vdw radius mm3'],
            ['A abundance crust', 'B abundance crust'],
            #['A abundance sea', 'B abundance sea'],
            ['A en ghosh', 'B en ghosh'],
            ['A vdw radius alvarez', 'B vdw radius alvarez'],
            ['A c6 gb', 'B c6 gb'],
            ['A atomic weight','B atomic weight'],
            ['A atomic radius rahm', 'B atomic radius rahm'],
            ['A covalent radius pyykko double', 'B covalent radius pyykko double'],
            ['A mendeleev number', 'B mendeleev number'],
            ['A pettifor number', 'B pettifor number'],
            ['A glawe number', 'B glawe number']
            ])
    
    ### CULLED FEATURES
    df.drop(['A heat of formation',
             'A specific heat','B specific heat', #both
             'A vdw radius', 'B vdw radius', # both
             'B dipole polarizability',
             'A covalent radius cordero','B covalent radius cordero', #both
             'A atomic number', 'B atomic number', #both
             'A atomic weight', 'B atomic weight', #both
             'B atomic volume',
             'A vdw radius uff',
             'B c6 gb', # interesting, A is 3rd most important feature
             'A vdw radius mm3', 'B vdw radius mm3', #both
             'B atomic radius',
             'A boiling point', 'B boiling point', #both
             'B vdw radius alvarez',
             'A covalent radius slater', 'B covalent radius slater', #both
             'A lattice constant',
             'B en ghosh',
             'A covalent radius pyykko double', 'B covalent radius pyykko double', #both
             'B covalent radius pyykko',
             'Valence A', 'Valence B', #both
             'alpha [deg]', 'beta [deg]', 'gamma [deg]']) #all
    
    df.saveas('db-sem2-culled-norm-v2')  
    
#-----------------------------------------------------------------------------#
#                              EXAMPLE USAGE                                  #   
#-----------------------------------------------------------------------------#

"""To save the processed dataset df."""
# df.saveas('NAME')