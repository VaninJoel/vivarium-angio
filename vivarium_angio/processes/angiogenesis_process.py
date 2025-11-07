from vivarium.core.process import Process
import os
import warnings
import sys
from io import StringIO
from time import time
from multiprocessing import Process as MultiProcess, Queue
import zarr
import pandas as pd
import scipy.ndimage as ndimage
import skimage.measure as measure
from skimage.morphology import medial_axis

# Suppress std output while importing the CC3D modules
original_stdout = sys.stdout
sys.stdout = StringIO()
warnings.filterwarnings("ignore", message="Setting not set: PlayerSizes")
warnings.filterwarnings("ignore", message="Setting not set: PlayerSizesFloating")
warnings.filterwarnings("ignore", message="Setting not set: RecentSimulations")

from cc3d.core.simservice.CC3DSimService import CC3DSimService
from cc3d.core.PyCoreSpecs import (
    PottsCore, 
    Metadata, 
    CellTypePlugin,
    VolumePlugin, 
    CenterOfMassPlugin,
    NeighborTrackerPlugin,
    PixelTrackerPlugin, 
    ContactPlugin,
    ConnectivityGlobalPlugin,
    ChemotaxisPlugin,
    ChemotaxisParameters,
    ChemotaxisTypeParameters,
    DiffusionSolverFE,
    UniformInitializer
)
from cc3d.core.PySteppables import SteppableBasePy
import numpy as np

# Restore the original stdout
sys.stdout = original_stdout

class StoreManager():
    """
    Manages Zarr store for simulation data.

    New structure (v2): Store path points directly to the Zarr hierarchy.
    Timesteps are stored at root level: {store_path}/10/, {store_path}/20/, etc.
    No nested 'experiments' group needed since experiment name is in directory path.
    """

    def __init__(self, store:str):
        self.store_path = store
        # Open zarr store at root level (no experiments group)
        root = zarr.open(self.store_path, mode='a')
        self.expname = None
        self.exparams = None

    def create_store(self):
        """Create a new Zarr store (not needed in new structure)."""
        root = zarr.open(self.store_path, mode='w')

    def get_experiments(self):
        """Get list of timesteps (not experiments) at root level."""
        root = zarr.open(self.store_path, mode='r')
        return list(root.keys())

    def setup_exp(self, expname:str, exparams:dict):
        """
        Store experiment parameters in root attributes.

        In new structure, store_path already includes experiment name,
        so we store parameters at root level of this Zarr store.
        """
        self.expname = expname
        self.exparams = exparams

        # Store parameters in root attributes
        root = zarr.open(self.store_path, mode='a')
        root.attrs['exp_name'] = expname
        root.attrs['params'] = exparams

    def get_exp_params(self, expname:str=None):
        """Get experiment parameters from root attributes."""
        root = zarr.open(self.store_path, mode='r')
        if 'params' in root.attrs:
            return root.attrs['params']
        else:
            print(f'No parameters found in Zarr store')
            return None

    def write_data(self, data:np.ndarray, step:int):
        """
        Write data at root level: {store_path}/{step}/data

        Old structure: store_path/experiments/exp_name/step/data
        New structure: store_path/step/data (exp_name is in store_path)
        """
        if not self.expname:
            print('No experiment set???')
            return

        root = zarr.open(self.store_path, mode='a')

        # Write directly at root level
        if str(step) in root:
            print(f'Step {step} already exists, skipping')
            return

        mcs_group = root.require_group(str(step))
        # Zarr v3 API: use array assignment instead of create_dataset with compression
        mcs_group['data'] = data

    def get_exp_length(self, expname:str=None):
        """Get number of timesteps in the Zarr store."""
        root = zarr.open(self.store_path, mode='r')
        # Count numeric keys (timesteps) at root level
        timesteps = [k for k in root.keys() if k.isdigit()]
        length = len(timesteps)
        return length

    def grab_cellview(self, expname:str, step:int):
        """Get cell data for a specific timestep."""
        root = zarr.open(self.store_path, mode='r')
        mcs_group = root[str(step)]['data']
        cell_data = mcs_group[:,:,:,0]
        cell_data = np.squeeze(cell_data)
        return cell_data

    def grab_fieldview(self, expname:str, step:int):
        """Get field data for a specific timestep."""
        root = zarr.open(self.store_path, mode='r')
        mcs_group = root[str(step)]['data']
        field_data = mcs_group[:,:,:,2]
        field_data = np.squeeze(field_data)
        return field_data

    def analyse_one_step(self, step:int, queue:bool=False, expname:str=None):
        """Analyze cell data for a specific timestep."""
        expname = self.expname if expname is None else expname
        root = zarr.open(self.store_path, mode='r')
        mcs_group = root[str(step)]['data']
        cell_data = self.grab_cellview(expname, step)
        cell_data = np.squeeze(cell_data)

        cell_data = np.where(cell_data != 0, 1, 0)
        i_cell_data = np.where(cell_data == 0, 1, 0)
        
        labeled_image, num_labels = ndimage.label(i_cell_data)
        regions = measure.regionprops(labeled_image)
        regions = sorted(regions, key=lambda x: x.area, reverse=True)
        area_list = []
        for i, region in enumerate(regions, start=1):
            if region.bbox[0] == 0 or region.bbox[1] == 0 or region.bbox[2] == 399 or region.bbox[3] == 399:
                continue
            if region.area > 10 and region.area < 10000:
                area_list.append(region.area)
        area_arr = np.array(area_list)

        skel, distance =  medial_axis(cell_data, return_distance=True)
        dist_on_skel = distance * skel
        sort_px_list = np.sort(dist_on_skel.ravel())
        nonzero_pixels = sort_px_list[sort_px_list > 0]
        nonzero_pixels = nonzero_pixels[::-1]
        pix_array = np.array(nonzero_pixels)
        n_regions = area_arr.shape[0]
        mean_l_area = np.mean(area_arr) if n_regions > 0 else 0
        std_l_area = np.std(area_arr) if n_regions > 0 else 0

        results_dict = {"expname": expname,
                        "step": step,
                        "n_regions": n_regions,
                        "mean_l_area": mean_l_area, 
                        "std_l_area": std_l_area,
                        "len_network": pix_array.shape[0], 
                        "mean_n_width": np.mean(pix_array),
                        "std_n_width": np.std(pix_array)}
        
        if not queue:
            return results_dict
        else:
            queue.put(results_dict)
        
    def analyse_step_range(self, expname:str ,start:int, end:int):
        results = []
        processes = []
        queue = Queue()
        for step in range(start, end):
            p = MultiProcess(target=self.analyse_one_step, args=(step, queue, expname))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        while not queue.empty():
            results.append(queue.get())
        
        return pd.DataFrame(results)
    
    def analyse_exp_list(self, exp_list:list):
        results = []
        processes = []
        queue = Queue()
        
        for expname in exp_list:
            step  = self.get_exp_length(expname)
            p = MultiProcess(target=self.analyse_one_step, args=(step, queue, expname))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        while not queue.empty():
            results.append(queue.get())

        return pd.DataFrame(results)

class WriterSteppable(SteppableBasePy):
    """
    Steppable to write the simulation data to a zarr store on command
    """
    def __init__(self, store:str, exp_name:str, exp_params:dict, field_names:list, frequency:int=1) -> None:
        SteppableBasePy.__init__(self,frequency)
        self.exp_params = exp_params
        self.store = store
        self.exp_name = exp_name
        self.field_names = field_names
        self.store_manager = StoreManager(store)
        self.store_manager.setup_exp(exp_name, exp_params)

        # print(f'Experiment {self.store_manager.expname} set up with params {self.store_manager.exparams}')

    def write_to_store(self, timestep:int):
        """The function writes the simulation data to the zarr store
        
        Parameters
        ----------
        timestep : int
            The current timestep, migh differ from the MCS in concurrent simulations
        """
        # We create a list of the shelves names to store the data
        shelves = ['type', 'id'] + self.field_names

        # We create a placeholder numpy array to store the values of the simulation
        shelve_vals = np.zeros((self.dim.x, self.dim.y, self.dim.z,len(shelves)))

        # We loop through all the lattice voxels and store the values of the cells and the fields
        search_start = time()
        flds = [getattr(self.field, field) for field in self.field_names]     
        for x, y, z in self.every_pixel():
            cell = self.cell_field[x,y,z]
            if cell:
                shelve_vals[x,y,z,0] = cell.type
                shelve_vals[x,y,z,1] = cell.id

            for i, field in enumerate(flds):
                shelve_vals[x,y,z,i+2] = field[x,y,z]

        search_end = time()
        # We write the data to the zarr store via the store manager
        self.store_manager.write_data(shelve_vals, timestep)
        write_end = time()
        print(f"searching time: {search_end - search_start} sec") 
        print(f"writing time: {write_end - search_end}) sec")
    
    def step(self, mcs):
        # We get the external input containing the task and the timestep
        # We can pass any information to the steppable via the external input as long as it is serializable
        # vegf = self.field.VEGF
        # print(vegf[5,5,5])
        task, timestep = self.external_input
        if task == 'write':
            self.write_to_store(timestep=timestep)

        # We can set the external output to pass information back to the main process like logging information
        self.external_output = f'Timestep {timestep} done using task {task}'
        
    def finish(self):
        # We can do some final processing here if needed
        pass

class AngiogenesisProcess(Process):
    """
    Vivarium Process for angiogenesis simulation with comprehensive metadata.

    This process wraps a CompuCell3D angiogenesis model and provides rich parameter
    metadata for GUI generation, validation, and scientific documentation.
    """

    defaults = {
        'jee': 2.0,
        'jem': 2.0,
        'lchem': 500.0,
        'lsc': 0.1,
        'vedir': 1.0,
        'veder': 0.3,
        'vesec': 0.3,
        'exp_name': 'vivarium_run',
    }

    # Define reference papers as class constants for reusability
    REFERENCES = {
        'merks2006': {
            'source': 'Merks et al., 2006',
            'title': 'Cell elongation is key to in silico replication of in vitro vasculogenesis and subsequent remodeling',
            'doi': '10.1016/j.ydbio.2005.10.003',
            'url': 'https://doi.org/10.1016/j.ydbio.2005.10.003',
            'journal': 'Developmental Biology',
            'year': 2006,
        },
        'merks2008': {
            'source': 'Merks & Glazier, 2008',
            'title': 'A cell-centered approach to developmental biology',
            'doi': '10.1016/j.physa.2007.11.053',
            'url': 'https://doi.org/10.1016/j.physa.2007.11.053',
            'journal': 'Physica A',
            'year': 2008,
        },
        'model_specific': {
            'source': 'This Model',
            'title': 'Parameters determined for this specific model implementation',
            'doi': None,
            'url': None,
        }
    }

    # Define parameter presets from literature
    PRESETS = {
        'default': {
            'name': 'Default Configuration',
            'description': 'Standard parameters for typical angiogenesis simulation',
            'reference': 'merks2006',
            'parameters': {
                'jee': 2.0,
                'jem': 2.0,
                'lchem': 500.0,
                'lsc': 0.1,
                'vedir': 1.0,
                'veder': 0.3,
                'vesec': 0.3,
                'sim_time': 100.0,
                'write_frequency': 10.0,
            }
        },
        'high_aggregation': {
            'name': 'High Cell Aggregation',
            'description': 'Parameters promoting strong cell clustering and compact network formation',
            'reference': 'merks2006',
            'parameters': {
                'jee': 2.0,
                'jem': 8.0,
                'lchem': 200.0,
                'lsc': 0.1,
                'vedir': 0.5,
                'veder': 0.3,
                'vesec': 0.3,
                'sim_time': 100.0,
                'write_frequency': 10.0,
            }
        },
        'high_branching': {
            'name': 'High Network Branching',
            'description': 'Parameters favoring extensive network formation with many branches',
            'reference': 'merks2006',
            'parameters': {
                'jee': 2.0,
                'jem': 2.0,
                'lchem': 800.0,
                'lsc': 0.1,
                'vedir': 1.5,
                'veder': 0.2,
                'vesec': 0.4,
                'sim_time': 100.0,
                'write_frequency': 10.0,
            }
        },
        'fast_simulation': {
            'name': 'Fast Simulation',
            'description': 'Quick test run with moderate parameters (for testing)',
            'reference': 'model_specific',
            'parameters': {
                'jee': 4.0,
                'jem': 4.0,
                'lchem': 500.0,
                'lsc': 0.1,
                'vedir': 1.0,
                'veder': 0.3,
                'vesec': 0.3,
                'sim_time': 50.0,
                'write_frequency': 10.0,
            }
        }
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.specs = []
        
        simCore = PottsCore(dim_x=200, dim_y=200, dim_z=1, steps=10000, fluctuation_amplitude=5.0, neighbor_order=1, boundary_x="Periodic", boundary_y="Periodic")
        self.specs.append(simCore)

        simMeta = Metadata(num_processors=1)
        self.specs.append(simMeta)

        simCelltypes = CellTypePlugin("Medium", "EC")
        self.specs.append(simCelltypes)

        simVolume = VolumePlugin()
        simVolume.param_new("EC", 50.0, 5.0)
        self.specs.append(simVolume)

        simCOM = CenterOfMassPlugin()
        self.specs.append(simCOM)

        simNtrack = NeighborTrackerPlugin()
        self.specs.append(simNtrack)

        simPXtrack = PixelTrackerPlugin()
        self.specs.append(simPXtrack)

        simContact = ContactPlugin(neighbor_order=4)
        simContact.param_new("Medium", "EC", self.parameters['jem'])
        simContact.param_new("EC", "EC", self.parameters['jee'])
        self.specs.append(simContact)

        simConnectivity = ConnectivityGlobalPlugin(True, "EC")
        self.specs.append(simConnectivity)

        simDiffusion = DiffusionSolverFE()
        vegf = simDiffusion.field_new("VEGF")
        bds = [('x', 'min'), ('x', 'max'), ('y', 'min'), ('y', 'max')]
        for bd in bds:
            setattr(vegf.bcs, f"{bd[0]}_{bd[1]}_type", 'Periodic')
        vegf.diff_data.diff_global = self.parameters['vedir']
        vegf.diff_data.decay_global = self.parameters['veder']
        vegf.diff_data.decay_types["EC"] = 0.0
        vegf.secretion_data_new("EC", self.parameters['vesec'])
        self.specs.append(simDiffusion)

        f = ChemotaxisParameters(field_name="VEGF", solver_name="DiffusionSolverFE")
        kwargs = {"lambda_chemo": self.parameters['lchem'], "towards": "Medium", "linear_sat_cf": self.parameters['lsc']}
        p = ChemotaxisTypeParameters("EC", **kwargs)
        f.params_append(p)
        simChemotx = ChemotaxisPlugin(f)
        self.specs.append(simChemotx)

        simInit = UniformInitializer()
        simInit.region_new(pt_min=[1, 1, 0], pt_max=[199, 199, 1], width=7, gap=3, cell_types=['EC'])
        self.specs.append(simInit)

        # Disable CC3D default output directory (we use Zarr instead)
        # Setting output_dir=None prevents CC3D from creating OutStore folder
        self.core_sim = CC3DSimService(output_dir=None, output_frequency=0)
        self.core_sim.register_specs(self.specs)

        # Use custom store path if provided, otherwise default
        self.store = self.parameters.get('store_path', 'temp_store.zarr')
        field_names = ['VEGF']
        self.core_sim.register_steppable(WriterSteppable(store=self.store,
                                                           exp_name=self.parameters['exp_name'],
                                                           exp_params=self.parameters,
                                                           field_names=field_names), frequency=1)
        
        self.core_sim.run()
        self.core_sim.init()
        self.core_sim.start()
        
        self.mcs = 0

    def ports_schema(self):
        """
        Enhanced schema with comprehensive metadata for GUI generation and validation.

        This schema is designed to be consumed by:
        1. GUI generation tools (main_window.py)
        2. Validation systems (utils/validation.py)
        3. Documentation generators
        4. Parameter export/import tools

        Returns:
            dict: Complete parameter schema with scientific metadata
        """

        return {
            'inputs': {
                # =====================================================================
                # CELL ADHESION PARAMETERS
                # =====================================================================

                'jee': {
                    # === CORE VIVARIUM FIELDS ===
                    '_default': self.defaults['jee'],
                    '_updater': 'set',
                    '_emit': True,

                    # === DISPLAY & DESCRIPTION ===
                    '_display_name': 'EC-EC Adhesion Energy',
                    '_description': 'Contact energy between adjacent Endothelial Cells (EC). Lower values promote stronger adhesion and cell-cell cohesion. This is the J parameter in the Cellular Potts Model Hamiltonian.',
                    '_long_description': 'In the Cellular Potts Model, the Hamiltonian includes a term Σ J(τ(σ),τ(σ\')) (1-δ(σ,σ\')) where J represents contact energy between cells of types τ. Lower J values favor contact (stronger adhesion), while higher values discourage contact. For EC-EC interactions, typical values range from 2 (strong adhesion) to 16 (weak adhesion).',

                    # === CATEGORIZATION ===
                    '_category': 'Cell Properties',
                    '_subcategory': 'Adhesion',
                    '_expert_level': 'intermediate',  # basic | intermediate | advanced

                    # === UNITS & CONVERSION ===
                    '_unit': 'Arbitrary Energy Units',
                    '_unit_long': 'Energy units (dimensionless in CPM)',
                    '_conversion_factor': None,  # No conversion needed

                    # === VALIDATION RANGES ===
                    '_physiological_range': (2.0, 16.0),
                    '_recommended_range': (2.0, 8.0),
                    '_mathematical_range': (0.0, 100.0),  # Model stable range
                    '_warning_threshold': 10.0,  # Warn if exceeds this

                    # === SCIENTIFIC DOCUMENTATION ===
                    '_reference_paper': {
                        **self.REFERENCES['merks2006'],
                        'value': 2.0,
                        'notes': 'Default value from Table 1',
                        'equation': 'H_adhesion term in Eq. 1',
                        'page': 47,
                    },
                    '_cbo_term': 'CBO:0000015',  # Cell-cell adhesion
                    '_model_context': 'Contact energy term in Cellular Potts Model Hamiltonian. Appears in energy calculation: ΔH = Σ J(τ,τ\')(1-δ) for cell boundary modifications.',

                    # === RELATIONSHIPS ===
                    '_related_parameters': ['jem', 'lchem'],
                    '_affects_output': [
                        'cell_cluster_size',
                        'network_connectivity',
                        'mean_cluster_separation'
                    ],
                    '_dependencies': {
                        'jem': 'Balance between EC-EC and EC-Medium adhesion determines aggregation vs spreading',
                    },

                    # === PRESETS ===
                    '_presets': {
                        'strong_adhesion': 2.0,
                        'moderate_adhesion': 4.0,
                        'weak_adhesion': 8.0,
                    },

                    # === GUI HINTS ===
                    '_gui_hints': {
                        'widget_type': 'slider',  # or 'spinbox'
                        'slider_step': 0.5,
                        'show_preset_buttons': True,
                        'tooltip_show_equation': True,
                    },

                    # === BIOLOGICAL MEANING ===
                    '_biological_meaning': 'Represents the strength of cadherin-mediated cell-cell adhesion. Lower values simulate tissues with strong intercellular junctions, while higher values simulate loosely connected tissues.',

                    # === PARAMETER HISTORY (for metadata tracking) ===
                    '_parameter_type': 'biophysical',  # biophysical | kinetic | geometric | control
                    '_measurable': False,  # Can this be directly measured experimentally?
                    '_fitting_method': 'literature_based',  # literature_based | fitted | estimated
                },

                'jem': {
                    '_default': self.defaults['jem'],
                    '_updater': 'set',
                    '_emit': True,

                    '_display_name': 'EC-Medium Adhesion Energy',
                    '_description': 'Contact energy between Endothelial Cells and the surrounding medium. Higher values cause cells to minimize medium contact, promoting aggregation into compact clusters.',
                    '_long_description': 'The EC-Medium adhesion parameter controls how much cells prefer to be in contact with other cells versus being exposed to medium. High values (>6) cause the initially connected network to fragment into separate aggregates. This is crucial for controlling whether the simulation produces a connected network or isolated cell clusters.',

                    '_category': 'Cell Properties',
                    '_subcategory': 'Adhesion',
                    '_expert_level': 'basic',  # More accessible - directly visible effect

                    '_unit': 'Arbitrary Energy Units',
                    '_unit_long': 'Energy units (dimensionless in CPM)',
                    '_conversion_factor': None,

                    '_physiological_range': (2.0, 16.0),
                    '_recommended_range': (2.0, 10.0),
                    '_mathematical_range': (0.0, 100.0),
                    '_warning_threshold': 12.0,

                    '_reference_paper': {
                        **self.REFERENCES['merks2006'],
                        'value': 2.0,
                        'notes': 'Default value. Higher values (8-16) tested for aggregation studies',
                        'equation': 'H_adhesion term in Eq. 1',
                        'page': 47,
                    },
                    '_cbo_term': 'CBO:0000016',  # Cell-matrix adhesion
                    '_model_context': 'EC-Medium contact energy in CPM Hamiltonian. Critical parameter controlling aggregation vs network formation.',

                    '_related_parameters': ['jee'],
                    '_affects_output': [
                        'network_fragmentation',
                        'cell_cluster_count',
                        'cluster_compactness'
                    ],
                    '_dependencies': {
                        'jee': 'Ratio jem/jee determines aggregation behavior: >1 promotes clustering, <1 promotes spreading',
                    },

                    '_presets': {
                        'network_forming': 2.0,
                        'moderate_aggregation': 4.0,
                        'strong_aggregation': 8.0,
                        'extreme_aggregation': 12.0,
                    },

                    '_gui_hints': {
                        'widget_type': 'slider',
                        'slider_step': 0.5,
                        'show_preset_buttons': True,
                        'highlight_major_changes': True,  # Visual warning when changing significantly
                        'live_preview': True,  # If possible, show effect
                    },

                    '_biological_meaning': 'Represents cell-matrix adhesion strength. In vivo, this corresponds to integrin-mediated adhesion to extracellular matrix. High values simulate conditions where cells prefer cell-cell contact over matrix contact, leading to aggregation.',

                    '_visual_effects': {
                        'low_value': 'Cells form connected networks',
                        'high_value': 'Cells aggregate into separate compact clusters',
                        'threshold': 6.0,  # Approximate transition point
                    },

                    '_parameter_type': 'biophysical',
                    '_measurable': False,
                    '_fitting_method': 'literature_based',
                },

                # =====================================================================
                # CHEMOTAXIS PARAMETERS
                # =====================================================================

                'lchem': {
                    '_default': self.defaults['lchem'],
                    '_updater': 'set',
                    '_emit': True,

                    '_display_name': 'Chemotaxis Strength (λ_chem)',
                    '_description': 'Coefficient determining the strength of directed cell migration up VEGF gradients. Higher values mean stronger chemotactic response.',
                    '_long_description': 'In the CPM, chemotaxis is implemented as an additional Hamiltonian term: H_chem = -λ_chem * Σ C(x) for pixels x at the extending membrane, where C(x) is the VEGF concentration. Negative sign means cells prefer moving toward higher concentrations. Typical range: 0 (no chemotaxis) to 1000 (very strong response).',

                    '_category': 'Cell Properties',
                    '_subcategory': 'Chemotaxis',
                    '_expert_level': 'basic',

                    '_unit': 'Arbitrary Energy Units',
                    '_unit_long': 'Energy per concentration unit',
                    '_conversion_factor': None,

                    '_physiological_range': (0, 1000),
                    '_recommended_range': (100, 800),
                    '_mathematical_range': (0, 5000),
                    '_warning_threshold': 1200,

                    '_reference_paper': {
                        **self.REFERENCES['merks2006'],
                        'value': 500.0,
                        'notes': 'Moderate chemotaxis strength',
                        'equation': 'λ in chemotaxis term (Eq. 3)',
                        'page': 48,
                    },
                    '_cbo_term': 'CBO:0000072',  # Chemotaxis
                    '_model_context': 'Chemotaxis Hamiltonian term: H_chem = -λ_chem * Σ_membrane VEGF(x). Drives cell migration toward VEGF sources.',

                    '_related_parameters': ['vedir', 'vesec', 'lsc'],
                    '_affects_output': [
                        'branch_extension_rate',
                        'network_directionality',
                        'sprouting_frequency'
                    ],
                    '_dependencies': {
                        'vedir': 'Fast diffusion flattens gradients, reducing effective chemotaxis even with high λ_chem',
                        'vesec': 'High secretion creates stronger gradients, amplifying chemotaxis effect',
                    },

                    '_presets': {
                        'no_chemotaxis': 0.0,
                        'weak': 200.0,
                        'moderate': 500.0,
                        'strong': 800.0,
                        'very_strong': 1000.0,
                    },

                    '_gui_hints': {
                        'widget_type': 'slider',
                        'slider_step': 50.0,
                        'show_preset_buttons': True,
                        'logarithmic_scale': False,
                    },

                    '_biological_meaning': 'Represents the sensitivity of endothelial cells to VEGF gradients. In vivo, this reflects VEGF receptor density and downstream signaling pathway strength.',

                    '_visual_effects': {
                        'low_value': 'Random cell migration, isotropic network',
                        'high_value': 'Directed sprouting toward VEGF sources, anisotropic network',
                    },

                    '_parameter_type': 'biophysical',
                    '_measurable': True,  # Can be estimated from cell migration assays
                    '_fitting_method': 'literature_based',
                    '_experimental_assay': 'Transwell chemotaxis assay, under-agarose assay',
                },

                'lsc': {
                    '_default': self.defaults['lsc'],
                    '_updater': 'set',
                    '_emit': True,

                    '_display_name': 'Chemotaxis Saturation Coefficient',
                    '_description': 'Saturation parameter for chemotaxis response. Controls whether chemotaxis is linear (0) or saturating (>0) with concentration.',
                    '_long_description': 'Implements saturation in chemotaxis: H_chem = -λ_chem * Σ C(x)/(1 + lsc*C(x)). When lsc=0, response is linear with concentration. When lsc>0, response saturates at high concentrations, representing receptor saturation. Binary choice in original model: 0 or 0.1.',

                    '_category': 'Cell Properties',
                    '_subcategory': 'Chemotaxis',
                    '_expert_level': 'advanced',

                    '_unit': 'dimensionless',
                    '_unit_long': '1/concentration (saturation coefficient)',
                    '_conversion_factor': None,

                    '_physiological_range': (0.0, 0.1),
                    '_recommended_range': (0.0, 0.1),
                    '_mathematical_range': (0.0, 1.0),

                    '_reference_paper': {
                        **self.REFERENCES['merks2006'],
                        'value': 0.1,
                        'notes': 'Binary choice: 0 (linear) or 0.1 (saturating)',
                        'equation': 'Saturation term in Eq. 3',
                        'page': 48,
                    },
                    '_cbo_term': 'CBO:0000072',  # Chemotaxis (same as lchem)
                    '_model_context': 'Saturation parameter in chemotaxis term. Determines if cells respond linearly or with saturation to VEGF concentration.',

                    '_related_parameters': ['lchem'],
                    '_affects_output': ['response_linearity'],

                    '_value_mapping': {
                        0.0: 'Linear Response',
                        0.1: 'Saturating Response',
                    },
                    '_presets': {
                        'linear': 0.0,
                        'saturating': 0.1,
                    },

                    '_gui_hints': {
                        'widget_type': 'combobox',  # Dropdown for binary choice
                        'show_as_categorical': True,
                    },

                    '_biological_meaning': 'Represents receptor saturation. At high VEGF concentrations, all receptors may be occupied, limiting further response increase.',

                    '_parameter_type': 'biophysical',
                    '_measurable': True,
                    '_fitting_method': 'literature_based',
                    '_hidden_basic': True,  # Hide from basic users (advanced concept)
                },

                # =====================================================================
                # VEGF FIELD PARAMETERS
                # =====================================================================

                'vedir': {
                    '_default': self.defaults['vedir'],
                    '_updater': 'set',
                    '_emit': True,

                    '_display_name': 'VEGF Diffusion Rate',
                    '_description': 'Rate at which VEGF diffuses through the medium. Controls how quickly VEGF spreads from secreting cells.',
                    '_long_description': 'Diffusion coefficient D in the reaction-diffusion equation: ∂C/∂t = D∇²C - kC + S, where C is VEGF concentration, k is decay rate, and S is secretion. Higher D creates flatter gradients over longer distances. Units are in pixels²/MCS in the simulation.',

                    '_category': 'Growth Factors',
                    '_subcategory': 'VEGF Dynamics',
                    '_expert_level': 'intermediate',

                    '_unit': 'pixels²/MCS',
                    '_unit_long': 'Diffusion coefficient (simulation units)',
                    '_conversion_factor': None,  # Could add real units conversion

                    '_physiological_range': (0.1, 2.0),
                    '_recommended_range': (0.5, 1.5),
                    '_mathematical_range': (0.0, 10.0),
                    '_warning_threshold': 3.0,

                    '_reference_paper': {
                        **self.REFERENCES['merks2006'],
                        'value': 1.0,
                        'notes': 'Moderate diffusion rate',
                        'equation': 'D in reaction-diffusion PDE (Eq. 2)',
                        'page': 48,
                    },
                    '_cbo_term': 'CBO:0000090',  # Molecular diffusion
                    '_model_context': 'Diffusion term in VEGF field PDE: ∂C/∂t = D∇²C + production - decay. Solved using DiffusionSolverFE.',

                    '_related_parameters': ['veder', 'vesec', 'lchem'],
                    '_affects_output': [
                        'gradient_steepness',
                        'gradient_range',
                        'network_isotropy'
                    ],
                    '_dependencies': {
                        'veder': 'Balance D/k determines gradient length scale: λ ~ sqrt(D/k)',
                        'vesec': 'High diffusion smooths out discrete secretion sources',
                    },

                    '_presets': {
                        'slow_diffusion': 0.5,
                        'moderate_diffusion': 1.0,
                        'fast_diffusion': 1.5,
                        'very_fast_diffusion': 2.0,
                    },

                    '_gui_hints': {
                        'widget_type': 'slider',
                        'slider_step': 0.1,
                        'show_preset_buttons': True,
                    },

                    '_biological_meaning': 'Represents VEGF diffusion through tissue/medium. In vivo, VEGF diffusion is restricted by ECM binding and sequestration. Higher values simulate more fluid environments.',

                    '_visual_effects': {
                        'low_value': 'Steep local gradients, localized sprouting',
                        'high_value': 'Shallow global gradients, diffuse sprouting pattern',
                    },

                    '_parameter_type': 'kinetic',
                    '_measurable': True,
                    '_fitting_method': 'literature_based',
                    '_experimental_assay': 'Fluorescence recovery after photobleaching (FRAP)',
                },

                'veder': {
                    '_default': self.defaults['veder'],
                    '_updater': 'set',
                    '_emit': True,

                    '_display_name': 'VEGF Decay Rate',
                    '_description': 'Rate at which VEGF is degraded or cleared from the medium. Higher values shorten VEGF range.',
                    '_long_description': 'Decay/degradation rate k in the reaction-diffusion equation. Represents proteolytic degradation, receptor-mediated uptake, and clearance. The characteristic length scale of VEGF gradients is λ ~ sqrt(D/k), so higher decay shortens gradient range.',

                    '_category': 'Growth Factors',
                    '_subcategory': 'VEGF Dynamics',
                    '_expert_level': 'intermediate',

                    '_unit': '1/MCS',
                    '_unit_long': 'Decay rate (inverse time)',
                    '_conversion_factor': None,

                    '_physiological_range': (0.1, 0.5),
                    '_recommended_range': (0.2, 0.4),
                    '_mathematical_range': (0.0, 2.0),
                    '_warning_threshold': 0.8,

                    '_reference_paper': {
                        **self.REFERENCES['merks2006'],
                        'value': 0.3,
                        'notes': 'Moderate decay rate',
                        'equation': 'k in reaction-diffusion PDE (Eq. 2)',
                        'page': 48,
                    },
                    '_cbo_term': 'CBO:0000091',  # Molecular degradation
                    '_model_context': 'Decay term in VEGF field PDE. Combined with diffusion, determines gradient spatial scale.',

                    '_related_parameters': ['vedir', 'vesec'],
                    '_affects_output': [
                        'gradient_range',
                        'steady_state_concentration',
                        'gradient_persistence'
                    ],
                    '_dependencies': {
                        'vedir': 'Length scale λ ~ sqrt(vedir/veder)',
                        'vesec': 'Steady-state level ~ vesec/veder',
                    },

                    '_presets': {
                        'slow_decay': 0.1,
                        'moderate_decay': 0.3,
                        'fast_decay': 0.5,
                    },

                    '_gui_hints': {
                        'widget_type': 'slider',
                        'slider_step': 0.05,
                        'show_preset_buttons': True,
                    },

                    '_biological_meaning': 'Represents VEGF half-life in tissue. In vivo, VEGF is rapidly cleared by receptor binding and proteolysis. Higher values simulate more efficient clearance.',

                    '_visual_effects': {
                        'low_value': 'Long-range gradients, distant cell response',
                        'high_value': 'Short-range gradients, localized cell response',
                    },

                    '_parameter_type': 'kinetic',
                    '_measurable': True,
                    '_fitting_method': 'literature_based',
                    '_experimental_assay': 'ELISA time course, radiolabeled VEGF clearance',
                },

                'vesec': {
                    '_default': self.defaults['vesec'],
                    '_updater': 'set',
                    '_emit': True,

                    '_display_name': 'VEGF Secretion Rate',
                    '_description': 'Rate at which Endothelial Cells secrete VEGF into the surrounding medium.',
                    '_long_description': 'Secretion source term S in the reaction-diffusion equation. Each EC pixel contributes VEGF at this rate per time step. Total production scales with cell count and size. Higher values increase steady-state VEGF levels and gradient amplitude.',

                    '_category': 'Growth Factors',
                    '_subcategory': 'VEGF Dynamics',
                    '_expert_level': 'intermediate',

                    '_unit': 'concentration/MCS',
                    '_unit_long': 'Secretion rate (concentration per time)',
                    '_conversion_factor': None,

                    '_physiological_range': (0.1, 0.5),
                    '_recommended_range': (0.2, 0.4),
                    '_mathematical_range': (0.0, 2.0),
                    '_warning_threshold': 0.8,

                    '_reference_paper': {
                        **self.REFERENCES['merks2006'],
                        'value': 0.3,
                        'notes': 'Moderate secretion rate',
                        'equation': 'S in reaction-diffusion PDE (Eq. 2)',
                        'page': 48,
                    },
                    '_cbo_term': 'CBO:0000092',  # Molecular secretion
                    '_model_context': 'Source term in VEGF field PDE. ECs secrete VEGF, creating gradients that guide chemotaxis.',

                    '_related_parameters': ['vedir', 'veder', 'lchem'],
                    '_affects_output': [
                        'vegf_concentration_levels',
                        'gradient_amplitude',
                        'autocrine_signaling_strength'
                    ],
                    '_dependencies': {
                        'veder': 'Steady-state concentration ~ vesec/veder',
                        'lchem': 'High secretion creates strong gradients that amplify chemotaxis',
                    },

                    '_presets': {
                        'low_secretion': 0.1,
                        'moderate_secretion': 0.3,
                        'high_secretion': 0.5,
                    },

                    '_gui_hints': {
                        'widget_type': 'slider',
                        'slider_step': 0.05,
                        'show_preset_buttons': True,
                    },

                    '_biological_meaning': 'Represents VEGF production rate by endothelial cells. In vivo, ECs can produce VEGF in response to hypoxia or autocrine loops during sprouting.',

                    '_visual_effects': {
                        'low_value': 'Weak gradients, slow sprouting',
                        'high_value': 'Strong gradients, rapid sprouting, autocrine amplification',
                    },

                    '_parameter_type': 'kinetic',
                    '_measurable': True,
                    '_fitting_method': 'literature_based',
                    '_experimental_assay': 'VEGF ELISA from conditioned media',
                },

                # =====================================================================
                # SIMULATION CONTROL PARAMETERS
                # =====================================================================

                'exp_name': {
                    '_default': self.defaults['exp_name'],
                    '_updater': 'set',
                    '_emit': True,

                    '_display_name': 'Experiment Name',
                    '_description': 'Unique identifier for this simulation run. Used for data storage organization in Zarr hierarchy.',

                    '_category': 'Simulation Setup',
                    '_subcategory': 'Identification',
                    '_expert_level': 'basic',

                    '_unit': 'text',
                    '_unit_long': 'Alphanumeric identifier',

                    '_validation_pattern': r'^[a-zA-Z0-9_-]+$',  # Regex for valid names
                    '_max_length': 50,

                    '_gui_hints': {
                        'widget_type': 'lineedit',
                        'placeholder': 'Enter experiment name...',
                        'auto_suggest': True,  # Suggest names based on parameter changes
                    },

                    '_parameter_type': 'control',
                    '_auto_generate': True,  # Can be auto-generated from parameters
                },

                'sim_time': {
                    '_default': 100.0,
                    '_updater': 'set',
                    '_emit': True,

                    '_display_name': 'Simulation Duration',
                    '_description': 'Total number of Monte Carlo Steps (MCS) to simulate.',
                    '_long_description': 'Each MCS represents one attempted update per lattice pixel on average. Typical simulations run 100-1000 MCS. Longer runs show equilibration and remodeling.',

                    '_category': 'Simulation Setup',
                    '_subcategory': 'Duration',
                    '_expert_level': 'basic',

                    '_unit': 'MCS',
                    '_unit_long': 'Monte Carlo Steps',
                    '_conversion_factor': None,

                    '_physiological_range': (10, 1000),
                    '_recommended_range': (50, 500),

                    '_presets': {
                        'quick_test': 50,
                        'standard': 100,
                        'extended': 500,
                        'long_equilibration': 1000,
                    },

                    '_gui_hints': {
                        'widget_type': 'spinbox',
                        'show_preset_buttons': True,
                    },

                    '_affects_output': ['simulation_runtime'],
                    '_performance_impact': 'high',  # Directly affects runtime

                    '_parameter_type': 'control',
                },

                'write_frequency': {
                    '_default': 10,
                    '_updater': 'set',
                    '_emit': True,
                    '_display_name': 'Data Save Interval',
                    '_description': 'How often to save simulation data to Zarr store',
                    '_long_description': 'Controls the frequency of data writing in Monte Carlo Steps (MCS). '
                                      'Lower values provide more time resolution but increase I/O overhead. '
                                      'Higher values reduce file size and may improve stability on Windows.',
                    '_category': 'Simulation Setup',
                    '_subcategory': 'Data Management',
                    '_unit': 'MCS',
                    '_physiological_range': (5, 100),
                    '_recommended_range': (10, 50),
                    '_mathematical_range': (1, 1000),
                    '_gui_hints': {
                        'widget_type': 'spinbox',
                        'slider_step': 5,
                    },
                    '_biological_meaning': 'Determines temporal resolution of saved data',
                    '_expert_level': 'intermediate',
                    '_presets': {
                        'high_resolution': 5,
                        'standard': 10,
                        'low_resolution': 20,
                        'minimal_io': 50,
                    },
                },
            },

            # =====================================================================
            # OUTPUTS
            # =====================================================================

            'outputs': {
                'timestep': {
                    '_default': 0,
                    '_updater': 'set',
                    '_emit': True,
                    '_description': 'Current simulation timestep (MCS)',
                    '_unit': 'MCS',
                },
                'experiment_metadata': {
                    '_default': {},
                    '_updater': 'set',
                    '_emit': True,
                    '_description': 'Complete experiment metadata including parameters, timestamps, and validation results',
                },
            }
        }

    def get_parameter_categories(self):
        """
        Extract hierarchical parameter structure for GUI tab generation.

        Returns:
            dict: {category: {subcategory: [param_names]}}
        """
        schema = self.ports_schema()
        structure = {}

        for param_name, param_info in schema['inputs'].items():
            category = param_info.get('_category', 'General')
            subcategory = param_info.get('_subcategory', 'Parameters')

            if category not in structure:
                structure[category] = {}
            if subcategory not in structure[category]:
                structure[category][subcategory] = []

            structure[category][subcategory].append(param_name)

        return structure

    def get_preset_names(self):
        """Get list of available preset names."""
        return list(self.PRESETS.keys())

    def get_preset(self, preset_name):
        """
        Get complete preset configuration.

        Args:
            preset_name (str): Name of preset to retrieve

        Returns:
            dict: Preset configuration with parameters and metadata
        """
        if preset_name not in self.PRESETS:
            raise ValueError(f"Preset '{preset_name}' not found")
        return self.PRESETS[preset_name]

    def apply_preset(self, preset_name):
        """
        Apply a preset to current parameters.

        Args:
            preset_name (str): Name of preset to apply

        Returns:
            dict: Updated parameter values
        """
        preset = self.get_preset(preset_name)
        self.parameters.update(preset['parameters'])
        return preset['parameters']

    def get_parameter_by_level(self, level='basic'):
        """
        Get parameters filtered by expertise level.

        Args:
            level (str): 'basic', 'intermediate', or 'advanced'

        Returns:
            list: Parameter names at or below specified level
        """
        schema = self.ports_schema()
        level_order = {'basic': 0, 'intermediate': 1, 'advanced': 2}
        target_level = level_order.get(level, 0)

        params = []
        for param_name, param_info in schema['inputs'].items():
            param_level = param_info.get('_expert_level', 'advanced')
            if level_order.get(param_level, 2) <= target_level:
                params.append(param_name)

        return params

    def validate_parameters(self, params):
        """
        Validate parameter set and return warnings/errors.

        Args:
            params (dict): Parameter values to validate

        Returns:
            dict: {param_name: [{'severity': str, 'message': str}]}
        """
        schema = self.ports_schema()
        issues = {}

        for param_name, value in params.items():
            if param_name not in schema['inputs']:
                continue

            param_schema = schema['inputs'][param_name]
            param_issues = []

            # Check physiological range
            if '_physiological_range' in param_schema:
                min_val, max_val = param_schema['_physiological_range']
                if value < min_val or value > max_val:
                    param_issues.append({
                        'severity': 'WARNING',
                        'message': f'Value {value} outside physiological range [{min_val}, {max_val}]',
                        'reference': param_schema.get('_reference_paper', {}).get('source', 'N/A')
                    })

            # Check warning threshold
            if '_warning_threshold' in param_schema:
                threshold = param_schema['_warning_threshold']
                if value > threshold:
                    param_issues.append({
                        'severity': 'WARNING',
                        'message': f'Value exceeds recommended threshold of {threshold}'
                    })

            if param_issues:
                issues[param_name] = param_issues

        return issues

    def generate_experiment_metadata(self, params, run_info=None):
        """
        Generate comprehensive experiment metadata.

        Args:
            params (dict): Parameter values used
            run_info (dict): Additional run information (optional)

        Returns:
            dict: Complete experiment metadata
        """
        from datetime import datetime
        import uuid

        schema = self.ports_schema()

        # Track parameter changes from defaults
        param_changes = {}
        for param_name, value in params.items():
            if param_name in schema['inputs']:
                default = schema['inputs'][param_name]['_default']
                if value != default:
                    param_changes[param_name] = {
                        'value': value,
                        'default': default,
                        'changed': True,
                    }

        metadata = {
            'experiment_id': str(uuid.uuid4()),
            'created_at': datetime.now().isoformat(),
            'model_name': 'Angiogenesis',
            'model_version': '1.0.0',
            'parameters': params,
            'parameter_changes': param_changes,
            'validation': self.validate_parameters(params),
            'schema_version': '1.0',
        }

        if run_info:
            metadata.update(run_info)

        return metadata

    # Original next_update method would remain unchanged

    def next_update(self, timestep, states):
        self.mcs += 1
        
        write_freq = self.parameters.get('write_frequency', 10)
        if self.mcs % write_freq == 0:
            self.core_sim.sim_input = ['write', self.mcs]
        else:
            self.core_sim.sim_input = ['skip_write', self.mcs]

        self.core_sim.step()
        
        return {
            'outputs': {
                'timestep': self.mcs
            }
        }

    def finish(self):
        self.core_sim.finish()
