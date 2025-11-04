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

    def __init__(self, store:str):
        self.store_path = store
        # Open zarr store and ensure experiments group exists
        root = zarr.open(self.store_path, mode='a')
        root.require_group('experiments')
        self.expname = None
        self.exparams = None

    def create_store(self):
        root = zarr.open(self.store_path, mode='w')
        root.create_group('experiments')

    def get_experiments(self):
        root = zarr.open(self.store_path, mode='r')
        return list(root['experiments'].keys())

    def setup_exp(self, expname:str, exparams:dict):
        # ALWAYS set self.expname and self.exparams, even if experiment exists
        self.expname = expname
        self.exparams = exparams

        # Only create group if it doesn't exist
        root = zarr.open(self.store_path, mode='a')
        if expname not in root['experiments'].keys():
            exp_group = root['experiments'].require_group(expname)
            exp_group.attrs['params'] = exparams
                
    def get_exp_params(self, expname:str):
        root = zarr.open(self.store_path, mode='r')
        if expname in root['experiments'].keys():
            exp_group = root['experiments'][expname]
            return exp_group.attrs['params']
        else:
            print(f'Experiment {expname} not found')
            return None

    def write_data(self, data:np.ndarray, step:int):
        if not self.expname:
            print('No experiment set???')
            return

        root = zarr.open(self.store_path, mode='a')
        exp_group = root['experiments'][self.expname]
        if str(step) in exp_group:
            print(f'Step {step} already exists, skipping')
            return
        mcs_group = exp_group.require_group(str(step))
        # Zarr v3 API: use array assignment instead of create_dataset with compression
        mcs_group['data'] = data

    def get_exp_length(self, expname:str):
        root = zarr.open(self.store_path, mode='r')
        exp_group = root['experiments'][expname]
        length = len(exp_group)
        return length

    def grab_cellview(self, expname:str, step:int):
        root = zarr.open(self.store_path, mode='r')
        exp_group = root['experiments'][expname]
        mcs_group = exp_group[str(step)]['data']
        cell_data = mcs_group[:,:,:,0]
        cell_data = np.squeeze(cell_data)
        return cell_data

    def grab_fieldview(self, expname:str, step:int):
        root = zarr.open(self.store_path, mode='r')
        exp_group = root['experiments'][expname]
        mcs_group = exp_group[str(step)]['data']
        field_data = mcs_group[:,:,:,2]
        field_data = np.squeeze(field_data)
        return field_data

    def analyse_one_step(self, step:int, queue:bool=False, expname:str=None):
        expname = self.expname if expname is None else expname
        root = zarr.open(self.store_path, mode='r')
        exp_group = root['experiments'][expname]
        mcs_group = exp_group[str(step)]['data']
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

        out_dir = os.path.join(os.getcwd(), 'OutStore')
        self.core_sim = CC3DSimService(output_dir=out_dir)
        self.core_sim.register_specs(self.specs)

        self.store = 'temp_store.zarr'
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
        return {
            'inputs': {
                'jee': {
                    '_default': self.defaults['jee'],
                    '_updater': 'set',
                    '_emit': True,
                    '_display_name': 'EC-EC Adhesion',
                    '_description': 'Contact energy between two Endothelial Cells (EC). Lower values promote stronger adhesion.',
                    '_category': 'Cell Properties',
                    '_subcategory': 'Adhesion',
                    '_unit': 'Arbitrary Energy Units',
                    '_physiological_range': (2.0, 16.0),
                    '_expert_level': 'intermediate',
                },
                'jem': {
                    '_default': self.defaults['jem'],
                    '_updater': 'set',
                    '_emit': True,
                    '_display_name': 'EC-Medium Adhesion',
                    '_description': 'Contact energy between ECs and medium. Higher values cause the initial cell network to fragment into more compact, separate clusters to minimize contact with the medium.',
                    '_category': 'Cell Properties',
                    '_subcategory': 'Adhesion',
                    '_unit': 'Arbitrary Energy Units',
                    '_physiological_range': (2.0, 16.0),
                    '_expert_level': 'intermediate',
                },
                'lchem': {
                    '_default': self.defaults['lchem'],
                    '_updater': 'set',
                    '_emit': True,
                    '_display_name': 'Chemotaxis Strength (Lambda)',
                    '_description': 'Coefficient that determines the strength of the chemotactic response of ECs to the VEGF gradient.',
                    '_category': 'Cell Properties',
                    '_subcategory': 'Chemotaxis',
                    '_unit': 'Arbitrary Energy Units',
                    '_physiological_range': (0, 1000),
                    '_expert_level': 'basic',
                },
                'lsc': {
                    '_default': self.defaults['lsc'],
                    '_updater': 'set',
                    '_emit': True,
                    '_display_name': 'Chemotaxis Saturation',
                    '_description': 'Saturation coefficient for the chemotaxis response. A binary choice in the original UI (0 or 0.1).',
                    '_category': 'Cell Properties',
                    '_subcategory': 'Chemotaxis',
                    '_unit': 'dimensionless',
                    '_physiological_range': (0.0, 0.1),
                    '_expert_level': 'advanced',
                },
                'vedir': {
                    '_default': self.defaults['vedir'],
                    '_updater': 'set',
                    '_emit': True,
                    '_display_name': 'VEGF Diffusion Rate',
                    '_description': 'The global diffusion rate of the Vascular Endothelial Growth Factor (VEGF) in the medium.',
                    '_category': 'Growth Factors',
                    '_subcategory': 'VEGF Dynamics',
                    '_unit': 'pixels^2/MCS',
                    '_physiological_range': (0.1, 2.0),
                    '_expert_level': 'intermediate',
                },
                'veder': {
                    '_default': self.defaults['veder'],
                    '_updater': 'set',
                    '_emit': True,
                    '_display_name': 'VEGF Decay Rate',
                    '_description': 'The global decay rate of VEGF in the medium.',
                    '_category': 'Growth Factors',
                    '_subcategory': 'VEGF Dynamics',
                    '_unit': '1/MCS',
                    '_physiological_range': (0.1, 0.5),
                    '_expert_level': 'intermediate',
                },
                'vesec': {
                    '_default': self.defaults['vesec'],
                    '_updater': 'set',
                    '_emit': True,
                    '_display_name': 'VEGF Secretion Rate',
                    '_description': 'The rate at which Endothelial Cells (EC) secrete VEGF.',
                    '_category': 'Growth Factors',
                    '_subcategory': 'VEGF Dynamics',
                    '_unit': 'concentration/MCS',
                    '_physiological_range': (0.1, 0.5),
                    '_expert_level': 'intermediate',
                },
            },
            'outputs': {
                'timestep': {
                    '_default': 0,
                    '_updater': 'set',
                    '_emit': True,
                },
            }
        }

    def next_update(self, timestep, states):
        self.mcs += 1
        
        if self.mcs % 10 == 0:
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
