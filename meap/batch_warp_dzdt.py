#!/usr/bin/env python
from traits.api import (HasTraits,  Array,  File, cached_property,
          Bool, Enum, Instance, on_trait_change, Property,
          DelegatesTo, Int, Button, List, Set, Float, Str,Directory)
import os
# Needed for Tabular adapter
from meap.gui_tools import (Item,HGroup,VGroup, HSplit, ObjectColumn, 
                           TableEditor, ProgressDialog)
from traitsui.menu import OKButton, CancelButton
import numpy as np
import time
from meap.gui_tools import MEAPView, messagebox
from meap.io import load_from_disk, PhysioData
from glob import glob
import multiprocessing

import logging
logger = logging.getLogger(__name__)

def process_physio_file((physio_file, output_file, srvf_lambda, srvf_max_karcher_iterations,
        srvf_update_min, srvf_karcher_mean_subset_size, srvf_use_moving_ensembled,
        bspline_before_warping, dzdt_num_inputs_to_group_warping, srvf_t_min,
        srvf_t_max, n_modes)):
    
    from meap.dzdt_warping import GroupRegisterDZDT
    from meap.io import load_from_disk
    
    # If file exists, don't overwrite
    import os
    if os.path.exists(output_file):
        return True
    
    # In case an OMP version is being used
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Load the input data
    phys = load_from_disk(physio_file)
    
    # Check if waveform points have been enabled
    if 0.0 in (phys.ens_avg_c_time, phys.ens_avg_b_time):
        raise ValueError("Label Waveform points is required")

    # Check if moving ensembles are to be used. If so, they must exist
    if srvf_use_moving_ensembled and phys.mea_dzdt_matrix.size == 0:
        raise ValueError("Moving ensembles are not available")
    
    reg = GroupRegisterDZDT(physiodata=phys)
    reg.srvf_lambda = srvf_lambda
    reg.srvf_max_karcher_iterations= srvf_max_karcher_iterations
    reg.srvf_update_min = srvf_update_min
    reg.srvf_karcher_mean_subset_size = srvf_karcher_mean_subset_size
    reg.srvf_use_moving_ensembled = srvf_use_moving_ensembled
    reg.bspline_before_warping = bspline_before_warping
    reg.dzdt_num_inputs_to_group_warping = dzdt_num_inputs_to_group_warping
    reg.srvf_t_min = srvf_t_min
    reg.srvf_t_max = srvf_t_max

    # Calculate the initial karcher mean
    reg.calculate_karcher_mean()
    reg.align_all_beats_to_initial()

    # Find the modes
    reg.n_modes = n_modes
    reg.detect_modes()
    
    abs_out = os.path.abspath(output_file)
    reg.physiodata.save(abs_out)

    # produces 2 files
    logger.info("saved %s", abs_out)
    return os.path.abspath(abs_out)
    

class FileToProcess(HasTraits):
    input_file = File()
    output_file = File()
    finished = Bool(False)
    
    def _output_file_changed(self):
        self.finished = os.path.exists(self.output_file)
        
class PhysioFileColumn(ObjectColumn):
    def get_cell_color(self,object):
        if object.finished: return "light blue"
        return


files_table = TableEditor(
    columns = [ 
        PhysioFileColumn(name="input_file", width=1.0, 
                         editable=False,label="Input File"),
        PhysioFileColumn(name="output_file", width=1.0, 
                         editable=False,label="Output File"),
        ],
    auto_size  = True
)

class BatchGroupRegisterDZDT(HasTraits):
    # Dummy physiodata to get defaults
    physiodata = Instance(PhysioData)
    
    # Parameters for SRVF registration
    srvf_lambda = DelegatesTo("physiodata")
    srvf_max_karcher_iterations = DelegatesTo("physiodata")
    srvf_update_min = DelegatesTo("physiodata")
    srvf_karcher_mean_subset_size = DelegatesTo("physiodata")
    srvf_use_moving_ensembled = DelegatesTo("physiodata")
    bspline_before_warping = DelegatesTo("physiodata")
    dzdt_num_inputs_to_group_warping = DelegatesTo("physiodata")
    srvf_t_min = DelegatesTo("physiodata")
    srvf_t_max = DelegatesTo("physiodata")
    n_modes = DelegatesTo("physiodata")

    # For saving outputs
    num_cores = Int(1)
    file_suffix = Str()
    overwrite = Bool(False)
    input_directory = Directory()
    output_directory = Directory()
    files = List(Instance(FileToProcess))

    b_run = Button(label="Run")
    
    def __init__(self,**traits):
        super(BatchGroupRegisterDZDT, self).__init__(**traits)
        self.num_cores = multiprocessing.cpu_count()
        
    def _physiodata_default(self):
        return PhysioData()

    @on_trait_change(
       ("physiodata.srvf_karcher_iterations, "
        "physiodata.srvf_use_moving_ensembled, "
        "physiodata.srvf_karcher_mean_subset_size, "
        "physiodata.bspline_before_warping"
    ))
    def params_edited(self):
        self._input_directory_changed()
        
    def _num_cores_changed(self):
        num_cores = multiprocessing.cpu_count()
        
        if self.num_cores < 1 or self.num_cores > num_cores:
            self.num_cores = multiprocessing.cpu_count()
        
    def _input_directory_changed(self):
        potential_files = glob(self.input_directory +"/*mea.mat")
        potential_files = [f for f in potential_files if not \
                           f.endswith("_aligned.mea.mat") ]
        
        # Check if the output already exists
        def make_output_file(input_file):
            suffix = os.path.split(os.path.abspath(input_file))[1]
            suffix = suffix[:-len(".mea.mat")] + self.file_suffix
            return self.output_directory + "/" + suffix
        
        self.files = [
            FileToProcess(input_file = f, output_file=make_output_file(f)) \
            for f in potential_files
        ]
        
        
    def _b_run_fired(self):
        files_to_run = [f for f in self.files if not f.finished]
        
        logger.info("Processing %d files using %d cpus", len(files_to_run), self.num_cores)
        
        pool = multiprocessing.Pool(self.num_cores)
        
        arglist = [(f.input_file, f.output_file, self.srvf_lambda, self.srvf_max_karcher_iterations,
        self.srvf_update_min, self.srvf_karcher_mean_subset_size, self.srvf_use_moving_ensembled,
        self.bspline_before_warping, self.dzdt_num_inputs_to_group_warping, self.srvf_t_min,
        self.srvf_t_max, self.n_modes) for f in files_to_run]
        pool.map(process_physio_file, arglist)
        
    

    mean_widgets =VGroup(
            Item("input_directory"),
            Item("output_directory"),
            Item("file_suffix"),
            Item("srvf_use_moving_ensembled",
                    label="Use Moving Ensembled dZ/dt"),
            Item("bspline_before_warping",label="B Spline smoothing"),
            Item("srvf_lambda",label="Lambda Value"),
            Item("dzdt_num_inputs_to_group_warping",
                    label="Template uses N beats"),
            Item("srvf_max_karcher_iterations", label="Max Iterations"),
            Item("n_modes"), Item("srvf_t_min"), Item("srvf_t_max"),
            Item("num_cores"),
            Item("b_run", show_label=False)
            
    )

    traits_view = MEAPView(
        HSplit(
            Item("files",editor=files_table,show_label=False),
            mean_widgets),
        resizable=True,
        win_title="Batch Warp dZ/dt",
        width=800, height=700,
        buttons = [OKButton,CancelButton]
    )
