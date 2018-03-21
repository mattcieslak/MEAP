#!/usr/bin/env python
from traits.api import (HasTraits,  Array,  File, cached_property,
          Bool, Enum, Instance, on_trait_change, Property,
          DelegatesTo, Int, Button, List, Set, Float, Str,Directory)
import os
# Needed for Tabular adapter
from meap.traitsui import (Item,HGroup,VGroup, HSplit, ObjectColumn, 
                           TableEditor, ProgressDialog)
from traitsui.menu import OKButton, CancelButton
import numpy as np
import time
from meap import MEAPView, messagebox
from meap.io import load_from_disk, PhysioData
from glob import glob

import logging
logger = logging.getLogger(__name__)

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

    # For saving outputs
    file_suffix = Str()
    overwrite = Bool(False)
    input_directory = Directory()
    output_directory = Directory()
    files = List(Instance(FileToProcess))

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
        
        
    def _input_directory_changed(self):
        potential_files = glob(self.input_directory +"/*mea.mat")
        potential_files = [f for f in potential_files if not \
                           f.endswith("_aligned.mea.mat") ]
        
        # Check if the output already exists
        def make_output_file(input_file):
            return input_file[:-len(".mea.mat")] + self.file_suffix
        
        self.files = [
            FileToProcess(input_file = f, output_file=make_output_file(f)) \
            for f in potential_files
        ]
        
        
    def _b_run_fired(self):
        files_to_run = [f for f in self.files if not f.finished]
        
    

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
