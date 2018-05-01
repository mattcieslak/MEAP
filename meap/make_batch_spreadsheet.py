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
import pandas as pd

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

class BatchFileTool(HasTraits):

    # For saving outputs
    file_suffix = Str("_finished.mea.mat")
    input_file_extension = Enum(".mea.mat", ".acq", ".mat")
    overwrite = Bool(False)
    input_directory = Directory()
    output_directory = Directory()
    files = List(Instance(FileToProcess))
    spreadsheet_file = File(exists=False)
    save = Button("Save Spreadsheet")

    def _input_file_extension_changed(self):
        self._input_directory_changed()
        
    def _input_directory_changed(self):
        potential_files = glob(self.input_directory +"/*" + self.input_file_extension)
        potential_files = [f for f in potential_files if not \
                           f.endswith(self.file_suffix) ]
        
        # Check if the output already exists
        def make_output_file(input_file):
            return input_file[:-len(self.input_file_extension)] + self.file_suffix
        
        self.files = [
            FileToProcess(input_file = f, output_file=make_output_file(f)) \
            for f in potential_files
        ]
        
        # If no output directory is set, use the input directory
        if self.output_directory == '':
            self.output_directory = self.input_directory
        
        
    def _b_save_fired(self):
        def get_row():
            return {"file":"", "outfile":"", "weight":"",
                    "height_ft":"", "weight":"", "electrode_distance_front":"",
                    "electrode_distance_back":"", "electrode_distance_left":"",
                    "electrode_distance_right":"", "resp_max":"", "resp_min":"",
                    "in_mri":"", "control_base_impedance":""}
        rows = []
        for f in self.files:
            row = get_row()
            row['file'] = f.input_file
            row['outfile'] = f.output_file
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_excel(self.spreadsheet_file, index=False)
        
    

    mean_widgets =VGroup(
            Item("input_file_extension"),
            Item("input_directory"),
            Item("output_directory"),
            Item("file_suffix"),
            Item("spreadsheet_file"),
            Item("save",show_label=False)
    )

    traits_view = MEAPView(
        HSplit(
            Item("files",editor=files_table,show_label=False),
            mean_widgets),
        resizable=True,
        win_title="Create Batch Spreadsheet",
        width=800, height=700,
        buttons = [OKButton,CancelButton]
    )
