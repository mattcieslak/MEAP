#!/usr/bin/env python
from traits.api import (HasTraits, Str, Array, Float,CFloat, Dict,
          Bool, Enum, Instance, on_trait_change,File,Property,
          Range, DelegatesTo, Int, Button, List, Color,Set,Event,CBool )
from filters import bandpass, smooth
import time
import numpy as np

from traitsui.api import (Group, View, Item, TableEditor,
        ObjectColumn, ExpressionColumn,HSplit, EnumEditor)
from traitsui.table_column \
    import ObjectColumn, ExpressionColumn

from meap.moving_ensemble import MovingEnsembler
from meap.beat import EnsembleAveragedHeartBeat,get_global_ensemble_average, GlobalEnsembleAveragedHeartBeat
from meap.io import PhysioData, load_from_disk
from numpy import nanmean, nanstd
from meap import MEAPView

import logging
logger = logging.getLogger(__name__)

# Stuff for excel
import xlrd
import xlwt
import os

from chaco.api import ArrayPlotData, ColorBar, \
                                 ColormappedSelectionOverlay, HPlotContainer, \
                                 jet,gist_rainbow, LinearMapper, Plot
from chaco.tools.api import PanTool, ZoomTool
from chaco.tools.line_inspector import LineInspector
from chaco.scatterplot import ScatterPlot
from chaco.lineplot import LinePlot

from enable.component_editor import ComponentEditor
from enable.api import ColorTrait
from chaco.api import marker_trait
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA

# These shouldn't change within a single acq file
CONSTANTS=["height", "weight", "resp_max", "resp_min", "gender"]

# ExpEvent properties to use for outlier detection
ODDITY_PROPERTIES = ["q_time","b_time","x_time","ea_sv"]


from meap.physio_analysis import EventType, ExpEvent, \
     PhysioFileAnalysis, PhysioExperiment

class MovingExpEvent(ExpEvent):
    pass

class PhysioFileMEAAnalysis(PhysioFileAnalysis):
    ensembler = Instance(MovingEnsembler)
    
    def __init__(self,**traits):
        super(PhysioFileMEAAnalysis,self).__init__(**traits)

    def _ensembler_default(self):
        return MovingEnsembler(physiodata=self.physiodata,
                               global_ensemble=self.global_ensemble)
        
    def ensemble_average(self):
        """
        Computes cv properties from epochs of data 
        """
        beats = self.btrain
        # For each event
        for ev_num, event in enumerate(self.events):
            event.ensemble_average()
    
    def process_data(self):
        if self.interactive:
            self.pipeline.configure_traits()
        else:
            self.pipeline.nongui_process()
            
ExpEvent.add_class_trait("parent",Instance(PhysioFileMEAAnalysis))


class MEAPhysioExperiment(PhysioExperiment):
    events = List(Instance(ExpEvent))
    physio_files = List(Instance(PhysioFileMEAAnalysis))
    
    def _b_save_fired(self):
        raise NotImplementedError()
    
    def file_constructur(self):
        return PhysioFileMEAAnalysis
    
    def event_constructor(self):
        return MovingEnsembler
    
    def run(self):
        # read the excel file
        self.open_xls()
        # organize all the data from it
        if not self.events_processed:
            self._process_events()
        # Run the pipeline on each acq_file
        self.extract_mea()
        
    def extract_mea(self):
        stacks = {}
        for evt in self.conditions:
            stacks[evt] = {}
            
        for pf in self.physio_files:
            pf.extract_mea()
            
            
    

    
    def save_mea_document(self,output_path=""):
        #self.ensemble_average()
        if not output_path:
            output_path = self.output_xls
        
    traits_view = MEAPView(
      HSplit(
        Group(
            Group(
                Item("input_path",label="Input XLS"), 
                Item("output_directory",label="Output Directory"),
                Item("b_run", show_label=False),
                Item("b_save", show_label=False),
                orientation="horizontal"),
        ),
            
        ),
      resizable=True,
      win_title="MEA Analysis"
      )

    

PhysioFileMEAAnalysis.add_class_trait("parent",Instance(MEAPhysioExperiment))
