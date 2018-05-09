#!/usr/bin/env python
from meap.gui_tools import (MEAPView, VGroup, Item, ImageResource,
                            SplashScreen, meap_splash)
from time import time, sleep
import sys,os
from traits.api import HasTraits, Button, Instance
from meap import __version__
# Force import of compiled pyd
from srvf_register import dynamic_programming_q2


class MEAPGreeter(HasTraits):
    preproc = Button("Preprocess")
    analyze = Button("Analyze")
    configure = Button("Configure MEAP")
    batch_spreadsheet = Button("Create Batch Spreadsheet")
    register_dZdt = Button("Batch Register dZ/dt")    
    
    def _preproc_fired(self):
        from meap.preprocessing import PreprocessingPipeline
        preproc=PreprocessingPipeline()
        preproc.configure_traits()
        
    def _analyze_fired(self):
        from meap.physio_analysis import PhysioExperiment
        pe = PhysioExperiment()
        pe.configure_traits()
        
    def _register_dZdt_fired(self):
        from meap.batch_warp_dzdt import BatchGroupRegisterDZDT
        batch = BatchGroupRegisterDZDT()
        batch.configure_traits()
        
    def _batch_spreadsheet_fired(self):
        from meap.make_batch_spreadsheet import BatchFileTool
        bft = BatchFileTool()
        bft.configure_traits()
        
    def _configure_fired(self):
        print "Not implemented yet!"
        
    traits_view=MEAPView(
        VGroup(
            Item("preproc"),
            Item("analyze"),
            Item("batch_spreadsheet"),
            Item("register_dZdt"),
            Item("configure"),
            show_labels=False
        )
    )
    
if __name__ == "__main__":
    print "Welcome to MEAP!"
    print "================"
    print ""
    print "Version:", __version__
    print "Please keep this window open. In case MEAP crashes, email"
    print "Matt the error message printed here and explain what you"
    print "were doing when it crashed. Please consider joining the"
    print "MEAP google group to post questions and receive updates"
    
    splash = SplashScreen(image=meap_splash)
    splash.open()
    greeter = MEAPGreeter()
    sleep(3)
    splash.close()
    greeter.configure_traits()
