#!/usr/bin/env python

from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'
from time import time, sleep
import sys,os
from traits.api import HasTraits, Button, Instance
from traitsui.api import View, HGroup, VGroup, Item
from meap import MEAPView
from chaco.api \
    import ArrayPlotData, ImageData, Plot, PlotGraphicsContext
from enable.component_editor import ComponentEditor
from chaco.tools.api import PanTool, ZoomTool
from pyface.image_resource import ImageResource
from pyface.api import SplashScreen
from meap import meap_splash, __version__


class MEAPGreeter(HasTraits):
    preproc = Button("Preprocess")
    analyze = Button("Analyze")
    configure = Button("Configure MEAP")
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
        
    def _configure_fired(self):
        print "Not implemented yet!"
        
    traits_view=MEAPView(
        VGroup(
            Item("preproc"),
            Item("analyze"),
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
