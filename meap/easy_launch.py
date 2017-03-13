#!/usr/bin/env python

import sys,os

def preprocess():
    os.environ['ETS_TOOLKIT']='qt4'
    os.environ['QT_API']='pyqt'
    from meap.preprocessing import PreprocessingPipeline
    
    if len(sys.argv) > 1:
        if not os.path.exists(sys.argv[1]):
            print "FILE DOES NOT EXIST"
            sys.exit(1)
        preproc=PreprocessingPipeline(input_path = sys.argv[1])
    else:
        preproc=PreprocessingPipeline()
    
    preproc.configure_traits()
    
def analyze():
    os.environ['ETS_TOOLKIT']='qt4'
    os.environ['QT_API']='pyqt'
    from meap.physio_analysis import PhysioExperiment
    
    if len(sys.argv) > 1:
        if not os.path.exists(sys.argv[1]):
            print "FILE DOES NOT EXIST"
            sys.exit(1)
        pe = PhysioExperiment(input_path=sys.argv[1])
    else:
        pe = PhysioExperiment()
    
    pe.configure_traits()
