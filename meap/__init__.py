import os
import numpy as np
import sys
import logging
import cPickle as pickle
from traitsui.api import View
from traits.api import Str
from traitsui.message import Message

# Begin a logging output
logging.basicConfig(format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Chaco spits out lots of deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

DEFAULT_SAMPLING_RATE=1000
SEARCH_WINDOW=30 #samples
BLOOD_RESISTIVITY=135. # Ohms cm
n_regions=0
__version__= "1.1.0"

# Are we bundled?
if getattr( sys, 'frozen', False ) :
    _ROOT = sys._MEIPASS
    # running in a bundle
else :
    _ROOT = os.path.abspath(os.path.dirname(__file__))

# MEAP only knows what to do with a couple of signals
SUPPORTED_SIGNALS=["ecg", "ecg2", "dzdt", "z0", "bp",
                   "respiration","systolic","diastolic",
                   "mri_trigger", "event", "pulse_ox"]

SMOOTHING_WINDOWS=("hanning", "flat", "hamming", "bartlett", "blackman")

# Use this to keep signal colors consistent across guis
colors = {"ecg":"green", 
          "ecg2":"green",
          "dzdt":"red",
          "resp_corrected_dzdt":"darkred",
          "bp":"magenta",
          "respiration":"maroon",
          "z0":"cyan",
          "resp_corrected_z0":"blue",
          "systolic":"magenta",
          "diastolic":"purple",
          "pulse_ox":"gray",
          "event":"darkgreen",
          "mri_trigger":"darkred"
         }

# Signals which can be ensemble-averaged
ENSEMBLE_SIGNALS = set((
    'ecg', 
    'ecg2', 
    'z0', 
    "resp_corrected_z0",
    "dzdt",
    "resp_corrected_dzdt",
    "bp",
    "systolic",
    "diastolic",
    "pulse_ox"
    ))

from pyface.image_resource import ImageResource
icon = ImageResource(
    os.path.join(_ROOT,"resources/logo512x512.png"))
meap_splash = ImageResource(os.path.join(_ROOT, "resources/meap.png"))

class MEAPView(View):
    win_title=Str
    resizable=True
    def __init__(self, *args, **traits):
        super(MEAPView,self).__init__(*args, **traits)
        self.title = "MEAP [ v%s ]: "%__version__ + self.win_title
        self.icon = icon
        
def messagebox(msg,title="Message",buttons=["OK"]):
    m = Message(message=msg)
    ui = m.edit_traits( 
         view = MEAPView(
                        ["message~", "|<>"], title=title,buttons=buttons,kind="modal"
         )
    )
        
    

from traitsui.api import Action
ParentButton = Action(name="Back",
                          action="show_parent",
                          image=icon
                          )

def fail(msg,interactive=False):
    logger.critical(msg)
    if interactive:
        messagebox(msg,buttons=["Cancel"],title="Fatal error!")
    

        

def print_classes(obj,pad=""):
    """ 
    Use this function when having trouble pickling
    a HasTraits object.
    """
    for k,v in obj.__getstate__().iteritems():
        #if hasattr(v,"__getstate__"):
        #    print_classes(v,pad=pad+"  ")
        #else:
        try:
            xx = pickle.dumps(v)
        except TypeError:
            print "failed to pickle", k
    

def outlier_feature_function(contents, is_beat=True):
    attrs_to_get = []
    
    # Getting directly from a heartbeat?     
    if is_beat:
        pre = ""
    else:
        pre = "ensemble_averaged_heartbeat."
        
    if "dzdt" in contents:
        attrs_to_get.extend([pre+"b.time",pre+"x.time"])
    if "ecg" in contents:
        attrs_to_get.append(pre+"r.value")
    if "bp" in contents:
        attrs_to_get.extend([pre+"systole.value",pre+"diastole.value"])
        
    if len(attrs_to_get) == 0:
        return lambda x: 1
    
    return lambda x: np.array(
        [reduce(getattr,attr.split("."),x) for attr in attrs_to_get])
        

def outlier_feature_function_for_physio_experiment(pe):
    """
    Returns a function that can be applied to ExpEvents, 
    extracting features that can be used for outlier detection
    """
    feature_sets = [pf.physiodata.contents for \
                    pf in pe.physio_files]
    common_features = set.intersection(*feature_sets)
    return outlier_feature_function(common_features,is_beat=False)
    
