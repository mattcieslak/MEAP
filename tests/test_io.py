import sys, os
sys.path.append("..")
import numpy as np
from meap.readers import AcqImporter
from meap.data_plot import DataPlot
from meap.pan_tomkins import PanTomkinsDetector
from meap.beat import HeartBeat, EnsembleAveragedHeartBeat
from meap.beat_train import BeatTrain
from meap.pipeline import MEAPPipeline
from meap.io import load_from_disk

acq_file = "../valsalva/Christina_Valsalva.acq"
mea_file = "../valsalva/test.mea"

# Open the acqknowledge file
#reader = AcqImporter(path=acq_file)
#physiodata = reader.get_physiodata()
mp=MEAPPipeline(infile=acq_file)
#physiodata.censored_intervals = np.array(
#    [[50,100],[300,400]])
#physiodata.censoring_sources = ["ecg", "dzdt"]

#ptk = PanTomkinsDetector(physiodata=physiodata)
#ptk.bandpass_max = 19

# Change some of the properties
#physiodata.ens_avg_b_time = 50
#physiodata.ens_avg_p_time = 55
#physiodata.ens_avg_q_time = 65
#physiodata.ens_avg_c_time = 75
#physiodata.ens_avg_x_time = 85
#physiodata.using_hand_marked_point_priors = True


#print "censor intervals", physiodata.censored_regions

#mp0 = MEAPPipeline(physiodata = physiodata)
#mp0.mark_global_ensemble_average()
#mp0.edit_traits()
#physiodata.save(mea_file)

def check_matfile():
    physiodata2 = load_from_disk("/home/matt/test3.mea.mat")
    print physiodata2.censored_regions
    mp1 = MEAPPipeline(physiodata=physiodata2)
    mp1.edit_traits()


mp = MEAPPipeline()