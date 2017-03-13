from meap.io import load_from_disk
from meap.pipeline import MEAPPipeline

mp = MEAPPipeline(file="/Users/matt/Desktop/Z_H_acqs/S01_1_DOT.acq")
mp.edit_traits()



from meap.io import load_from_disk
from meap.moving_ensemble import MovingEnsembler

phys = load_from_disk("/Users/matt/Desktop/Z_H_acqs/S01_1_DOT.mea.mat")
me = MovingEnsembler(physiodata=phys)
me.edit_traits()



