from meap import fail, MEAPView
from meap.readers import Importer, AcqImporter, MatfileImporter
from meap.pan_tomkins import PanTomkinsDetector
from meap.moving_ensemble import MovingEnsembler
from meap.resp_preproc import RespirationProcessor
from meap.beat import GlobalEnsembleAveragedHeartBeat
from meap.subject_info import SubjectInfo
from meap.io import PhysioData, load_from_disk
from meap.data_plot import DataPlot
from meap.physio_regressors import FMRITool
from meap.dzdt_warping import GroupRegisterDZDT
import os
from traits.api import (HasTraits, Bool,  Instance, File, Button, Dict)
from traitsui.api import Item, VGroup, spring

import logging
logger = logging.getLogger(__name__)


class MEAPPipeline(HasTraits):
    physiodata = Instance(PhysioData)
    file = File
    outfile = File
    mapping_txt = File
    
    importer = Instance(Importer)
    importer_kwargs = Dict
    b_import = Button(label="Import data")
    b_subject_info = Button(label="Subject Info")
    b_inspect = Button(label="Inspect data")
    b_resp = Button(label="Process Resp")
    b_detect = Button(label="Detect QRS Complexes")
    b_custom_points = Button(label="Label Waveform Points")
    b_moving_ens = Button(label="Compute Moving Ensembles")
    b_register = Button(label="Register dZ/dt")
    b_fmri_tool = Button(label="Process fMRI")
    b_load_experiment = Button(label="Load Design File")
    b_save = Button(label="save .meap file")
    b_clear_mem = Button(label="Clear Memory")
    interactive = Bool(False)
    saved = Bool(False)
    peaks_detected = Bool(False)
    global_ensemble_marked = Bool(False)

    def _b_clear_mem_fired(self):
        self.physiodata = None
        self.global_ensemble_marked = False
        self.peaks_detected = False

    def import_data(self):
        logger.info("Loading %s", self.file)
        
        for k,v in self.importer_kwargs.iteritems():
            logger.info("Using attr '%s' from excel file",k)
        
        if not os.path.exists(self.file):
            fail("unable to open file %s"%self.file, interactive=self.interactive)
            return

        elif self.file.endswith(".acq"):
            importer = AcqImporter(path = str(self.file),
                        mapping_txt = self.mapping_txt, **self.importer_kwargs)

        elif self.file.endswith(".mea.mat"):
            pd = load_from_disk(self.file)
            self.physiodata = pd
            if self.physiodata.using_hand_marked_point_priors:
                self.global_ensemble_marked = True
            if self.physiodata.peak_indices.size > 0:
                self.peaks_detected = True
            return
        elif self.file.endswith(".mat"):
            importer = MatfileImporter(path = str(self.file),
                                        **self.importer_kwargs)
        else:
            return
            
        ui = importer.edit_traits(kind="livemodal")
        if not ui.result:
            logger.info("Import cancelled")
            return
        self.physiodata = importer.get_physiodata()
    
    def _b_import_fired(self):
        self.import_data()

    def _b_resp_fired(self):
        RespirationProcessor(physiodata=self.physiodata).edit_traits(kind="livemodal")

    def _b_subject_info_fired(self):
        SubjectInfo(physiodata = self.physiodata).edit_traits(kind="livemodal")
        
    def _b_custom_points_fired(self):
        ge = GlobalEnsembleAveragedHeartBeat(physiodata=self.physiodata)
        if not self.physiodata.using_hand_marked_point_priors:
            ge.mark_points()
        ge.edit_traits(kind="livemodal")
        self.physiodata.using_hand_marked_point_priors = True
        self.global_ensemble_marked = True
    
    def _b_inspect_fired(self):
        DataPlot(physiodata=self.physiodata).edit_traits()
    
    def _b_detect_fired(self):
        detector = PanTomkinsDetector(physiodata=self.physiodata)
        ui = detector.edit_traits(kind="livemodal")
        if ui.result:
            self.peaks_detected = True
    
    def _b_moving_ens_fired(self):
        MovingEnsembler(physiodata=self.physiodata).edit_traits()
        
    def _b_register_fired(self):
        GroupRegisterDZDT(physiodata=self.physiodata).edit_traits()
        
    def _b_save_fired(self):
        print "writing", self.outfile
        if os.path.exists(self.outfile):
            logger.warn("%s already exists", self.outfile)
        self.physiodata.save(self.outfile)
        self.saved = True
        logger.info("saved %s",self.outfile)

    def _b_fmri_tool_fired(self):
        FMRITool(physiodata=self.physiodata).edit_traits()

    
    traits_view = MEAPView(
        VGroup(
            VGroup(
                Item("file"),
                Item("outfile")
            ),
            VGroup(
                spring,
                Item("b_import",
                     show_label=False,
                     enabled_when="file.endswith('.acq') or file.endswith('.mat')"),
                Item("b_inspect", enabled_when="physiodata is not None"),
                Item("b_subject_info", enabled_when="physiodata is not None"),
                Item("b_resp",enabled_when="physiodata is not None"),    
                Item("b_detect",enabled_when="physiodata is not None"),
                Item("b_custom_points",enabled_when="peaks_detected"),
                Item("b_moving_ens",enabled_when="global_ensemble_marked"),
                Item("b_register",enabled_when="peaks_detected"),
                Item("b_fmri_tool", enabled_when="physiodata.processed_respiration_data.size > 0"),
                Item("b_save",
                     enabled_when="outfile.endswith('.mea') or outfile.endswith('.mea.mat')"),
                Item("b_clear_mem"),
                spring,
                show_labels=False
            )
            )
    )
