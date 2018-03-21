#!/usr/bin/env python
from traits.api import (HasTraits, Str, Array, Float,CFloat, Dict,
          Bool, Enum, Instance, on_trait_change,File,Property,
          Range, DelegatesTo, Int, Button, List, Color,Set )
# Needed for Tabular adapter
from meap.gui_tools import (Group, HGroup, VGroup, Item, TableEditor,
        ObjectColumn, MEAPView)
#from traitsui.table_adapter import TableAdapter
from meap.io import PhysioData
from meap.pipeline import MEAPPipeline

import logging
logger = logging.getLogger(__name__)

# Stuff for excel
import xlrd
import os


class PhysioFilePreprocessor(HasTraits):
    # Holds the data from
    physiodata = Instance(PhysioData)
    interactive = Bool(True)
    pipeline = Instance(MEAPPipeline)
    file = DelegatesTo("pipeline")
    outfile=DelegatesTo("pipeline")
    specs = Dict
    importer_kwargs = DelegatesTo("pipeline")
    mea_saved = Property(Bool,
            depends_on="file,outfile")

    def _outfile_default(self):
        if self.file.endswith(".acq"):
            return self.file[:-4] + ".mea"
        return self.file+".mea"

    def __init__(self,**traits):
        super(PhysioFilePreprocessor,self).__init__(**traits)

        # Some of the data contained in the excel sheet should get
        # attached to the physiodata file once it is loaded. These
        # go into the physiodata_kwargs attribute
        pd_traits = PhysioData().editable_traits()
        for spec,spec_value in self.specs.iteritems():
            if "subject_" + spec in pd_traits:
                self.importer_kwargs["subject_" + spec] = spec_value

    def _pipeline_default(self):
        pipeline = MEAPPipeline()
        return pipeline

    def _get_mea_saved(self):
        if self.file.endswith("mea"):
            return True
        return os.path.exists(self.pipeline.outfile) or \
               os.path.exists(self.pipeline.outfile + ".mat")

    @on_trait_change("pipeline.saved")
    def pipeline_saved(self):
        """A dumb hack"""
        old_outfile = self.outfile
        self.outfile =  "asdf"
        self.outfile = old_outfile

    pipeline_view = MEAPView(
                      Group(
                        Item("pipeline",style="custom"),
                        show_labels=False
                      )
                    )

class PhysioFileColumn(ObjectColumn):
    def get_cell_color(self,object):
        #if not object.pipeline is None: return "maroon"
        if object.mea_saved: return "light blue"
        return

pipeline_table = TableEditor(
    columns = [ PhysioFileColumn(name="pipeline.file",width=1.0, editable=False,label="Input Physio File") ],
    auto_size  = True,
    row_factory=PhysioFilePreprocessor,
    edit_view="pipeline_view"
    )


class PreprocessingPipeline(HasTraits):
    # Holds the set of multiple acq files
    group_excel_file = File
    # Which columns contain numeric data?
    # Holds the unique eveny types for this experiment
    physio_files = List(Instance(PhysioFilePreprocessor))
    # Interactively process the data?
    interactive = Bool(True)
    header = List

    @on_trait_change("group_excel_file")
    def open_xls(self,fpath=""):
        if not fpath:
            fpath = self.group_excel_file
        logger.info("Loading " + fpath)
        wb = xlrd.open_workbook(fpath)
        sheet = wb.sheet_by_index(0)
        header = [ str(item.value) for item in sheet.row(0) ]

        for rnum in range(1,sheet.nrows):
            rw = sheet.row(rnum)
            if not len(rw) == len(header):
                #raise ValueError("not enough information in row %d" % (rnum + 1))
                continue
            #convert to numbers and strings
            specs = {}
            for item, hdr in zip(rw,header):
                if hdr == "in_mri":
                    specs[hdr] = bool(item.value)
                else:
                    try:
                        specs[hdr] = float(item.value)
                    except Exception:
                        specs[hdr] = str(item.value)

            self.physio_files.append(
                PhysioFilePreprocessor(
                    interactive=self.interactive,
                    file = specs['file'],
                    outfile = specs.get('outfile',''),
                    specs=specs
                    )
                )
        self.header = header


    traits_view = MEAPView(
        VGroup(
            VGroup("group_excel_file",
            show_border=True, label="Excel Files"),
            HGroup(
                Item("physio_files",editor=pipeline_table,
                 show_label=False)),
            ),
        resizable=True,
        width=300, height=500
    )
