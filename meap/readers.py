from traits.api import (HasTraits, Str, Array, Float, CFloat, CBool,
          Bool, Enum, Instance, on_trait_change,File,Property,
          Range,Int, CInt, List, Button, Dict)
from meap.gui_tools import ( View, Item, VGroup, HGroup, Group,
     RangeEditor, TableEditor, Handler, Include,HSplit, EnumEditor, HSplit, Action,
     CheckListEditor, ObjectColumn, fail, MEAPView )
from traitsui.menu import OKButton, CancelButton

import numpy as np
# importers
import bioread
from scipy.io.matlab import loadmat

from meap import DEFAULT_SAMPLING_RATE, SUPPORTED_SIGNALS
from meap.io import PhysioData
from meap.filters import downsample
import os
import re

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_N_MRI_TRIGGERS=5000
def get_mri_triggers(trig_arr,sampling_rate):
    # Check to see if the input array is a timeseries or is already 
    # in timestamp format
    if len(trig_arr) < MAX_N_MRI_TRIGGERS:
        return trig_arr
    tdiff = np.ediff1d(np.round(trig_arr),to_begin=0)
    if not len(np.unique(tdiff)) == 2:
        logger.warn("found multiple states for mri trigger")
    threshold = 2
    trig_indices = np.flatnonzero(tdiff > threshold)
    return trig_indices / float(sampling_rate)
    
class Channel(HasTraits):
    contains = Enum(["None"] + SUPPORTED_SIGNALS)
    name = Str
    decimate = Bool(False)
    sampling_rate = CInt
    
    def _decimate_default(self):
        if self.contains == "mri_trigger": return False
        return not self.sampling_rate == 1000

channel_table = TableEditor(
    columns =
    [ ObjectColumn(name="name",editable=False),
      ObjectColumn(name="contains",editable=True),
      ObjectColumn(name="sampling_rate",editable=False),
      ObjectColumn(name="decimate",editable=True)
    ],
    auto_size=True,
)

class Importer(HasTraits):
    path = File
    channels = List(Instance(Channel))
    mapping_txt = File()
    mapper = Dict()
    
    # --- Subject information
    subject_age = CFloat(20.)
    subject_gender = Enum("M","F")
    subject_weight = CFloat(135.,label="Weight (lbs)")
    subject_height_ft = CInt(5,label="Height (ft)",
                        desc="Subject's height in feet")
    subject_height_in = CInt(10,label = "Height (in)",
                        desc="Subject's height in inches")
    subject_electrode_distance_front = CFloat(32,
                        label="Impedance electrode distance (front)")
    subject_electrode_distance_back = CFloat(32,
                        label="Impedance electrode distance (back)")
    subject_electrode_distance_right = CFloat(32,
                        label="Impedance electrode distance (right)")
    subject_electrode_distance_left = CFloat(32,
                        label="Impedance electrode distance (left)")
    subject_in_mri = CBool(False)
    subject_control_base_impedance = CFloat(0.,label="Control Imprdance",
                        desc="If in MRI, store the z0 value from outside the MRI")
    subject_resp_max = CFloat(100.,label="Respiration circumference max (cm)")
    subject_resp_min = CFloat(0.,label="Respiration circumference min (cm)")
    
    def _channel_map_default(self):
        return ChannelMapper()
    
    def _mapping_txt_changed(self):
        if not os.path.exists(self.mapping_txt): return
        with open(self.mapping_txt) as f:
            lines = [l.strip() for l in f]
        self.mapper = {}
        for line in lines:
            strip = line.split("->")
            if not len(strip) == 2:
                logger.info("Unable to parse line: %s", line)
                continue
            map_from, map_to = [l.strip() for l in strip]
            if map_to in SUPPORTED_SIGNALS:
                self.mapper[map_from] = map_to
                logger.info("mapping %s to %s",map_from,map_to)
            else:
                logger.info("Unrecognized channel: %s", map_to)
        
    traits_view = MEAPView(
        Group(
            Item("channels", editor=channel_table),
            label="Specify channel contents",
            show_border=True,
            show_labels=False
            ),
         Group(
             Item("subject_age"), Item("subject_gender"),
             Item("subject_height_ft"), Item("subject_height_in"),
             Item("subject_weight"),
             Item("subject_electrode_distance_front"),
             Item("subject_electrode_distance_back"),
             Item("subject_electrode_distance_left"),
             Item("subject_electrode_distance_right"),
             Item("subject_resp_max"),Item("subject_resp_min"),
             Item("subject_in_mri"),Item('subject_control_base_impedance'),
             label="Participant Measurements"
             ),
        buttons=[OKButton, CancelButton],
        resizable=True,
        win_title="Import Data",
        )
    
    
    def guess_channel_contents(self):
        mapped = set([v for k,v in self.mapper.iteritems()])
        for chan in self.channels:
            if chan.name in self.mapper:
                chan.contains = self.mapper[chan.name]
                continue
            cname = chan.name.lower()
            if "magnitude" in cname and not "z0" in mapped:
                chan.contains = "z0"
            elif "ecg" in cname and not "ecg" in mapped:
                chan.contains = "ecg"
            elif "dz/dt" in cname or "derivative" in cname and \
                 not "dzdt" in mapped:
                chan.contains = "dzdt"
            elif "diastolic" in cname and not "diastolic" in mapped:
                chan.contains = "diastolic"
            elif "systolic" in cname and not "systolic" in mapped:
                chan.contains = "systolic"
            elif "blood pressure" in cname or "bp" in cname and \
                 not "bp" in mapped:
                chan.contains = "bp"
            elif "resp" in cname and not "respiration" in mapped:
                chan.contains = "respiration"
            elif "trigger" in cname and not "mri_trigger" in mapped:
                chan.contains = "mri_trigger"
            elif "stimulus" in cname and not "event" in mapped:
                chan.contains = "event"
            
                
    def subject_data(self):
        
        return dict(subject_age = self.subject_age,
            subject_gender = self.subject_gender,
            subject_weight = self.subject_weight,
            subject_height_ft = self.subject_height_ft, 
            subject_height_in = self.subject_height_in,
            subject_electrode_distance_front = self.subject_electrode_distance_front,  
            subject_electrode_distance_back = self.subject_electrode_distance_back ,
            subject_electrode_distance_left = self.subject_electrode_distance_left ,
            subject_electrode_distance_right = self.subject_electrode_distance_right ,
            subject_resp_max = self.subject_resp_max,
            subject_resp_min = self.subject_resp_min,
            subject_in_mri = self.subject_in_mri
            )
    
    def get_physiodata(self):
        raise NotImplementedError("Must be overwritten by subclass")
    
        
                
class AcqImporter(Importer):

    def __init__(self,**traits):
        super(AcqImporter,self).__init__(**traits)
        if not os.path.exists(self.path):
            fail("No such file: "+self.path)
        
        acqdata = bioread.read_file(self.path)
        self.channels = [
             Channel(name=chan.name,
                     sampling_rate = chan.samples_per_second,
                     decimate = chan.samples_per_second > DEFAULT_SAMPLING_RATE)
                         for chan in acqdata.channels ]
        self.guess_channel_contents()
        self.acqdata = acqdata
        
    def get_events(self):
        data = {"event_names":[]}
        
        # Loop over all the included channels
        for col,chan in enumerate(self.channels):
            contents = chan.contains
            if not contents == "event":
                continue
            fixed_name = "EVENT_" +re.sub(r"[- .^!/\/\(\)~`\"'#%&?:+=]+","_",chan.name)
            if chan.decimate:
                logger.info("Downsampling %s from %d Hz to %d Hz",
                            chan.name,chan.sampling_rate,DEFAULT_SAMPLING_RATE)
                orig_size = self.acqdata.channels[col].data[:-10].shape[0]
                downsampled = downsample(self.acqdata.channels[col].data[:-10],
                         chan.sampling_rate,DEFAULT_SAMPLING_RATE)
                logger.info("Downsampled from %d to %d samples", orig_size, downsampled.shape[0])
                data[fixed_name] = downsampled
                chan.sampling_rate=DEFAULT_SAMPLING_RATE
            else:
                logger.info("Using original sampling rate")
                data[fixed_name] = self.acqdata.channels[col].data[:-10]
            data['event_names'].append(fixed_name)
        data["event_sampling_rate"] = chan.sampling_rate
        data["event_included"] = True
        data["event_decimated"] = chan.decimate
        data["event_start_time"] = 0.
        data["event_sampling_rate_unit"] = "Hz"
        data["event_unit"] = self.acqdata.channels[col].units
        return data

    def get_physiodata(self,config=None):
        # Get a dict of the subject information
        data = self.subject_data()
        data["original_file"] = self.path
        data.update(self.get_events())
        
        # Loop over all the included channels
        for col,chan in enumerate(self.channels):
            contents = chan.contains
            if contents in ("None","event"):
                continue
            # Is there already a channel named this?
            if contents+"_included" in data:
                fail(contents + " appears multiple times in the data")
            # Downsample the signal if sampling_rate > 1 kHz
            if contents == "mri_trigger":
                data["mri_trigger_times"] = get_mri_triggers(
                                   self.acqdata.channels[col].data[:-10],
                                   chan.sampling_rate)
            else:
                if chan.decimate:
                    logger.info("Downsampling %s from %d Hz to %d Hz",
                                chan.name,chan.sampling_rate, DEFAULT_SAMPLING_RATE)
                    orig_size = self.acqdata.channels[col].data[:-10].shape[0]
                    downsampled = downsample(self.acqdata.channels[col].data[:-10],
                             chan.sampling_rate,DEFAULT_SAMPLING_RATE)
                    logger.info("Downsampled from %d to %d samples", orig_size, downsampled.shape[0])
                    data[contents+"_data"] = downsampled
                    chan.sampling_rate=DEFAULT_SAMPLING_RATE
                else:
                    logger.info("Using original sampling rate")
                    data[contents+"_data"] = self.acqdata.channels[col].data[:-10]
                    
            data[contents+"_sampling_rate"] = chan.sampling_rate
            data[contents+"_included"] = True
            data[contents+"_decimated"] = chan.decimate
            data[contents+"_start_time"] = 0.
            data[contents+"_channel_name"] = chan.name
            data[contents+"_sampling_rate_unit"] = "Hz"
            data[contents+"_unit"] = self.acqdata.channels[col].units
        if config is None:
            return PhysioData(**data)
        return PhysioData(config=config,**data)
    
        
class MatfileImporter(Importer):

    def __init__(self,**traits):
        super(MatfileImporter,self).__init__(**traits)
        m = loadmat(self.path)
        self.m = m
        fs = float(self.m['isi'].squeeze())
        sampling_unit = str(self.m['isi_units'].squeeze())
        if not sampling_unit == "ms":
            raise ValueError("Haven't seen this sampling unit before: " + sampling_unit)
        self.sampling_rate = ( 1 / fs ) * 1000 #msec/sec
        self.channels = [Channel(name=cname,sampling_rate=self.sampling_rate) for \
                         cname in m['labels']]
        self.guess_channel_contents()
        
    def get_events(self):
        data = {"event_names":[]}
        
        # Loop over all the included channels
        for col,chan in enumerate(self.channels):
            contents = chan.contains
            if not contents == "event":
                continue
            # Extract the array, except for the last 10 samples
            data_array = self.m["data"][:-10,col]
            fixed_name = "EVENT_" +re.sub(r"[- .^!/\/\(\)~`\"'#%&?:+=]+","_",chan.name)
            if chan.decimate:
                logger.info("Downsampling %s from %d Hz to %d Hz",
                            chan.name,chan.sampling_rate,DEFAULT_SAMPLING_RATE)
                orig_size = data_array.shape[0]
                downsampled = downsample(data_array,
                         chan.sampling_rate,DEFAULT_SAMPLING_RATE)
                logger.info("Downsampled from %d to %d samples", orig_size, downsampled.shape[0])
                data[fixed_name] = downsampled
                chan.sampling_rate=DEFAULT_SAMPLING_RATE
            else:
                logger.info("Using original sampling rate")
                data[fixed_name] = data_array
            data['event_names'].append(fixed_name)
        data["event_sampling_rate"] = chan.sampling_rate
        data["event_included"] = True
        data["event_decimated"] = chan.decimate
        data["event_start_time"] = 0.
        data["event_sampling_rate_unit"] = "Hz"
        data["event_unit"] = str(self.m['units'][col])
        return data
        
    def get_physiodata(self, config=None):
        # Get a dict of the subject information
        data = self.subject_data()
        data["original_file"] = self.path
        data.update(self.get_events())
        
        # Loop over all the included channels
        for col,chan in enumerate(self.channels):
            contents = chan.contains
            if contents == "None":
                continue
            # Is there already a channel named this?
            if contents+"_included" in data:
                fail(contents + " appears multiple times in the data")
                
            # Extract the array, except for the last 10 samples
            if "data" in self.m:
                data_array = self.m["data"][:-10,col]
            elif "physioDat" in self.m:
                data_array = self.m["physioDat"][:-10,col]
            else:
                raise ValueError("Unrecognized matfile format")
            
            # Downsample the signal if sampling_rate > 1 kHz
            if contents == "mri_trigger":
                data["mri_trigger_times"] = get_mri_triggers(data_array,
                                                             chan.sampling_rate)
            else:
                # Downsample the signal if sampling_rate > 1 kHz
                if chan.decimate:
                    logger.info("Downsampling %s from %d Hz to %d Hz",
                                chan.name, chan.sampling_rate, DEFAULT_SAMPLING_RATE)
                    orig_size = data_array.shape[0]
                    downsampled = downsample(data_array, chan.sampling_rate, DEFAULT_SAMPLING_RATE)
                    logger.info("Downsampled from %d to %d samples", orig_size, downsampled.shape[0])
                    data[contents+"_data"] = downsampled
                    chan.sampling_rate=DEFAULT_SAMPLING_RATE
                else:
                    logger.info("Using original sampling rate")
                    data[contents+"_data"] = np.copy(data_array)
                
            data[contents+"_sampling_rate"] = chan.sampling_rate
            data[contents+"_included"] = True
            data[contents+"_decimated"] = chan.decimate
            data[contents+"_start_time"] = 0.
            data[contents+"_channel_name"] = chan.name
            data[contents+"_sampling_rate_unit"] = "Hz"
            data[contents+"_unit"] = str(self.m['units'][col])
            
        # Clear the original matfile from memory.
        del self.m
        
        # Return PhysioData
        if config is None:
            return PhysioData(**data)
        return PhysioData(config=config,**data)
