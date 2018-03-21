from traits.api import (HasTraits, Instance, on_trait_change,Property, 
                        Range, Float,DelegatesTo)
from meap.gui_tools import (VGroup, Item, Group, Handler, VPlotContainer,
                           DataRange1D, MEAPView)
from traitsui.menu import OKButton, CancelButton
import numpy as np
from meap.io import PhysioData
from meap.timeseries import TimeSeries
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataPlotHandler(Handler):
    def close(self,info,is_ok):
        """
        This is necessary to take all the censoring regions from the 
        different signals and save them to the physiodata object
        """
        # dont save if the user hit cancel
        if not is_ok: 
            logger.info("User cancelled, not saving censored_regions")
            return 1

        pd = info.object.physiodata
        censor_sources = []
        censored_intervals = []
        for signal in pd.contents:
            if signal.startswith("resp_corr"):
                logger.info("not plotting %s", signal)                
                continue
            ts = getattr(info.object, signal + "_ts")
            logger.info("Saving %d regions from %s",
                        len(ts.censored_regions),signal)
            for cens_reg in ts.censored_regions:
                censored_intervals.append(
                        (cens_reg.start_time,cens_reg.end_time))
                censor_sources.append(signal)
        pd.censoring_sources = censor_sources
        pd.censored_intervals = np.array(censored_intervals)
        return True

class DataPlot(HasTraits):
    physiodata = Instance(PhysioData)
    qrs_source_signal = DelegatesTo("physiodata")

    # These may or may not be filled
    ecg_ts = Instance(TimeSeries)
    ecg2_ts = Instance(TimeSeries)
    z0_ts = Instance(TimeSeries)
    dzdt_ts = Instance(TimeSeries)
    respiration_ts = Instance(TimeSeries)
    systolic_ts = Instance(TimeSeries)
    diastolic_ts = Instance(TimeSeries)
    bp_ts = Instance(TimeSeries)
    doppler_ts = Instance(TimeSeries)
    
    # Plotting traits
    start_time = Range(0,10000.0, initial=0.0)
    window_size = Range(1.0,300.0,initial=30.0)
    end_time = Property(Float, depends_on=["start_time","window_size"])
    raw_plots = Instance(VPlotContainer,transient=True)
    # Buttons for censoring a timeseries
    index_range = Instance(DataRange1D,transient=True)

    def __init__(self,**traits):
        """
        Represents the data collected by AcqKnowledge
        """
        super(DataPlot,self).__init__(**traits)
        # Create the TimeSeries objects:
        for signal in self.physiodata.contents:
            if signal.startswith("resp_corr"):
                logger.info("not plotting %s", signal)                
                continue
            setattr(self, signal+"_ts", 
                    TimeSeries(physiodata=self.physiodata,contains=signal))
            self.index_range = getattr(self, signal+"_ts").plot.index_range
            
        for signal in self.physiodata.contents:
            if signal.startswith("resp_corr"):
                logger.info("not plotting %s", signal)                
                continue
            getattr(self, signal+"_ts").plot.index_range = self.index_range 

    @on_trait_change("window_size,start_time")
    def update_plot_range(self):
        self.index_range.high = self.end_time
        self.index_range.low = self.start_time

    def _get_end_time(self):
        return self.start_time + self.window_size

    def default_traits_view(self):
        plots = [ Item(sig+"_ts", style="custom", show_label=False) for sig in \
                        sorted(self.physiodata.contents) if not sig.startswith("resp_corr")]
        if "ecg2" in self.physiodata.contents:
            widgets = VGroup(
                    Item("qrs_source_signal",label="ECG to use"),
                    Item("start_time"),
                    Item("window_size"),
                    )
        else:
            widgets = VGroup(
                    Item("start_time"),
                    Item("window_size"),
                    )
        return MEAPView(
          VGroup(
            VGroup(*plots),
            Group(
                widgets,
                orientation="horizontal"
                )
            ),
            width=1000,height=600,resizable=True,
            win_title="Aqcknowledge Data",
            buttons = [OKButton,CancelButton],
            handler=DataPlotHandler()
            )
