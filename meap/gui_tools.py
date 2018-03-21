import logging
from meap import __version__
import os
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from traits.api import Str

# Are we bundled?
if getattr( sys, 'frozen', False ) :
    _ROOT = sys._MEIPASS
    # running in a bundle
else :
    _ROOT = os.path.abspath(os.path.dirname(__file__))
"""
Running MEAP without a graphics server is tricky.
Here we check if we can import traitsui. If not we 
make dummy classes so the rest of the library still
works.

"""
try:
    from traits.etsconfig.api import ETSConfig
    ETSConfig.toolkit = 'qt4'
    from traitsui.api import (View, HGroup, VGroup,
        Group, Item, Action, ObjectColumn, TableEditor,
        HSplit, VSplit, SetEditor, Handler, ExpressionColumn,
        EnumEditor, RangeEditor, Include, CheckListEditor,
        spring
        )
    from traitsui.menu import OKButton, CancelButton
    from traitsui.message import Message
    
    # Enable
    from enable.component_editor import ComponentEditor
    from enable.api import Component, ColorTrait, KeySpec
    from enable.tools.api import DragTool
        
    # Pyface
    from pyface.image_resource import ImageResource
    from pyface.api import SplashScreen
    from pyface.api import ProgressDialog
    
    # Chaco
    from chaco.api import  (
        DataLabel, ScatterPlot, ArrayDataSource, Plot, ArrayPlotData, 
        HPlotContainer, VPlotContainer, BaseTool, ScatterInspectorOverlay,
        jet, gist_rainbow, ColorBar, ColormappedSelectionOverlay,
        LinearMapper, marker_trait
        )
    from chaco.overlays.coordinate_line_overlay import CoordinateLineOverlay
    from chaco.tools.api import ScatterInspector
    from chaco.tools.api import PanTool, ZoomTool
    from chaco.tools.api import RangeSelection, RangeSelectionOverlay 
    from chaco.tools.line_inspector import LineInspector
    from chaco.tools.scatter_inspector import ScatterInspector
    from chaco.data_range_1d import DataRange1D
    from chaco.scatterplot import ScatterPlot
    from chaco.lineplot import LinePlot
    
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
    
    def fail(msg,interactive=False):
        logger.critical(msg)
           
except Exception, e:
    logger.info(e)
    
    # Variables
    icon=None
    meap_splash = None
    
    # Classes
    class TraitsUIDummy(object):
        def __init__(self,*args, **kwargs):
            pass
        def edit_traits(self,*args, **kwargs):
            pass
        
    class MEAPView(TraitsUIDummy):
        pass
    class View(TraitsUIDummy):
        pass
    class HGroup(TraitsUIDummy):
        pass
    class VGroup(TraitsUIDummy):
        pass
    class Group(TraitsUIDummy):
        pass
    class Item(TraitsUIDummy):
        pass
    class Action(TraitsUIDummy):
        pass
    class ObjectColumn(TraitsUIDummy):
        pass
    class TableEditor(TraitsUIDummy):
        pass
    class HSplit(TraitsUIDummy):
        pass
    class VSplit(TraitsUIDummy):
        pass
    class SetEditor(TraitsUIDummy):
        pass
    class Handler(TraitsUIDummy):
        pass
    class ExpressionColumn(TraitsUIDummy):
        pass
    class EnumEditor(TraitsUIDummy):
        pass
    class RangeEditor(TraitsUIDummy):
        pass
    class Include(TraitsUIDummy):
        pass
    class CheckListEditor(TraitsUIDummy):
        pass
    class spring(TraitsUIDummy):
        pass
    class OKButton(TraitsUIDummy):
        pass
    class CancelButton(TraitsUIDummy):
        pass
    class Message(TraitsUIDummy):
        pass
    class ComponentEditor(TraitsUIDummy):
        pass
    class Component(TraitsUIDummy):
        pass
    class ColorTrait(TraitsUIDummy):
        pass
    class KeySpec(TraitsUIDummy):
        pass
    class DragTool(TraitsUIDummy):
        pass
    
    # Pyface
    class ImageResource(TraitsUIDummy):
        pass
    class SplashScreen(TraitsUIDummy):
        pass
    class ProgressDialog(TraitsUIDummy):
        pass
    
    # Chaco
    class DataLabel(TraitsUIDummy):
        pass
    class ScatterPlot(TraitsUIDummy):
        pass
    class ArrayDataSource(TraitsUIDummy):
        pass
    class Plot(TraitsUIDummy):
        pass
    class ArrayPlotData(TraitsUIDummy):
        pass
    class HPlotContainer(TraitsUIDummy):
        pass
    class VPlotContainer(TraitsUIDummy):
        pass
    class BaseTool(TraitsUIDummy):
        pass
    class ScatterInspectorOverlay(TraitsUIDummy):
        pass
    class jet(TraitsUIDummy):
        pass
    class gist_rainbow(TraitsUIDummy):
        pass
    class ColorBar(TraitsUIDummy):
        pass
    class ColorMappedSelectionOverlay(TraitsUIDummy):
        pass
    class LinearMapper(TraitsUIDummy):
        pass
    class marker_trait(TraitsUIDummy):
        pass
    class CoordinateLineOverlay(TraitsUIDummy):
        pass
    class ScatterInspector(TraitsUIDummy):
        pass
    class PanTool(TraitsUIDummy):
        pass
    class ZoomTool(TraitsUIDummy):
        pass
    class RangeSelection(TraitsUIDummy):
        pass
    class RangeSelectionOverlay(TraitsUIDummy):
        pass
    class LineInspector(TraitsUIDummy):
        pass
    class ScatterInspector(TraitsUIDummy):
        pass
    class DataRange1D(TraitsUIDummy):
        pass
    class ScatterPlot(TraitsUIDummy):
        pass
    class LinePlot(TraitsUIDummy):
        pass
    
    
    class MEAPView(TraitsUIDummy):
        pass
    def messagebox(msg, *args, **kwargs):
        logger.debug(msg)
    def fail(msg,*args, **kwargs):
        logger.info(e)
    