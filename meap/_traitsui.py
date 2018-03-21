import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
has_ui = True

try:
    from traitsui.api import View
except Exception, e:
    logger.info(e)
    has_ui = False
    

"""
Running MEAP without a graphics server is tricky.
Here we check if we can import traitsui. If not we 
make dummy classes so the rest of the library still
works.

"""
if has_ui:
    from traitsui.message import Message
    from traitsui.api import ( View, HGroup, VGroup,
        Group, Item, Action, ObjectColumn, TableEditor,
        HSplit, VSplit, SetEditor, Handler, ExpressionColumn,
        EnumEditor, RangeEditor, Include, CheckListEditor,
        )
    from traitsui.menu import OKButton, CancelButton
    
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
else:
    icon=None
    meap_splash = None
    def MEAPView(object):
        def __init__(self,*args, **kwargs):
            pass
        def edit_traits(self,*args, **kwargs):
            pass

if has_ui:
    from meap._traitsui import Action
    def messagebox(msg,title="Message",buttons=["OK"]):
        m = Message(message=msg)
        ui = m.edit_traits(
             view = MEAPView(
                   ["message~", "|<>"], title=title,
                   buttons=buttons,kind="modal")
        )
else:
    class Action(object):
        def __init__(self, *args, **kwargs):
            pass
    def messagebox(msg, title="Message", buttons=["OK"]):
        print msg



def messagebox(msg,title="Message",buttons=["OK"]):
    m = Message(message=msg)
    ui = m.edit_traits(
         view = MEAPView(
                        ["message~", "|<>"], title=title,buttons=buttons,kind="modal"
         )
    )
