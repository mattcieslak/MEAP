#!/usr/bin/env python
# Major library imports
import numpy as np
# Enthought library imports
from meap.gui_tools import DragTool
from traits.api import ( Instance, Int, Tuple, Event,
      Any,  Array, Enum, Float, CInt, CFloat, Str)
# Chaco imports
from meap.gui_tools import BaseTool, CoordinateLineOverlay

from logging import getLogger
logger = getLogger(__name__)

class PointDraggingTool(DragTool):
    """
    The PointDraggingTool is only added as a component editor to
    the scatter plot that contains 
    """
    #points = [ "p", "q", "r", "s", "t",
    #           "c", "x", "o", #"y",
    #           "systole", "diastole","b"
    #         ]
    #point_arr = Instance(Array)

    #beat = Instance(HeartBeat)

    # The pixel distance from a point that the cursor is still considered
    # to be 'on' the point
    threshold = Int(8)
    # The index of the point being dragged
    _drag_index = CInt(-1)
    # The original dataspace values of the index and value datasources
    # corresponding to _drag_index
    _orig_value = Tuple
    # Event for when the x value of the point is changed
    point_changed = Event
    # which point is being dragged?
    currently_dragging_point = Str()
                        #Enum([ "p", "q", "r", "s", "t",
                        #"b", "c", "x", "o", #"y",
                        #"systole", "diastole"
                        #])
    current_y = Array
    current_x = Array
    current_time = Float
    current_value = Float
    current_index = CInt
    
    point_edited = Event
    
    
    def is_draggable(self, x, y):
        # Check to see if (x,y) are over one of the points in self.component
        if self._lookup_point(x, y) is not None:
            return True
        else:
            return False
        
    def normal_mouse_move(self, event):
        plot = self.component

        ndx = plot.map_index((event.x, event.y), self.threshold)
            
        if ndx is None:
            if plot.index.metadata.has_key('selections'):
                del plot.index.metadata['selections']
        else:
            plot.index.metadata['selections'] = [ndx]
        plot.invalidate_draw()
        plot.request_redraw()
        
    def drag_start(self, event):
        """ 
        The user has clicked around a point
        """
        print "Event", event
        plot = self.component
        self.component.name = "dragging_component"
        ndx = plot.map_index((event.x, event.y), self.threshold)
        if ndx is None:
            return
        self._drag_index = ndx
        self._orig_value = (plot.index.get_data()[ndx], plot.value.get_data()[ndx])
        # which point was selected        
        point = self.beat.points[self._drag_index]
        self.currently_dragging_point = point.name
        logger.info( "editing %s point", self.currently_dragging_point)
        self.current_y = getattr(self.beat, "plt_%s" % point.applies_to)
        self.current_x  = getattr(self.beat, "%s_time" % point.applies_to)

    def dragging(self, event):
        plot = self.component
        data_x, data_y = plot.map_data((event.x, event.y))
        # If the user drags the point past the bounds of the timeseries
        # Stick the point at the last measurement.
        if not len(self.current_x): return
        if data_x > self.current_x[-1]:
            plot.index._data[self._drag_index] = self.current_x[-1]
            plot.value._data[self._drag_index] = self.current_y[-1]
        # Otherwise, force the y value to stick to the data for the appropriate
        # signal
        else:
            self.current_time = data_x
            self.current_index = (np.abs(self.current_x - data_x)).argmin()
            self.current_value = self.current_y[self.current_index]
            plot.index._data[self._drag_index] = data_x
            plot.value._data[self._drag_index] = self.current_value
            self.point_changed = True
        plot.index.data_changed = True
        plot.value.data_changed = True
        plot.request_redraw()
        
    def drag_cancel(self, event):
        plot = self.component
        plot.index._data[self._drag_index] = self._orig_value[0]
        plot.value._data[self._drag_index] = self._orig_value[1]
        plot.index.data_changed = True
        plot.value.data_changed = True
        plot.request_redraw()
        
    def drag_end(self, event):
        plot = self.component
        if plot.index.metadata.has_key('selections'):
            del plot.index.metadata['selections']
        plot.invalidate_draw()
        plot.request_redraw()
        self.point_edited = True
        
    def _lookup_point(self, x, y):
        """ Finds the point closest to a screen point if it is within self.threshold

        Parameters
        ==========
        x : float
            screen x-coordinate
        y : float
            screen y-coordinate
        Returns
        =======
        (screen_x, screen_y, distance) of datapoint nearest to the input *(x,y)*.
        If no data points are within *self.threshold* of *(x,y)*, returns None.
        """
        if hasattr(self.component, 'get_closest_point'):
            # This is on BaseXYPlots
            return self.component.get_closest_point((x,y), threshold=self.threshold)
        return None


class BTool(BaseTool, CoordinateLineOverlay):
    '''This tool uses LinePlot.hittest() to get the closest point
    on the line to the mouse position and to draw it to the screen.
    Also implements an Overlay in order to draw the point.
    '''
    # A reference to the lineplot the tool acts on
    line_plot = Any()

    # Time trait that can be set externally
    time = Float()
    selected_time = Float()
    
    # Whether to draw the overlay
    visible=True

    # The point to draw on the plot, or None if no point
    pt = Any()

    # How many pixels away we may be from the line in order to do 
    threshold = Int(40)

    def _time_changed(self):
        #logger.info("Time Changed")
        self.request_redraw()

    def normal_left_down(self,event):
        # This will trigger the listener on self.beat.
        self.selected_time = self.time

    def normal_mouse_move(self, event):
        # Compute the nearest point and draw it whenever the mouse moves
        x,y = event.x, event.y

        if self.line_plot.orientation == "h":
            x,y = self.component.map_data((x,y))
        else:
            x,y = self.component.map_data((y,x))

        x,y = self.line_plot.map_screen((x,y))
        self.pt = self.line_plot.hittest((x,y), threshold=self.threshold)
        if self.pt is not None:
            self.time = self.pt[0]

    def overlay(self, plot, gc, view_bounds=None, mode="normal"):
        # If we have a point, draw it to the screen as a small square
        if self.pt is not None:
            x,y = plot.map_screen(self.pt)
            gc.draw_rect((int(x)-2, int(y)-2, 4, 4))
            self._draw_vertical_line(gc,x)
            self.pt = None
        else:
            t,_ = plot.map_screen((self.time,0))
            self._draw_vertical_line(gc,t)

class DBTool(BTool):
    x_selected_time = Float()
    def normal_left_down(self,event):
        # This will trigger the listener on self.beat.
        self.x_selected_time = self.time
    

from meap.gui_tools import ColorTrait

class BMarker(CoordinateLineOverlay):
    """
    Static marker for the b-point
    """
    time = Float()
    
    # Whether to draw the overlay
    visible=True

    def _time_changed(self):
        #logger.info("time changed in BMarker")
        self.request_redraw()

    def overlay(self, plot, gc, view_bounds=None, mode="normal"):
        gc.set_stroke_color(self.color_)
        gc.set_line_dash(self.line_style_)        
        t,_ = plot.map_screen((self.time,0))
        self._draw_vertical_line(gc,t)        