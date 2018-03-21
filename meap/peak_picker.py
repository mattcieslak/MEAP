#!/usr/bin/env python


""" Defines the RangeSelection controller class.
"""
# Major library imports
from numpy import array

# Enthought library imports
from traits.api import Any, Array, Bool, Enum, Event, Float, Int, Instance, \
                         List, Property, Str, Trait, Tuple
from meap.gui_tools import KeySpec, ColorTrait, RangeSelection, RangeSelectionOverlay


class PeakPickingOverlay(RangeSelectionOverlay):
    border_color = ColorTrait("black")
    purpose_metadata_name = Str("selection_purpose")
    fill_color_ = Property(ColorTrait)
    def _get_fill_color_(self):
        ds = getattr(self.plot, self.axis)
        selection = ds.metadata.get(self.purpose_metadata_name, None)
        print "getting fill color", selection
        if selection is None:
            return "white"
        if selection == "add":
            return "lightskyblue"
        if selection == "delete":
            return "red"

class PeakPickingTool(RangeSelection):
    """ Selects a range along the index or value axis.

    The user right-click-drags to select a region, which stays selected until
    the user left-clicks to deselect.
    """

    left_button_selects = Bool(True)
    selection_purpose = Enum("add","delete","add_dne")
    purpose_metadata_name = Str("selection_purpose")
    done_selecting = Event
    
    def _selection_purpose_changed(self):
        plt = getattr(self.plot, self.axis)
        plt.metadata[self.purpose_metadata_name] = self.selection_purpose

    def normal_left_down(self, event):
        """ Handles the left mouse button being pressed when the tool is in
        the 'normal' state.

        If the tool allows the left mouse button to start a selection, then
        it does so.
        """
        self.selection_purpose = "add"
        return self.normal_down(event)

    def normal_middle_down(self, event):
        """ Handles the middle mouse button being pressed when the tool is in
        the 'normal' state.

        If the tool allows the middle mouse button to start a selection, then
        it does so.
        """
        self.selection_purpose = "add_dne"
        return self.normal_down(event)

    def normal_right_down(self, event):
        """ Handles the right mouse button being pressed when the tool is in
        the 'normal' state.

        Puts the tool into 'selecting' mode, changes the cursor to show that it
        is selecting, and starts defining the selection.

        """
        self.selection_purpose = "delete"
        return self.normal_down(event)
        
    def normal_down(self,event):
        pos = self._get_axis_coord(event)
        mapped_pos = self.mapper.map_data(pos)
        self.selection = (mapped_pos, mapped_pos)
        self._set_sizing_cursor(event)
        self._down_point = array([event.x, event.y])
        self.event_state = "selecting"
        self.selection_mode = "set"
        self.selecting_mouse_move(event)
        return

    #------------------------------------------------------------------------
    # Event handlers for the "selecting" event state
    #------------------------------------------------------------------------

    def selecting_mouse_move(self, event):
        """ Handles the mouse being moved when the tool is in the 'selecting'
        state.

        Expands the selection range at the appropriate end, based on the new
        mouse position.
        """
        if self.selection is not None:
            axis_index = self.axis_index
            low = self.plot.position[axis_index]
            high = low + self.plot.bounds[axis_index] - 1
            tmp = self._get_axis_coord(event)
            if tmp >= low and tmp <= high:
                new_edge = self.mapper.map_data(self._get_axis_coord(event))
                #new_edge = self._map_data(self._get_axis_coord(event))
                if self._drag_edge == "high":
                    low_val = self.selection[0]
                    if new_edge >= low_val:
                        self.selection = (low_val, new_edge)
                    else:
                        self.selection = (new_edge, low_val)
                        self._drag_edge = "low"
                else:
                    high_val = self.selection[1]
                    if new_edge <= high_val:
                        self.selection = (new_edge, high_val)
                    else:
                        self.selection = (high_val, new_edge)
                        self._drag_edge = "high"

                self.component.request_redraw()
            event.handled = True
        return

    def selecting_button_up(self, event):
        # Check to see if the selection region is bigger than the minimum
        event.window.set_pointer("arrow")

        end = self._get_axis_coord(event)

        if len(self._down_point) == 0:
            cancel_selection = False
        else:
            start = self._down_point[self.axis_index]
            self._down_point = []
            cancel_selection = self.minimum_selection > abs(start - end)

        if cancel_selection:
            self.deselect(event)
            event.handled = True
        else:
            self.event_state = "selected"

            # Fire the "completed" event
            self.selection_completed = self.selection
            event.handled = True
        self.done_selecting = True
        return
