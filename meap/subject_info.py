#!/usr/bin/env python
from meap import  MEAPView
from meap.io import PhysioData
from traits.api import (HasTraits, DelegatesTo, Instance)
from meap.traitsui import Item, VGroup
from traitsui.menu import OKButton


class SubjectInfo(HasTraits):
    physiodata = Instance(PhysioData)
    subject_age =DelegatesTo("physiodata")
    subject_gender =DelegatesTo("physiodata")
    subject_weight =DelegatesTo("physiodata")
    subject_height_ft = DelegatesTo("physiodata")
    subject_height_in =DelegatesTo("physiodata")
    subject_electrode_distance_front =DelegatesTo("physiodata")
    subject_electrode_distance_back = DelegatesTo("physiodata")
    subject_electrode_distance_right = DelegatesTo("physiodata")
    subject_electrode_distance_left = DelegatesTo("physiodata")
    subject_resp_max = DelegatesTo("physiodata")
    subject_resp_min = DelegatesTo("physiodata")
    subject_in_mri = DelegatesTo("physiodata")
    subject_control_base_impedance = DelegatesTo("physiodata")

    traits_view = MEAPView(
         VGroup(
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
        buttons=[OKButton],
        resizable=True,
        win_title="Subject Info",
        )



