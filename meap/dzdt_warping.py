#!/usr/bin/env python
from traits.api import (HasTraits,  Array,  File, cached_property,
          Bool, Enum, Instance, on_trait_change, Property,
          DelegatesTo, Int, Button, List, Set, Float )
import os
# Needed for Tabular adapter
from traitsui.api import Item,HGroup,VGroup, HSplit
from traitsui.menu import OKButton, CancelButton
from enable.component_editor import ComponentEditor
from chaco.api import (Plot, ArrayPlotData, HPlotContainer,jet)
import numpy as np
eps = np.finfo(np.float64).eps
from pyface.api import ProgressDialog

from meap.beat_train import MEABeatTrain
from meap.meap_timeseries import MEAPTimeseries
from meap.beat import GlobalEnsembleAveragedHeartBeat
import time
from meap import MEAPView, ParentButton, messagebox
from meap.io import PhysioData

import logging
logger = logging.getLogger(__name__)

from srvf_register import RegistrationProblem
from srvf_register.dynamic_programming_q2 import dp
from srvf_register.SqrtMeanInverse import SqrtMeanInverse

from scipy.integrate import trapz
from scipy.interpolate import UnivariateSpline


class GroupRegisterDZDT(HasTraits):
    physiodata = Instance(PhysioData)

    global_ensemble = Instance(GlobalEnsembleAveragedHeartBeat)
    srvf_lambda = DelegatesTo("physiodata")
    srvf_max_karcher_iterations = DelegatesTo("physiodata")
    srvf_update_min = DelegatesTo("physiodata")
    srvf_karcher_mean_subset_size = DelegatesTo("physiodata")
    dzdt_srvf_karcher_mean = DelegatesTo("physiodata")
    dzdt_karcher_mean = DelegatesTo("physiodata")
    dzdt_warping_functions = DelegatesTo("physiodata")
    srvf_use_moving_ensembled = DelegatesTo("physiodata")
    bspline_before_warping = DelegatesTo("physiodata")
    original_functions = Array

    # Holds indices of beats used to calculate karcher mean
    dzdt_karcher_mean_inputs = DelegatesTo("physiodata")
    dzdt_karcher_mean_over_iterations = DelegatesTo("physiodata")
    dzdt_num_inputs_to_group_warping = DelegatesTo("physiodata")
    srvf_iteration_distances = DelegatesTo("physiodata")
    srvf_iteration_energy = DelegatesTo("physiodata")
    karcher_mean_calculated = Bool(False)
    all_beats_registered = Bool(False)
    srvf_t_min = DelegatesTo("physiodata")
    srvf_t_max = DelegatesTo("physiodata")

    # For selecting points on the Karcher Mean
    btool_t = Float(0.)
    btool_t_selection = Float(0.)
    xtool_t = Float(0.)
    xtool_t_selection = Float(0.)
    # graphics items
    karcher_plot = Instance(Plot, transient=True)
    registration_plot = Instance(HPlotContainer, transient=True)
    karcher_plotdata = Instance(ArrayPlotData, transient=True)
    currently_editing = Enum("b", "x")
    t_offset = Float()

    # Buttons
    b_calculate_karcher_mean = Button(label="Calculate Karcher Mean")
    b_align_all_beats = Button(label="Warp all")
    b_update_b_point = Button(label="Update B-Point")
    b_update_x_point = Button(label="Update X-Point")

    interactive = Bool(False)

    def __init__(self,**traits):
        super(GroupRegisterDZDT,self).__init__(**traits)

        # Set the initial path to whatever's in the physiodata file
        logger.info("Initializing dZdt registration")

        # Is there already a Karcher mean in the physiodata?
        self.karcher_mean_calculated = self.dzdt_srvf_karcher_mean.size > 0 \
                                and self.dzdt_karcher_mean.size > 0

        # offset from t=0 at R peak
        self.t_offset = self.physiodata.dzdt_pre_peak
        self.original_functions = self.physiodata.mea_dzdt_matrix[
                                    : , self.srvf_t_min:self.srvf_t_max] if \
                self.srvf_use_moving_ensembled else self.physiodata.dzdt_matrix[
                                    : , self.srvf_t_min:self.srvf_t_max]
        self.sample_times = np.arange(self.srvf_t_max - self.srvf_t_min,
                                        dtype=np.float)

        self.n_functions = self.original_functions.shape[0]
        self.n_samples = self.original_functions.shape[1]
        self.karcher_mean_time = self.sample_times + self.srvf_t_min \
                                    - self.physiodata.dzdt_pre_peak

        if self.bspline_before_warping:
            logger.info("Smoothing inputs with B-Splines")
            self.original_functions = np.row_stack([ UnivariateSpline(
                self.sample_times, func, s=0.05)(self.sample_times) \
                for func in self.original_functions]
            )

        self._init_warps()

        self.select_new_samples()

    def _init_warps(self):
        if self.dzdt_warping_functions.shape[0] == \
                                self.original_functions.shape[0]:
            self.all_beats_registered = True

    def _global_ensemble_default(self):
        return GlobalEnsembleAveragedHeartBeat(physiodata=self.physiodata)

    def _karcher_plot_default(self):
        """
        Instead of defining these in __init__, only
        construct the plots when a ui is requested
        """
        self.interactive = True
        unreg_mean = self.global_ensemble.dzdt_signal
        # Temporarily fill in the karcher mean
        if not self.karcher_mean_calculated:
            karcher_mean = unreg_mean[self.srvf_t_min:self.srvf_t_max]
        else:
            karcher_mean = self.dzdt_karcher_mean

        self.karcher_plotdata = ArrayPlotData(
            time = self.global_ensemble.dzdt_time,
            karcher_time = self.karcher_mean_time,
            unregistered_mean = self.global_ensemble.dzdt_signal,
            karcher_mean = karcher_mean
        )
        karcher_plot = Plot(self.karcher_plotdata)
        karcher_plot.plot(("time","unregistered_mean"),
                    line_width=1,color="lightblue")
        karcher_plot.plot(("karcher_time","karcher_mean"),
                    line_width=3,color="blue")
        karcher_plot.title = "Means"
        return karcher_plot

    def _registration_plot_default(self):
        """
        Instead of defining these in __init__, only
        construct the plots when a ui is requested
        """
        unreg_mean = self.global_ensemble.dzdt_signal
        # Temporarily fill in the karcher mean
        if not self.karcher_mean_calculated:
            karcher_mean = unreg_mean[self.srvf_t_min:self.srvf_t_max]
        else:
            karcher_mean = self.dzdt_karcher_mean

        self.single_registration_plotdata = ArrayPlotData(
            karcher_time = self.karcher_mean_time,
            karcher_mean = karcher_mean,
            registered_func = karcher_mean
        )

        self.single_plot = Plot(self.single_registration_plotdata)
        self.single_plot.plot(("karcher_time","karcher_mean"),
                    line_width=2,color="blue")
        self.single_plot.plot(("karcher_time","registered_func"),
                    line_width=2,color="maroon")


        if self.all_beats_registered:
            self._forward_warp_beats()
            image_data = self.registered_functions
        else:
            image_data = self.original_functions

        # Create a plot of all the functions registered or not
        self.all_registration_plotdata = ArrayPlotData(
            image_data = image_data
        )
        self.all_plot = Plot(self.all_registration_plotdata)
        self.all_plot.img_plot("image_data", colormap = jet)

        return HPlotContainer(self.single_plot, self.all_plot)


    def _forward_warp_beats(self):
        """ Create registered beats to plot, since they're not stored """
        pass
        logger.info("Applying warps to functions for plotting")
        # Re-create gam
        gam = self.dzdt_warping_functions - self.srvf_t_min
        gam = gam / (self.srvf_t_max - self.srvf_t_min)

        aligned_functions = np.zeros_like(self.original_functions)
        t = self.sample_times
        for k in range(self.n_functions):
            aligned_functions[k] = np.interp((t[-1] - t[0])*gam[k] + t[0],
                                             t, self.original_functions[k])

        self.registered_functions = aligned_functions


    @on_trait_change("dzdt_num_inputs_to_group_warping")
    def select_new_samples(self):
        nbeats = self.original_functions.shape[0]
        nsamps = min(self.dzdt_num_inputs_to_group_warping, nbeats)
        self.dzdt_karcher_mean_inputs = np.random.choice(nbeats,size=nsamps,replace=False)

    @on_trait_change(
       ("physiodata.srvf_lambda, physiodata.srvf_karcher_iterations, "
        "physiodata.srvf_update_min, "
        "physiodata.srvf_karcher_mean_subset_size"
    ))
    def params_edited(self):
        self.dirty = True
        self.karcher_mean_calculated = False

    def _b_calculate_karcher_mean_fired(self):
        self.calculate_karcher_mean()

    def calculate_karcher_mean(self):
        reg_prob = RegistrationProblem(
            self.original_functions[self.dzdt_karcher_mean_inputs].T,
            sample_times=self.sample_times,
            max_karcher_iterations = self.srvf_max_karcher_iterations,
            lambda_value = self.srvf_lambda,
            update_min = self.srvf_update_min
        )
        reg_prob.run_registration_parallel()
        reg_problem = reg_prob
        self.dzdt_karcher_mean = reg_problem.function_karcher_mean
        self.dzdt_srvf_karcher_mean = reg_problem.srvf_karcher_mean
        self.karcher_mean_calculated = True

        # Update plots if this is interactive
        if self.interactive:
            self.karcher_plotdata.set_data("karcher_mean",
                                            self.dzdt_karcher_mean)
            self.karcher_plot.request_redraw()
        self.rp = reg_problem

    def _b_align_all_beats_fired(self):
        self.align_all_beats()

    def align_all_beats(self):
        if not self.karcher_mean_calculated:
            logger.warn("Calculate Karcher mean first")
            return
        logger.info("Aligning all beats to the Karcher Mean")
        if self.interactive:
            progress = ProgressDialog(title="ICG Warping", min=0,
                    max = len(self.physiodata.peak_times), show_time=True,
                    message="Warping dZ/dt to Karcher Mean...")
            progress.open()

        template_func = self.dzdt_srvf_karcher_mean
        normed_template_func = template_func / np.linalg.norm(template_func)
        fy, fx = np.gradient(self.original_functions.T, 1, 1)
        srvf_functions = (fy / np.sqrt(np.abs(fy) + eps)).T

        gam = np.zeros(self.original_functions.shape, dtype=np.float)
        aligned_functions = np.zeros_like(self.original_functions)

        t = self.sample_times
        for k in range(self.n_functions):
            q_c = srvf_functions[k] / np.linalg.norm(srvf_functions[k])
            G,T = dp(normed_template_func, t, q_c, t, t, t, self.srvf_lambda)
            gam0 = np.interp(self.sample_times, T, G)
            gam[k] = (gam0-gam0[0])/(gam0[-1]-gam0[0])  # change scale
            aligned_functions[k] = np.interp((t[-1] - t[0])*gam[k] + t[0],
                                             t, self.original_functions[k])

            if self.interactive:

                # Update the image plot
                img_data = self.all_registration_plotdata.get_data("image_data")
                img_data[k] = aligned_functions[k]
                self.all_registration_plotdata.set_data("image_data", img_data)

                # Update the registration plot
                self.single_registration_plotdata.set_data("registered_func",
                                                        aligned_functions[k])

                (cont,skip) = progress.update(k)

        self.registered_functions = aligned_functions

        self.dzdt_warping_functions = gam * (
                                        self.srvf_t_max - self.srvf_t_min) + \
                                        self.srvf_t_min

        if self.interactive:
            progress.update(k+1)

    mean_widgets =VGroup(
            Item("karcher_plot",editor=ComponentEditor(),
                 show_label=False),
            Item("registration_plot", editor=ComponentEditor(),
                show_label=False)
            Item("srvf_lambda",label="Lambda Value"),
            Item("dzdt_num_inputs_to_group_warping",
                    label="Template uses N beats"),
            Item("srvf_max_karcher_iterations", label="Max Iterations"),
            HGroup(
                Item("b_calculate_karcher_mean", label="Step 1:",
                    enabled_when="dirty"),
                Item("b_align_all_beats", label="Step 2:",
                    enabled_when="karcher_mean_calculated")
            ),
            HGroup(
                Item("b_update_b_point", show_label=False,
                    enabled_when="karcher_mean_calculated"),
                Item("b_update_x_point", show_label=False,
                    enabled_when="karcher_mean_calculated")
            )
    )

    traits_view = MEAPView(
        HSplit(
            mean_widgets
        ),
        resizable=True,
        win_title="ICG Warping Tools",
        width=800, height=700,
        buttons = [ParentButton,OKButton,CancelButton]
    )
