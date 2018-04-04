#!/usr/bin/env python
from traits.api import (HasTraits,  Array,  File, cached_property,
          Bool, Enum, Instance, on_trait_change, Property,
          DelegatesTo, Int, Button, List, Set, Float )
import os
# Needed for Tabular adapter
from meap.gui_tools import Item,HGroup,VGroup, HSplit, ProgressDialog
from meap.gui_tools import (ComponentEditor, Plot, ArrayPlotData, 
                           VPlotContainer, HPlotContainer,jet, fail,
                           OKButton, CancelButton)
import numpy as np
eps = np.finfo(np.float64).eps

from meap.beat import GlobalEnsembleAveragedHeartBeat
import time
from meap.gui_tools import MEAPView, messagebox
from meap.io import PhysioData

import logging
logger = logging.getLogger(__name__)

from srvf_register import RegistrationProblem
from srvf_register.dynamic_programming_q2 import dp

from scipy.interpolate import UnivariateSpline

from meap.point_marker2 import BTool, BMarker

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_samples

from scipy.cluster.hierarchy import complete, dendrogram
from scipy.cluster.hierarchy import fcluster


def fisher_rao_dist(psi1, psi2):
    """ Equation 4 from Kurtek 2017"""
    return np.nan_to_num(np.arccos(np.inner(psi1,psi2)))

# Rescales the cluster mean to be on the great circle
def rescale_cluster_mean(cluster_mean):
    densities = cluster_mean **2
    scaled_densities = densities / densities.sum()
    return np.sqrt(scaled_densities)

# Add outlier detection
# Calculate a couple karcher means and look at how different they are from each other
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
    dzdt_functions_to_warp = DelegatesTo("physiodata")
    srvf_t_min = DelegatesTo("physiodata")
    srvf_t_max = DelegatesTo("physiodata")

    # Holds indices of beats used to calculate initial Karcher mean
    dzdt_karcher_mean_inputs = DelegatesTo("physiodata")
    dzdt_karcher_mean_over_iterations = DelegatesTo("physiodata")
    dzdt_num_inputs_to_group_warping = DelegatesTo("physiodata")
    srvf_iteration_distances = DelegatesTo("physiodata")
    srvf_iteration_energy = DelegatesTo("physiodata")
    
    # For calculating multiple modes
    all_beats_registered_to_initial = Bool(False)
    all_beats_registered_to_mode = Bool(False)
    n_modes = DelegatesTo("physiodata")    
    max_kmeans_iterations = DelegatesTo("physiodata")
    mode_dzdt_karcher_means = DelegatesTo("physiodata")
    mode_cluster_assignment = DelegatesTo("physiodata")
    mode_dzdt_srvf_karcher_means = DelegatesTo("physiodata") 
    
    # graphics items
    karcher_plot = Instance(Plot, transient=True)
    registration_plot = Instance(HPlotContainer, transient=True)
    karcher_plotdata = Instance(ArrayPlotData, transient=True)
    currently_editing = Enum("none", "b", "x")

    # Buttons
    b_calculate_karcher_mean = Button(label="Calculate Karcher Mean")
    b_align_all_beats = Button(label="Warp all")
    b_update_b_point = Button(label="Update B-Point")
    b_update_x_point = Button(label="Update X-Point")

    interactive = Bool(False)

    # Hold the physio plots
    # For selecting points on the Karcher Mean
    ptool_t = Float(0.)
    ptool_t_selection = Float(0.)
    point_plots = Instance(VPlotContainer,transient=True)
    ptool_index_in_warps = Int()

    def __init__(self,**traits):
        super(GroupRegisterDZDT,self).__init__(**traits)

        # Set the initial path to whatever's in the physiodata file
        logger.info("Initializing dZdt registration")

        # Is there already a Karcher mean in the physiodata?
        self.karcher_mean_calculated = self.dzdt_srvf_karcher_mean.size > 0 \
                                and self.dzdt_karcher_mean.size > 0

        # Process dZdt data before srvf analysis
        self._update_original_functions()

        self.n_functions = self.dzdt_functions_to_warp.shape[0]
        self.n_samples = self.dzdt_functions_to_warp.shape[1]
        self.karcher_mean_time = self.sample_times + self.srvf_t_min \
                                    - self.physiodata.dzdt_pre_peak
        self._init_warps()

        self.select_new_samples()

    def _init_warps(self):
        """ For loading warps from mea.mat """
        if self.dzdt_warping_functions.shape[0] == \
                                self.dzdt_functions_to_warp.shape[0]:
            self.all_beats_registered_to_mode = True
            self._forward_warp_beats()

    def _global_ensemble_default(self):
        return GlobalEnsembleAveragedHeartBeat(physiodata=self.physiodata)

    def _ptool_t_changed(self):
        """ Responds when the cursor moves in the mean plot"""
        if self.currently_editing == "none":
            return

        """
        self.point_plotdata = ArrayPlotData(
            peak_times=self.physiodata.peak_times.flatten(),
            x_times = self.physiodata.x_indices - self.physiodata.dzdt_pre_peak,
            lvet = self.physiodata.lvet,
            pep = self.physiodata.pep
        )
        """
        # ptool_t is in dzdt_time
        icg_t_offset = self.physiodata.dzdt_pre_peak
        self.ptool_index_in_warps = int(
                    np.floor(self.ptool_t - self.srvf_t_min + icg_t_offset))
        logger.info("Closest index %d, ptool_t %.2f",
                    self.ptool_index_in_warps, self.ptool_t)
        if self.currently_editing == "b":
            b_indices = self.dzdt_warping_functions[:, self.ptool_index_in_warps]
            lvet = self.physiodata.x_indices - b_indices
            pep = b_indices - icg_t_offset
            self.point_plotdata.set_data("pep", pep)

        elif self.currently_editing == "x":
            x_indices = self.dzdt_warping_functions[:, self.ptool_index_in_warps]
            lvet = x_indices - self.physiodata.b_indices
            self.point_plotdata.set_data("x_times",x_indices - icg_t_offset)

        self.point_plotdata.set_data("lvet", lvet)

    def _ptool_t_selection_changed(self):
        if self.currently_editing == "none": return
        self.karcher_plot.title = ""
        if self.currently_editing == "b":
            orig_time = self.physiodata.ens_avg_b_time
            self.physiodata.ens_avg_b_time = self.ptool_t
            self.physiodata.b_indices = self.dzdt_warping_functions[:,
                                            self.ptool_index_in_warps].copy()
            logger.info("Changed B from %d to %d", orig_time,
                        self.physiodata.ens_avg_b_time)
        elif self.currently_editing == "x":
            orig_time = self.physiodata.ens_avg_x_time
            self.physiodata.ens_avg_x_time = self.ptool_t
            self.physiodata.x_indices = self.dzdt_warping_functions[:,
                                            self.ptool_index_in_warps].copy()
            logger.info("Changed X from %d to %d", orig_time,
                        self.physiodata.ens_avg_x_time)
        self.physiodata.lvet = self.point_plotdata.get_data("lvet")
        self.physiodata.pep = self.point_plotdata.get_data("pep")
        self.currently_editing = "none"

    def _b_update_b_point_fired(self):
        if not self.all_beats_registered_to_mode:
            messagebox("Click Warp all first.")
            return
        self.currently_editing = "b"
        logger.info("Editing B point")
        self.karcher_plot.title = "Select B Point"

    def _b_update_x_point_fired(self):
        if not self.all_beats_registered_to_mode:
            messagebox("Click Warp all first.")
            return
        self.currently_editing = "x"
        logger.info("Editing X point")
        self.karcher_plot.title = "Select X Point"

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
        line_plot = karcher_plot.plot(("karcher_time","karcher_mean"),
                    line_width=3,color="blue")[0]
        # Create an overlay tool
        karcher_plot.datasources['karcher_time'].sort_order = "ascending"
        p_tool = BTool(line_plot=line_plot, component=karcher_plot)
        karcher_plot.overlays.append(p_tool)
        p_tool.sync_trait("time",self,"ptool_t")
        p_tool.sync_trait("selected_time",self,"ptool_t_selection")
        # Create a marker
        p_marker = BMarker(line_plot=line_plot, component=karcher_plot,
                       color="black",selected_time=self.global_ensemble.b.time)
        karcher_plot.overlays.append(p_marker)

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


        if self.all_beats_registered_to_initial or \
           self.all_beats_registered_to_mode:
            image_data = self.registered_functions.copy()
        else:
            image_data = self.dzdt_functions_to_warp.copy()

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

        aligned_functions = np.zeros_like(self.dzdt_functions_to_warp)
        t = self.sample_times
        for k in range(self.n_functions):
            aligned_functions[k] = np.interp((t[-1] - t[0])*gam[k] + t[0],
                                             t, self.dzdt_functions_to_warp[k])

        self.registered_functions = aligned_functions

    @on_trait_change("dzdt_num_inputs_to_group_warping")
    def select_new_samples(self):
        nbeats = self.dzdt_functions_to_warp.shape[0]
        nsamps = min(self.dzdt_num_inputs_to_group_warping, nbeats)
        self.dzdt_karcher_mean_inputs = np.random.choice(nbeats,size=nsamps,replace=False)

    @on_trait_change(
       ("physiodata.srvf_lambda, physiodata.srvf_karcher_iterations, "
        "physiodata.srvf_update_min, physiodata.srvf_use_moving_ensembled, "
        "physiodata.srvf_karcher_mean_subset_size, "
        "physiodata.bspline_before_warping"
    ))
    def params_edited(self):
        self.dirty = True
        self.karcher_mean_calculated = False
        self.all_beats_registered_to_initial = False
        self.all_beats_registered_to_mode = False
        self._update_original_functions()

    def _update_original_functions(self):
        logger.info("updating functions to register")
        # offset from t=0 at R peak
        self.dzdt_functions_to_warp = self.physiodata.mea_dzdt_matrix[
                                    : , self.srvf_t_min:self.srvf_t_max] if \
                self.srvf_use_moving_ensembled else self.physiodata.dzdt_matrix[
                                    : , self.srvf_t_min:self.srvf_t_max]
        self.sample_times = np.arange(self.srvf_t_max - self.srvf_t_min,
                                        dtype=np.float)

        if self.bspline_before_warping:
            logger.info("Smoothing inputs with B-Splines")
            self.dzdt_functions_to_warp = np.row_stack([ UnivariateSpline(
                self.sample_times, func, s=0.05)(self.sample_times) \
                for func in self.dzdt_functions_to_warp]
            )
        if self.interactive:
            self.all_registration_plotdata.set_data("image_data",
                                                    self.dzdt_functions_to_warp.copy())
            self.all_plot.request_redraw()

    def _b_calculate_karcher_mean_fired(self):
        self.calculate_karcher_mean()

    def calculate_karcher_mean(self):
        """
        Calculates an initial Karcher Mean.
        """
        reg_prob = RegistrationProblem(
            self.dzdt_functions_to_warp[self.dzdt_karcher_mean_inputs].T,
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
            self.single_registration_plotdata.set_data("karcher_mean",
                                            self.dzdt_karcher_mean)
            self.single_plot.request_redraw()
        self.rp = reg_problem

    def _b_align_all_beats_fired(self):
        self.align_all_beats_to_initial()
        
    def detect_modes(self):
        """
        Uses the SRD-based clustering method described in Kurtek 2017
        """
        if not self.karcher_mean_calculated:
            fail("Must calculate an initial Karcher mean first")
            return
        if not self.all_beats_registered_to_initial:
            fail("All beats must be registered to the initial Karcher mean")
            return
        
        # Calculate the SRDs
        dzdt_functions_to_warp = self.dzdt_functions_to_warp.T
        warps = self.dzdt_warping_functions.T
        wmax = warps.max()
        wmin = warps.min()
        warps = (warps - wmin) / (wmax - wmin)
        densities = np.diff(warps,axis=0)
        SRDs = np.sqrt(densities)        
        # pairwise distances
        srd_pairwise = pairwise_distances(SRDs.T,metric=fisher_rao_dist)
        tri = srd_pairwise[np.triu_indices_from(srd_pairwise,1)]
        linkage = complete(tri)
        
        # Performs an iteration of k-means
        def cluster_karcher_means(initial_assignments):
            cluster_means = []
            cluster_ids = np.unique(initial_assignments).tolist()
            warping_functions = np.zeros_like(SRDs)
        
            # Calculate a Karcher mean for each cluster
            for cluster_id in cluster_ids:
                print "Cluster ID:", cluster_id
                cluster_id_mask = initial_assignments == cluster_id
                cluster_srds = SRDs[:,cluster_id_mask]
        
                # If there is only a single SRD in this cluster, it is the mean
                if cluster_id_mask.sum() == 1:
                    cluster_means.append(cluster_srds)
                    continue
        
                # Run group registration to get Karcher mean
                cluster_reg = RegistrationProblem(
                        cluster_srds,
                        sample_times=np.arange(SRDs.shape[0], dtype=np.float),
                        max_karcher_iterations = self.srvf_max_karcher_iterations,
                        lambda_value = self.srvf_lambda,
                        update_min = self.srvf_update_min 
                )
                cluster_reg.run_registration_parallel()
                cluster_means.append(cluster_reg.function_karcher_mean)
                warping_functions[:,cluster_id_mask] = cluster_reg.mean_to_orig_warps
        
            scaled_cluster_means = [rescale_cluster_mean(cm) for cm in cluster_means]
        
            # There are now k cluster means, which is closest for each SRD?
            # Also save its distance to its cluster's Karcher mean
            srd_cluster_assignments = []
            srd_cluster_distances = []
            for srd in SRDs.T:
                distances = [fisher_rao_dist(cluster_mean,srd) for cluster_mean in scaled_cluster_means]
                cluster_num = cluster_ids[np.argmin(distances)]
                srd_cluster_assignments.append(cluster_num)
                corresponding_mean = cluster_means[cluster_ids.index(cluster_num)]
                srd_cluster_distances.append(fisher_rao_dist(corresponding_mean, srd))
        
            return np.array(srd_cluster_assignments), np.array(srd_cluster_distances)**2, scaled_cluster_means, warping_functions
        
        # Iterate until assignments stabilize
        last_assignments = fcluster(linkage, self.n_modes, criterion="maxclust")
        stabilized = False
        n_iters = 0
        old_assignments = [last_assignments]
        old_means = []
        while not stabilized and n_iters < self.max_kmeans_iterations:
            logger.info("Iteration %d", n_iters)
            assignments, distances, cluster_means, warping_funcs = cluster_karcher_means(
                last_assignments)
            stabilized = np.all(last_assignments == assignments)
            last_assignments = assignments.copy()
            old_assignments.append(last_assignments)
            old_means.append(cluster_means)
            n_iters += 1            
            
        # Finalize the clusters by aligning all the functions to the cluster mean
        # Iterate until assignments stabilize
        cluster_means = []
        cluster_ids = np.unique(assignments)
        warping_functions = np.zeros_like(dzdt_functions_to_warp)
        self.registered_functions = np.zeros_like(self.dzdt_functions_to_warp)
        # Calculate a Karcher mean for each cluster
        for cluster_id in cluster_ids:
            cluster_id_mask = assignments == cluster_id
            cluster_funcs = self.dzdt_functions_to_warp.T[:,cluster_id_mask]
        
            # If there is only a single SRD in this cluster, it is the mean
            if cluster_id_mask.sum() == 1:
                cluster_means.append(cluster_funcs)
                continue
        
            # Run group registration to get Karcher mean
            cluster_reg = RegistrationProblem(
                    cluster_funcs,
                    sample_times=np.arange(self.dzdt_functions_to_warp.shape[1], dtype=np.float),
                    max_karcher_iterations = self.srvf_max_karcher_iterations,
                    lambda_value = self.srvf_lambda,
                    update_min = self.srvf_update_min
            )
            cluster_reg.run_registration_parallel()
            cluster_means.append(cluster_reg.function_karcher_mean)
            warping_functions[:,cluster_id_mask] = cluster_reg.mean_to_orig_warps        
            self.registered_functions[cluster_id_mask] = cluster_reg.registered_functions.T        
        self.mode_dzdt_karcher_means = np.row_stack(cluster_means)
        self.mode_cluster_assignment = assignments
        self.all_beats_registered_to_mode = True
        
    def align_all_beats_to_initial(self):
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
        fy, fx = np.gradient(self.dzdt_functions_to_warp.T, 1, 1)
        srvf_functions = (fy / np.sqrt(np.abs(fy) + eps)).T

        gam = np.zeros(self.dzdt_functions_to_warp.shape, dtype=np.float)
        aligned_functions = self.dzdt_functions_to_warp.copy()
        logger.info("aligned_functions %d", id(aligned_functions))
        logger.info("self.dzdt_functions_to_warp %d", id(self.dzdt_functions_to_warp))

        t = self.sample_times
        for k in range(self.n_functions):
            q_c = srvf_functions[k] / np.linalg.norm(srvf_functions[k])
            G,T = dp(normed_template_func, t, q_c, t, t, t, self.srvf_lambda)
            gam0 = np.interp(self.sample_times, T, G)
            gam[k] = (gam0-gam0[0])/(gam0[-1]-gam0[0])  # change scale
            aligned_functions[k] = np.interp((t[-1] - t[0])*gam[k] + t[0],
                                             t, self.dzdt_functions_to_warp[k])

            if self.interactive:

                # Update the image plot
                self.all_registration_plotdata.set_data("image_data", aligned_functions)

                # Update the registration plot
                self.single_registration_plotdata.set_data("registered_func",
                                                        aligned_functions[k])
                self.single_plot.request_redraw()

                (cont,skip) = progress.update(k)

        self.registered_functions = aligned_functions.copy()

        self.dzdt_warping_functions = gam * (
                                        self.srvf_t_max - self.srvf_t_min) + \
                                        self.srvf_t_min

        if self.interactive:
            progress.update(k+1)

        self.all_beats_registered_to_initial = True
    

    def _point_plots_default(self):
        """
        Instead of defining these in __init__, only
        construct the plots when a ui is requested
        """

        self.point_plotdata = ArrayPlotData(
            peak_times=self.physiodata.peak_times.flatten(),
            x_times = self.physiodata.x_indices - self.physiodata.dzdt_pre_peak,
            lvet = self.physiodata.lvet,
            pep = self.physiodata.pep
        )

        container = VPlotContainer(
            resizable="hv", bgcolor="lightgray", fill_padding=True, padding=10
        )

        for sig in ("pep", "x_times","lvet"):
            temp_plt = Plot(self.point_plotdata)
            temp_plt.plot(("peak_times",sig),line_width=2)
            temp_plt.title = sig
            container.add(temp_plt)
        container.padding_top =10

        return container

    mean_widgets =VGroup(
        HGroup(
                Item("b_update_b_point", show_label=False,
                    enabled_when="karcher_mean_calculated"),
                Item("b_update_x_point", show_label=False,
                    enabled_when="karcher_mean_calculated")
        ),
        VGroup(
            Item("karcher_plot",editor=ComponentEditor(),
                 show_label=False),
            Item("registration_plot", editor=ComponentEditor(),
                show_label=False), show_labels=False),
            Item("srvf_use_moving_ensembled",
                    label="Use Moving Ensembled dZ/dt"),
            Item("bspline_before_warping",label="B Spline smoothing"),
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
    )

    traits_view = MEAPView(
        HSplit(
            Item("point_plots", editor=ComponentEditor(),show_label=False),
            mean_widgets
        ),
        resizable=True,
        win_title="ICG Warping Tools",
        width=800, height=700,
        buttons = [OKButton,CancelButton]
    )
