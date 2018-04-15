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
from meap.beat_train import ModeKarcherBeatTrain
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
    return np.nan_to_num(np.arccos(np.inner(psi1.squeeze(),psi2.squeeze())))

# Rescales the cluster mean to be on the great circle
def rescale_cluster_mean(cluster_mean):
    densities = cluster_mean.squeeze() **2
    densities_sum = densities.sum()
    if densities_sum <= 0:
        densities = np.ones_like(cluster_mean)
    scaled_densities = densities / densities.sum()
    return np.sqrt(scaled_densities)

def get_closest_mean(srds, modes_dict):
    """
    returns an array of which mode is closest to each srd and an
    array of FR distances 
    """
    srd_cluster_assignments = []
    srd_cluster_distances = []
    
    cluster_ids = modes_dict.keys()
    for srd in srds.T:
        distances = [fisher_rao_dist(modes_dict[key],srd) for key in cluster_ids]
        cluster_num = cluster_ids[np.argmin(distances)]
        srd_cluster_assignments.append(cluster_num)
        corresponding_mean = modes_dict[cluster_num]
        srd_cluster_distances.append(fisher_rao_dist(corresponding_mean, srd))
    
    return np.array(srd_cluster_assignments), np.array(srd_cluster_distances)**2


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
    
    # Configures the slice of time to be registered
    srvf_t_min = DelegatesTo("physiodata")
    srvf_t_max = DelegatesTo("physiodata")
    dzdt_karcher_mean_time = DelegatesTo("physiodata")
    dzdt_mask = Array
    
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

    # Buttons
    b_calculate_karcher_mean = Button(label="Calculate Karcher Mean")
    b_align_all_beats = Button(label="Warp all")
    b_find_modes = Button(label="Discover Modes")
    b_edit_modes = Button(label="Score Modes")
    interactive = Bool(False)

    # Holds the karcher modes
    mode_beat_train = Instance(ModeKarcherBeatTrain)
    edit_listening = Bool(False,desc="If true, update_plots is called"
            " when a beat gets hand labeled")

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
        self._init_warps()

        self.select_new_samples()
        
    def _mode_beat_train_default(self):
        logger.info("creating default mea_beat_train")
        assert self.physiodata is not None
        mkbt = ModeKarcherBeatTrain(physiodata=self.physiodata)
        return mkbt
        
    def _init_warps(self):
        """ For loading warps from mea.mat """
        if self.dzdt_warping_functions.shape[0] == \
                                self.dzdt_functions_to_warp.shape[0]:
            self.all_beats_registered_to_mode = True
            self._forward_warp_beats()

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
            karcher_mean = unreg_mean[self.dzdt_mask]
        else:
            karcher_mean = self.dzdt_karcher_mean

        self.karcher_plotdata = ArrayPlotData(
            time = self.global_ensemble.dzdt_time,
            unregistered_mean = self.global_ensemble.dzdt_signal,
            karcher_mean = karcher_mean,
            karcher_time = self.dzdt_karcher_mean_time
        )
        karcher_plot = Plot(self.karcher_plotdata)
        karcher_plot.plot(("time","unregistered_mean"),
                    line_width=1,color="lightblue")
        line_plot = karcher_plot.plot(("karcher_time","karcher_mean"),
                    line_width=3,color="blue")[0]
        # Create an overlay tool
        karcher_plot.datasources['karcher_time'].sort_order = "ascending"

        return karcher_plot

    def _registration_plot_default(self):
        """
        Instead of defining these in __init__, only
        construct the plots when a ui is requested
        """
        unreg_mean = self.global_ensemble.dzdt_signal
        # Temporarily fill in the karcher mean
        if not self.karcher_mean_calculated:
            karcher_mean = unreg_mean[self.dzdt_mask]
        else:
            karcher_mean = self.dzdt_karcher_mean

        self.single_registration_plotdata = ArrayPlotData(
            karcher_time = self.dzdt_karcher_mean_time,
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
        t = self.dzdt_karcher_mean_time
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
       ("physiodata.srvf_lambda, "
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
        logger.info("updating time slice and functions to register")
        
        # Get the time relative to R
        dzdt_time = np.arange(self.physiodata.dzdt_matrix.shape[1],dtype=np.float) \
                    - self.physiodata.dzdt_pre_peak
        self.dzdt_mask = (dzdt_time >= self.srvf_t_min) * (dzdt_time <= self.srvf_t_max)
        srvf_time = dzdt_time[self.dzdt_mask]
        self.dzdt_karcher_mean_time = srvf_time
        
        # Extract corresponding data
        self.dzdt_functions_to_warp = self.physiodata.mea_dzdt_matrix[
                                    : , self.dzdt_mask] if \
                self.srvf_use_moving_ensembled else self.physiodata.dzdt_matrix[
                                    : , self.dzdt_mask]

        if self.bspline_before_warping:
            logger.info("Smoothing inputs with B-Splines")
            self.dzdt_functions_to_warp = np.row_stack([ UnivariateSpline(
                self.dzdt_karcher_mean_time, func, s=0.05)(self.dzdt_karcher_mean_time) \
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
            sample_times=self.dzdt_karcher_mean_time,
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
        
    def _b_find_modes_fired(self):
        self.detect_modes()
        
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
            cluster_means = {}
            cluster_ids = np.unique(initial_assignments).tolist()
            warping_functions = np.zeros_like(SRDs)
        
            # Calculate a Karcher mean for each cluster
            for cluster_id in cluster_ids:
                print "Cluster ID:", cluster_id
                cluster_id_mask = initial_assignments == cluster_id
                cluster_srds = SRDs[:,cluster_id_mask]
        
                # If there is only a single SRD in this cluster, it is the mean
                if cluster_id_mask.sum() == 1:
                    cluster_means[cluster_id] = cluster_srds
                    continue
        
                # Run group registration to get Karcher mean
                cluster_reg = RegistrationProblem(
                        cluster_srds,
                        sample_times = np.arange(SRDs.shape[0], dtype=np.float),
                        max_karcher_iterations = self.srvf_max_karcher_iterations,
                        lambda_value = self.srvf_lambda,
                        update_min = self.srvf_update_min 
                )
                cluster_reg.run_registration_parallel()
                cluster_means[cluster_id] = cluster_reg.function_karcher_mean
                warping_functions[:,cluster_id_mask] = cluster_reg.mean_to_orig_warps
            
            # Scale the cluster Karcher means so the FR distance works
            scaled_cluster_means = {}
            for k,v in cluster_means.iteritems():
                scaled_cluster_means[k] = rescale_cluster_mean(v)
        
            # There are now k cluster means, which is closest for each SRD?
            # Also save its distance to its cluster's Karcher mean
            srd_cluster_assignments, srd_cluster_distances = get_closest_mean(SRDs, scaled_cluster_means)
            return srd_cluster_assignments, srd_cluster_distances, scaled_cluster_means, warping_functions
        
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
        cluster_means = {}
        cluster_ids = np.unique(assignments)
        warping_functions = np.zeros_like(dzdt_functions_to_warp)
        self.registered_functions = np.zeros_like(self.dzdt_functions_to_warp)
        # Calculate a Karcher mean for each cluster
        for cluster_id in cluster_ids:
            cluster_id_mask = assignments == cluster_id
            cluster_funcs = self.dzdt_functions_to_warp.T[:,cluster_id_mask]
        
            # If there is only a single SRD in this cluster, it is the mean
            if cluster_id_mask.sum() == 1:
                cluster_means[cluster_id] = cluster_funcs
                continue
        
            # Run group registration to get Karcher mean
            cluster_reg = RegistrationProblem(
                    cluster_funcs,
                    sample_times = self.dzdt_karcher_mean_time,
                    max_karcher_iterations = self.srvf_max_karcher_iterations,
                    lambda_value = self.srvf_lambda,
                    update_min = self.srvf_update_min
            )
            cluster_reg.run_registration_parallel()
            cluster_means[cluster_id] = cluster_reg.function_karcher_mean
            warping_functions[:, cluster_id_mask] = cluster_reg.mean_to_orig_warps    
            self.registered_functions[cluster_id_mask] = cluster_reg.registered_functions.T        
        
        # Save the warps to the modes as the final warping functions        
        self.dzdt_warping_functions = warping_functions.T \
                                      * (self.srvf_t_max - self.srvf_t_min) \
                                      + self.srvf_t_min
        
        # re-order the means
        cluster_ids = sorted(cluster_means.keys())
        final_assignments = np.zeros_like(assignments)
        final_modes = []
        for final_id, orig_id in enumerate(cluster_ids):
            final_assignments[assignments==orig_id] = final_id
            final_modes.append(cluster_means[orig_id].squeeze())
        
        self.mode_dzdt_karcher_means = np.row_stack(final_modes)
        self.mode_cluster_assignment = final_assignments
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

        t = self.dzdt_karcher_mean_time
        for k in range(self.n_functions):
            q_c = srvf_functions[k] / np.linalg.norm(srvf_functions[k])
            G,T = dp(normed_template_func, t, q_c, t, t, t, self.srvf_lambda)
            gam0 = np.interp(self.dzdt_karcher_mean_time, T, G)
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
    
    
    def _b_edit_modes_fired(self):
        self.mode_beat_train.edit_traits()
        
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
        VGroup(
            Item("karcher_plot",editor=ComponentEditor(),
                 show_label=False),
            Item("registration_plot", editor=ComponentEditor(),
                show_label=False), show_labels=False),
            Item("srvf_use_moving_ensembled",
                    label="Use Moving Ensembled dZ/dt"),
            Item("bspline_before_warping",label="B Spline smoothing"),
            Item("srvf_t_min",label="Epoch Start Time"),
            Item("srvf_t_max",label="Epoch End Time"),
            Item("srvf_lambda",label="Lambda Value"),
            Item("dzdt_num_inputs_to_group_warping",
                    label="Template uses N beats"),
            Item("srvf_max_karcher_iterations", label="Max Karcher Iterations"),
        HGroup(
            Item("b_calculate_karcher_mean", label="Step 1:",
                enabled_when="dirty"),
            Item("b_align_all_beats", label="Step 2:",
                enabled_when="karcher_mean_calculated")
        ),
        label = "Initial Karcher Mean"
    )
    
    mode_widgets = VGroup(
        Item("n_modes", label="Number of Modes/Clusters"),
        Item("max_kmeans_iterations"),
        Item("b_find_modes", show_label=False,
             enabled_when = "all_beats_registered_to_initial"),
        Item("b_edit_modes", show_label=False,
             enabled_when = "all_beats_registered_to_mode")
    )

    traits_view = MEAPView(
        HSplit(
            mean_widgets,
            mode_widgets
        ),
        resizable=True,
        win_title="ICG Warping Tools",
        width=800, height=700,
        buttons = [OKButton,CancelButton]
    )
