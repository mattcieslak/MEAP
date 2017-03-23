from traits.api import (HasTraits, Bool, File, Array, Property,Instance, DelegatesTo)
import joblib
import logging
import os
import numpy as np
from meap import messagebox
from meap.io import PhysioData
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from pyface.api import ProgressDialog
from copy import deepcopy
from meap.filters import smooth

logger = logging.getLogger(__name__)

"""
Classification works by converting each beat into a matrix
where each row is centered on a millisecond between the r
peak and c peak. The center time is padded by 
"""

def mat_extract(matrix, start_inds, end_inds):
    return np.row_stack([row[start:end] for row,start,end in \
            zip(matrix, start_inds, end_inds)])

class BPointClassifier(HasTraits):
    physiodata = Instance(PhysioData)
    # Config from physiodata
    pre_point_msec = DelegatesTo("physiodata",
            "bpoint_classifier_pre_point_msec")
    post_point_msec =DelegatesTo("physiodata",
            "bpoint_classifier_post_point_msec")
    sample_every_n_msec = DelegatesTo("physiodata",
            "bpoint_classifier_sample_every_n_msec")
    false_distance_min = DelegatesTo("physiodata",
            "bpoint_classifier_false_distance_min")
    use_bpoint_prior = DelegatesTo("physiodata",
            "bpoint_classifier_use_bpoint_prior")
    include_derivative = DelegatesTo("physiodata",
            "bpoint_classifier_include_derivative")
    # Config-independent traits    
    bpoint_classifier_file = DelegatesTo('physiodata')
    classifier = Instance(AdaBoostRegressor)
    bpoint_prior = Property(Array)
    save_path = File
    trained = Bool(False)

    def __init__(self, **traits):
        super(HasTraits,self).__init__(**traits)
        # Returns a classifier loaded from disk if one exists, otherwise an
        # empty adaboost regressor
        if self.classifier is None:
            if os.path.exists(self.physiodata.bpoint_classifier_file):
                logger.info("loading pre-existing classifier %s", 
                        self.bpoint_classifier_file)
                try:
                    self.classifier = joblib.load(self.bpoint_classifier_file)
                    self.trained = True
                    self.saved_since_train = True
                except Exception, e:
                    logger.warn(
                            "Unable to load %s. Making an empty classifier instead.\n" \
                            "Error was %s",
                                self.physiodata.bpoint_classifier_file,e)
            else:
                self.classifier = AdaBoostRegressor(
                    DecisionTreeRegressor(max_leaf_nodes=None,max_depth=None),
                               n_estimators=300,learning_rate=1.)
        else:
            logger.info("using provided classifier %s", str(self.classifier))
            self.trained = True

    def make_training_set(self):
        marked_samples = np.flatnonzero(self.physiodata.hand_labeled) 
        samples = []
        labels = []
        for beatnum in marked_samples:
            r,b,c = (
                self.physiodata.r_indices[beatnum],
                self.physiodata.b_indices[beatnum],
                self.physiodata.c_indices[beatnum])

            samples.append(self.beat_to_time_feature_matrix(beatnum))
            labels.append(np.arange(r,c) - b)

        return samples, labels
        
    def train(self):
        classifier = self.classifier
        if self.trained:
            logger.info("Re-training classifier")
        else:
            logger.info("Initial training of classifier")
        samples, labels = self.make_training_set()
        classifier.fit(np.row_stack(samples), np.concatenate(labels))
        self.trained = True
        logger.info("Training done.")
    
    def beat_to_time_feature_matrix(self, beatnum):
        """
        Turns a heartbeat into a matrix. There is one row for each millisecond between
        the r point and the c point. Each row represents the time corresponding to
        its center.  That timepoint is padded by self.pre_point_msec and 
        self.post_point_msec. If self.include_derivative, the derivative is appended to 
        the end of each row 
        """
        include_derivative=self.include_derivative
        pre_msec = self.pre_point_msec
        post_msec = self.post_point_msec
        r_ind = self.physiodata.r_indices[beatnum]
        c_ind = self.physiodata.c_indices[beatnum]
        _sig = self.physiodata.mea_dzdt_matrix[beatnum]
        # The targets are msec from the b-point
        if include_derivative:
            signal = [np.concatenate([_sig[(ind-pre_msec):(ind+post_msec+1)],
                10*np.diff(
                   smooth(_sig[(ind-pre_msec):(ind+post_msec+1)], 7))]) \
                                for ind in xrange(r_ind,c_ind)]
        else:
            signal = [_sig[(ind-pre_msec):(ind+post_msec+1)] for ind \
                                                    in xrange(r_ind,c_ind)]
        return np.row_stack(signal)
    
    def beat_obj_to_time_feature_matrix(self, beat):
        """
        Turns a heartbeat into a matrix. There is one row for each millisecond between
        the r point and the c point. Each row represents the time corresponding to
        its center.  That timepoint is padded by self.pre_point_msec and 
        self.post_point_msec. If self.include_derivative, the derivative is appended to 
        the end of each row 
        """
        include_derivative=self.include_derivative
        pre_msec = self.pre_point_msec
        post_msec = self.post_point_msec
        r_ind = beat.r.index
        c_ind = beat.c.index
        _sig = beat.dzdt_signal
        # The targets are msec from the b-point
        if include_derivative:
            signal = [np.concatenate([_sig[(ind-pre_msec):(ind+post_msec+1)],
                10*np.diff(
                   smooth(_sig[(ind-pre_msec):(ind+post_msec+1)], 7))]) \
                                for ind in xrange(r_ind,c_ind)]
        else:
            signal = [_sig[(ind-pre_msec):(ind+post_msec+1)] for ind \
                                                    in xrange(r_ind,c_ind)]
        return np.row_stack(signal)

    def estimate_bpoint(self,beatnum=None,beat_obj=None):
        """
        Returns the index of the estimated b point
        """
        if beatnum is not None:
            features = self.beat_to_time_feature_matrix(beatnum)
            r_index = self.physiodata.r_indices[beatnum]
        elif beat_obj is not None:
            features = self.beat_obj_to_time_feature_matrix(beat_obj)
            r_index = beat_obj.r.index
            
        # Have the regressor predict the distance from the b-point for all candidates
        preds = np.abs(self.classifier.predict(features))
        multi_estimates = preds==preds.min()
        total_mins = multi_estimates.sum()
        if total_mins > 1:
            logger.warn("Multiple possible bpoints found (%d)" % total_mins)
            offset = np.flatnonzero(multi_estimates).mean()
        else:
            offset = np.argmin(preds)
        return int(r_index  + offset)

    def _get_bpoint_prior(self):
        return self.physiodata.b_indices[self.physiodata.hand_labeled > 0]
    
    def save(self):
        outpath =  self.physiodata.bpoint_classifier_file
        if not os.path.exists(outpath):
            logger.warn("Overwriting existing classifier file")
        logger.info("Saving classifier to " + outpath)
        joblib.dump(self.classifier, outpath,compress=3)
        if not os.path.exists(outpath):
            logger.warn("Failed to write classifier file!!")
            return False
        return True

    def check_performance(self):
        logger.info("Checking classifier performance...")
        orig_sd = np.std(self.bpoint_prior)
        samples, labels = self.make_training_set()
        classifier = deepcopy(self.classifier)
        
        progress = ProgressDialog(title="B-Point Classifier Cross Validation", min=0,
                max = len(labels), show_time=True,
                message="Cross validating the classifier...")
        progress.open()

        errors = []
        for i in  range(len(labels)):
            training_samples =  [ s for _i,s in enumerate(samples) if _i != i ]
            training_labels = [ s for _i,s in enumerate(labels) if _i != i ]
            testing_sample = samples[i]
            testing_labels = labels[i]

            classifier.fit(np.row_stack(training_samples), 
                                np.concatenate(training_labels))

            prediction = np.abs(classifier.predict(testing_sample))
            multi_estimates = prediction == prediction.min()
            total_mins = multi_estimates.sum()
            if total_mins > 1:
                guess = np.flatnonzero(multi_estimates).mean()
            else:
                guess = np.argmin(prediction)

            errors.append( np.argmin(np.abs(testing_labels)) - guess )
            logger.info("Sample %d error: %.2f msec",i, errors[-1])
            (cont,skip) = progress.update(i)
        (cont,skip) = progress.update(i+1)
            
        errors = np.array(errors)
        messagebox(
            """
            Training completed on %d samples
            Prediction error: %.4fmsec (+- %.4f)
            Original sd %.4f
            """%(samples.shape[0],
            errors.mean(), errors.std(), orig_sd))
