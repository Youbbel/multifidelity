
#################################
# imports
#################################

from __future__ import annotations

import trieste
import tensorflow as tf
import numpy as np
from trieste.acquisition import (
    AcquisitionFunction,
    SingleModelAcquisitionBuilder,
    Product,
)
from typing import Optional, cast, Generic, TypeVar, Sequence
from trieste.data import Dataset
from trieste.types import TensorType
import tensorflow_probability as tfp
from trieste.acquisition import AcquisitionFunctionClass
#from trieste.objectives import scaled_branin
from trieste.models.gpflow.builders import build_gpr
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.interfaces import (
    ProbabilisticModel,
    TrainableProbabilisticModel,
)
from trieste.acquisition.optimizer import generate_continuous_optimizer
from gpflow.logdensities import multivariate_normal
import gpflow
from gpflow.models import GPR
import math
import matplotlib.pyplot as plt

from trieste.data import add_fidelity_column
from trieste.data import split_dataset_by_fidelity
import time

OBJECTIVE = "OBJECTIVE"

ProbabilisticModelType = TypeVar(
    "ProbabilisticModelType", bound="ProbabilisticModel", contravariant=True
)

#################################
# problem parameters
#################################

# random seed
my_seed = 35

# pb dimension
input_dim = 1

# bounds
lb = np.array([0.0])
ub = np.array([1.0])

# size of initial dataset
init_low = 9
init_medium = 6
init_high = 3


# CPU costs
low_cost = 1.0
medium_cost = 2.0
high_cost = 5.0

# number of iterations
num_steps=0

#################################
# fidelity model wrapper
#################################

class MultiFidelityModel(TrainableProbabilisticModel):
    """
    This is a wrapper for a two fidelity model following (https://hal.inria.fr/hal-02901774/document)
    """
    def __init__(
        self,
        low_fidelity_model: GaussianProcessRegression,
        virtual_low_fidelity_model: GaussianProcessRegression,
        medium_residual_model: GaussianProcessRegression,
        virtual_medium_fidelity_model: GaussianProcessRegression,
        high_residual_model: GaussianProcessRegression,
        num_fidelities: int
    ):
        r"""
        TODO
        The order of individual models specified at :meth:`__init__` determines the order of the
        :class:`MultiFidelityModel` fidelities.

        :param low_fidelity_model: The GP model of the lowest fidelity
        :param residual_model: The GP model of the residual
        """
        self._low_fidelity_model = low_fidelity_model
        self._virtual_low_fidelity_model = virtual_low_fidelity_model
        self._medium_residual_model = medium_residual_model
        self._virtual_medium_fidelity_model = virtual_medium_fidelity_model
        self._high_residual_model = high_residual_model
        self._rho_low = gpflow.Parameter(1.0, trainable=True) # set this as a Parameter so that we can optimize it
        self._rho_medium = gpflow.Parameter(1.0, trainable=True) # set this as a Parameter so that we can optimize it

        self._num_fidelities = num_fidelities

    @property
    def num_fidelities(self) -> int:
        return self._num_fidelities

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        predict latents
        :param query_points: The points at which to make predictions, of shape [..., D+1]. The last
            dimension is the fidelity level.
        :return: The predictions from the model with fidelity corresponding to the last dimension of
            query_points.
        """
        input_points = query_points[..., :-1]
        fidelities = query_points[..., -1:]  # [..., 1]
        lowfi_mask = (fidelities == 0.)
        medfi_mask = (fidelities == 1.0)
        highfi_mask = (fidelities == 2.0)

        mean_lowfi, var_lowfi = self._low_fidelity_model.predict(input_points)
        mean_virtual_lowfi, var_virtual_lowfi = self._virtual_low_fidelity_model.predict(input_points)

        #mean_medfi, var_medfi = self._medium_fidelity_model.predict(input_points)
        mean_virtual_medfi, var_virtual_medfi = self._virtual_medium_fidelity_model.predict(input_points)

        mean_residual_medfi, var_residual_medfi = self._medium_residual_model.predict(input_points)
        mean_residual_highfi, var_residual_highfi = self._high_residual_model.predict(input_points)

        mean = tf.where(lowfi_mask, mean_lowfi, tf.where(medfi_mask, mean_residual_medfi + self._rho_low * mean_lowfi, mean_residual_highfi + self._rho_medium *mean_virtual_medfi))
        var =  tf.where(lowfi_mask, var_lowfi, tf.where(medfi_mask, var_residual_medfi + (self._rho_low**2) * var_virtual_lowfi, var_residual_highfi + (self._rho_medium**2) *var_virtual_medfi ))

        #mean = tf.where(lowfi_mask, mean_lowfi, tf.where(medfi_mask, mean_residual_medfi + self._rho_low * mean_lowfi, mean_residual_highfi + self._rho_medium * (mean_residual_medfi + self._rho_low * mean_lowfi)))

        #var =  tf.where(lowfi_mask, var_lowfi, tf.where(medfi_mask, var_residual_medfi + (self._rho_low**2) * var_virtual_lowfi, var_residual_highfi + (self._rho_medium**2) *(var_residual_medfi + (self._rho_low**2) * var_virtual_lowfi) ))

        return mean, var

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:

        input_points = query_points[..., :-1]
        fidelities = query_points[..., -1:]  # [..., 1]
        lowfi_mask = (fidelities == 0.)
        medfi_mask = (fidelities == 1.0)

        samples_lowfi = self._low_fidelity_model.sample(input_points, num_samples)
        samples_residual_medfi = self._medium_residual_model.sample(input_points, num_samples)

        #samples_medfi = self._medium_fidelity_model.sample(input_points, num_samples)
        samples_residual_highfi = self._high_residual_model.sample(input_points, num_samples)

        samples = tf.where(lowfi_mask, samples_lowfi, tf.where(medfi_mask,  samples_residual_medfi + self._rho_low*samples_lowfi ,  samples_residual_highfi + self._rho_medium*(samples_residual_medfi + self._rho_low*samples_lowfi )))
                 #tf.where(lowfi_mask, samples_lowfi, samples_residual + self._rho*samples_lowfi)

        return samples

    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        "predict observations"
        input_points = query_points[..., :-1]
        fidelities = query_points[..., -1:]  # [..., 1]
        lowfi_mask = (fidelities == 0.)
        medfi_mask = (fidelities == 1.0)

        f_mean, f_var = self.predict(query_points)
        obv_noise = tf.where(lowfi_mask,
                             self._low_fidelity_model.get_observation_noise(), tf.where(medfi_mask,
                             (self._rho_low**2)*self._low_fidelity_model.get_observation_noise() + self._medium_residual_model.get_observation_noise(),
                             (self._rho_medium**2)*self._virtual_medium_fidelity_model.get_observation_noise() + self._high_residual_model.get_observation_noise()
                            )
                            )
        return f_mean, f_var #+ obv_noise


    def covariance_with_top_fidelity(self, query_points: TensorType) -> TensorType:
        "covariance between current latent and the highest level latent"
        input_points = query_points[..., :-1]
        fidelities = query_points[..., -1:]  # [..., 1]
        lowfi_mask = (fidelities == 0.)
        medfi_mask = (fidelities == 1.0)

        f_mean, f_var = self.predict(query_points)
        covs = tf.where(lowfi_mask,                                   #
                            self._rho_low *self._rho_medium* f_var,
                            tf.where(medfi_mask, self._rho_medium * f_var, f_var)
                       )
        return covs


    def update(self, dataset: Dataset) -> None:
        """
        Update the two models on their corresponding data. The data for each model is
        extracted by splitting the observations in ``dataset`` by fidelity level.

        :param dataset: The query points and observations for *all* the wrapped models.
        """
        lowfi_points, medfi_points, highfi_points, lowfi_mask, medfi_mask, highfi_mask, ind_lowfi, ind_medfi, ind_highfi = filter_by_fidelity(dataset.query_points)

        self._low_fidelity_model.update(Dataset(lowfi_points, tf.gather(dataset.observations, ind_lowfi)))

        low_virtual_query_points = tf.concat([lowfi_points, medfi_points],0)
        low_virtual_observations = tf.concat([tf.gather(dataset.observations, ind_lowfi), self._low_fidelity_model.predict(medfi_points)[0]],0)
        self._virtual_low_fidelity_model.update(Dataset(low_virtual_query_points, low_virtual_observations))


        medium_residuals = tf.gather(dataset.observations, ind_medfi) - self._rho_low * self._low_fidelity_model.predict(medfi_points)[0]
        self._medium_residual_model.update(Dataset(medfi_points, medium_residuals))

        medium_virtual_query_points = tf.concat([medfi_points,highfi_points],0)
        medium_virtual_observations = tf.concat([tf.gather(dataset.observations, ind_medfi),self._medium_residual_model.predict(highfi_points)[0]+self._rho_low *self._low_fidelity_model.predict(highfi_points)[0]],0)
        self._virtual_medium_fidelity_model.update(Dataset(medium_virtual_query_points, medium_virtual_observations))

        high_residuals = tf.gather(dataset.observations, ind_highfi) - self._rho_medium * (self._medium_residual_model.predict(highfi_points)[0]+self._rho_low *self._low_fidelity_model.predict(highfi_points)[0])
        self._high_residual_model.update(Dataset(highfi_points, high_residuals))


    def log(self, dataset: Optional[Dataset] = None) -> None:
        """
        Log model-specific information at a given optimization step.

        :param dataset: Optional data that can be used to log additional data-based model summaries.
        """
        return

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize all the models on their corresponding data. The data for each model is
        extracted by splitting the observations in ``dataset``  by fidelity level.

        Note that we have to code up a custom loss function when optimizing our residual model, so that we
        can include the correlation parameter as an optimisation variable.

        :param dataset: The query points and observations for *all* the wrapped models.
        """
        lowfi_points, medfi_points, highfi_points, lowfi_mask, medfi_mask, highfi_mask, ind_lowfi, ind_medfi, ind_highfi = filter_by_fidelity(dataset.query_points)

        self._low_fidelity_model.optimize(Dataset(lowfi_points, tf.gather(dataset.observations, ind_lowfi)))

        medium_virtual_query_points = tf.concat([medfi_points,highfi_points],0)
        medium_virtual_observations = tf.concat([tf.gather(dataset.observations, ind_medfi),self._medium_residual_model.predict(highfi_points)[0]+self._rho_low *self._low_fidelity_model.predict(highfi_points)[0]],0)
        self._virtual_medium_fidelity_model.optimize(Dataset(medium_virtual_query_points, medium_virtual_observations))

        gpflow_medium_residual_model = self._medium_residual_model.model
        gpflow_high_residual_model = self._high_residual_model.model

        medium_fidelity_observations = tf.gather(dataset.observations, ind_medfi)
        high_fidelity_observations = tf.gather(dataset.observations, ind_highfi)
        predictions_from_low_fidelity = self._low_fidelity_model.predict(medfi_points)[0]
        #predictions_from_medium_fidelity = self._medium_residual_model.predict(highfi_points)[0]+self._rho_low *self._low_fidelity_model.predict(highfi_points)[0]
        predictions_from_medium_fidelity = self._virtual_medium_fidelity_model.predict(highfi_points)[0]

        def medium_loss(): # hard-coded log liklihood calculation for the residual model
            residuals = medium_fidelity_observations - self._rho_low * predictions_from_low_fidelity
            K = gpflow_medium_residual_model.kernel(medfi_points)
            #ks = gpflow_residual_model._add_noise_cov(K)
            L = tf.linalg.cholesky(K)
            m = gpflow_medium_residual_model.mean_function(medfi_points)
            log_prob = multivariate_normal(residuals, m, L)
            return -1.0 * tf.reduce_sum(log_prob)

        def high_loss(): # hard-coded log liklihood calculation for the residual model
            residuals = high_fidelity_observations - self._rho_medium * predictions_from_medium_fidelity
            K = gpflow_high_residual_model.kernel(highfi_points)
            #ks = gpflow_residual_model._add_noise_cov(K)
            L = tf.linalg.cholesky(K)
            m = gpflow_high_residual_model.mean_function(highfi_points)
            log_prob = multivariate_normal(residuals, m, L)
            return -1.0 * tf.reduce_sum(log_prob)

        medium_trainable_variables = gpflow_medium_residual_model.trainable_variables + self._rho_low.variables
        self._medium_residual_model.optimizer.optimizer.minimize(medium_loss, medium_trainable_variables)
        self._medium_residual_model.update(Dataset(medfi_points, medium_fidelity_observations - self._rho_low * predictions_from_low_fidelity))

        high_trainable_variables = gpflow_high_residual_model.trainable_variables + self._rho_medium.variables
        self._high_residual_model.optimizer.optimizer.minimize(high_loss, high_trainable_variables)
        self._high_residual_model.update(Dataset(highfi_points, high_fidelity_observations - self._rho_medium * predictions_from_medium_fidelity))


def filter_by_fidelity(query_points: TensorType):

    input_points = query_points[:, :-1]  # [..., D+1]
    fidelities = query_points[:, -1:]  # [..., 1]

    lowfi_mask = (fidelities[:, 0] == 0.)
    ind_lowfi = tf.where(lowfi_mask)[:, 0]

    medfi_mask = (fidelities[:, 0] == 1.)
    ind_medfi = tf.where(medfi_mask)[:, 0]

    highfi_mask = (fidelities[:, 0] == 2.)
    ind_highfi = tf.where(highfi_mask)[:, 0]

    lowfi_points = tf.gather(input_points, ind_lowfi, axis=0)
    medfi_points = tf.gather(input_points, ind_medfi, axis=0)
    highfi_points = tf.gather(input_points, ind_highfi, axis=0)
    return lowfi_points, medfi_points, highfi_points, lowfi_mask, medfi_mask, highfi_mask, ind_lowfi, ind_medfi, ind_highfi


#################################
# MUMBO acquisition function
#################################

##  This is an implementation of the MUMBO acqusiiton function (https://arxiv.org/pdf/2006.12093.pdf)
## suitiable for noisy problems with a single fidelity level
class MUMBO(trieste.acquisition.MinValueEntropySearch):

    def __repr__(self) -> str:
        return "MUMBO()"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer.
        :return: The max-value entropy search acquisition function modified for objective
            minimisation. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        min_value_samples = self.get_min_value_samples_on_top_fidelity(model, dataset)
        return trieste.acquisition.function.entropy.mumbo(model, min_value_samples)


    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        min_value_samples = self.get_min_value_samples_on_top_fidelity(model, dataset)
        function.update(min_value_samples)  # type: ignore
        return function

    def get_min_value_samples_on_top_fidelity(self, model: ProbabilisticModelType, dataset: Dataset):
        """
        :param model: The model.
        :param dataset: The data from the observer.
        """
        query_points = self._search_space.sample(num_samples=self._grid_size)
        tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
        query_points = tf.concat([dataset.query_points, query_points], 0)
        #query_points_on_top_fidelity = tf.concat([query_points[:,:-1],tf.ones_like(query_points[:,-1:])],-1)
        query_points_on_top_fidelity = tf.concat([query_points[:,:-1], tf.fill(tf.shape(query_points[:,-1:]), tf.reduce_max(query_points[:,-1]))], axis=-1) #Include n-levels
        num_samples=5 #self._num_samples
        return self._min_value_sampler.sample(model, num_samples, query_points_on_top_fidelity)




class mumbo_local(trieste.acquisition.function.min_value_entropy_search):

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        #x_on_top_fidelity = tf.concat([tf.squeeze(x, -2)[:,:-1],2*tf.ones_like(tf.squeeze(x,-2)[:,-1:])],-1)
        x_squeezed = tf.squeeze(x, -2)
        x_on_top_fidelity = add_fidelity_column(x_squeezed[:, :-1], 2)
        print(x_on_top_fidelity)
        fmean, fvar = self._model.predict(x_on_top_fidelity)
        fsd = tf.math.sqrt(fvar)
        ymean, yvar = self._model.predict_y(tf.squeeze(x,-2))
        cov = self._model.covariance_with_top_fidelity(tf.squeeze(x, -2))
        correlations = cov / tf.math.sqrt(fvar*yvar)
        correlations = tf.clip_by_value(correlations,-1.0,1.0)

        # Calculate moments of extended skew Gaussian distributions (ESG)
        # These will be used to define reasonable ranges for the numerical
        # intergration of the ESG's differential entropy.
        gamma = (tf.squeeze(self._samples) - fmean) / fsd
        normal = tfp.distributions.Normal(tf.cast(0, fmean.dtype), tf.cast(1, fmean.dtype))
        log_minus_cdf = normal.log_cdf(-gamma)
        ratio = tf.math.exp(normal.log_prob(gamma) - log_minus_cdf)
        ESGmean = correlations * ratio
        ESGvar = 1 + correlations * ESGmean * (gamma - ratio)
        ESGvar = tf.math.maximum(ESGvar, 0) # Clip  to improve numerical stability

        # get upper limits for numerical integration
        # we need this range to contain almost all of the ESG's probability density
        # we found +-5 standard deviations provides a tight enough approximation
        upper_limit = ESGmean + 5 * tf.math.sqrt(ESGvar)
        lower_limit = ESGmean - 5 * tf.math.sqrt(ESGvar)

        # perform numerical integrations
        z = tf.linspace(lower_limit, upper_limit, num=1000) # build discretisation
        minus_correlations = tf.math.sqrt(1 - correlations**2) # calculate ESG density at these points
        minus_correlations = tf.math.maximum(minus_correlations, 1e-10) # clip below for numerical stability


        density = tf.math.exp(normal.log_prob(z) - log_minus_cdf + normal.log_cdf(-(gamma-correlations*z)/minus_correlations))
        # calculate point-wise entropy function contributions (carefuly where density is 0)
        entropy_function = - density * tf.where(density!=0, tf.math.log(density), 0.0)
        approximate_entropy = tfp.math.trapz(entropy_function, z, axis=0) # perform integration over ranges

        approximate_entropy = tf.reduce_mean(approximate_entropy, axis=-1) # build monte-carlo estimate over the gumbel samples
        f_acqu_x = tf.cast(0.5 * tf.math.log(2.0 * math.pi * math.e), tf.float64) - approximate_entropy
        return f_acqu_x[:,None]



class CostWeighting(SingleModelAcquisitionBuilder):

    def __init__(self, low_fidelity_cost, medium_fidelity_cost, high_fidelity_cost):
        self._low_fidelity_cost = low_fidelity_cost
        self._medium_fidelity_cost = medium_fidelity_cost
        self._high_fidelity_cost = high_fidelity_cost

    def prepare_acquisition_function(self, model, dataset=None):
        def acquisition(x):
            tf.debugging.assert_shapes(
                [(x, [..., 1, None])],
                message="This acquisition function only supports batch sizes of one.",
            )
            fidelities = x[..., -1:]  # [..., 1]
            costs = tf.where(fidelities == 0., self._low_fidelity_cost, tf.where(fidelities == 1., self._medium_fidelity_cost, self._high_fidelity_cost))
            return tf.cast(1.0 / costs, x.dtype)[:,0,:]
        return acquisition

    def update_acquisition_function(self, function, model, dataset=None):
        return function



#################################
# definition of observer
#################################

import subprocess

# Replace this with your own observer
# 2-LEVEL-Simulator
# def linear_simulator(x_input, fidelity, add_noise=False):
#     fe = ((6.0 * x_input - 2.0) ** 2) * tf.math.sin(
#           12.0 * x_input - 4.0)
#     fc = 0.5 * fe + 10 *(x_input - 0.5) - 5
#     if add_noise:
#         noise = tf.random.normal(f.shape, stddev=1e-1, dtype=f.dtype)
#     else:
#         noise = 0
#
#     f = tf.where(fidelity > 0, fe, fc)
#     print(x_input)
#     return f

# n-level-simulator
def linear_simulator(x_input, fidelity, add_noise=False):
    f = 0.5 * ((6.0 * x_input - 2.0) ** 2) * tf.math.sin(
        12.0 * x_input - 4.0
    ) + 10.0 * (x_input - 1.0)
    f = f + fidelity * (f - 20.0 * (x_input - 1.0))
    if add_noise:
        noise = tf.random.normal(f.shape, stddev=1e-1, dtype=f.dtype)
    else:
        noise = 0
    f = tf.where(fidelity > 0, f + noise, f)
    return f


def observer(x):

    # If nested case :
    # if x.get_shape()[0]!=1 :
    #     mf_rows = tf.boolean_mask(x, tf.equal(x[:, 1], 1.0)) # select rows with fidelity=1
    #     switch_mbf = tf.concat([mf_rows[:, :1], tf.zeros_like(mf_rows[:, 1:])], axis=1) # Switch them to fidelity=0
    #     hf_rows = tf.boolean_mask(x, tf.equal(x[:, 1], 2.0)) # select rows with fidelity=1
    #     switch_hbf = tf.concat([hf_rows[:, :1], tf.zeros_like(hf_rows[:, 1:])], axis=1) # Switch them to fidelity=0
    #     switch_hmf = tf.concat([hf_rows[:, :1], tf.ones_like(hf_rows[:, 1:])], axis=1) # Switch them to fidelity=1
    #     x = tf.concat([x, switch_mbf,switch_hbf,switch_hmf], axis=0) # Add the switched values to the vector x (Nested case : include hf points in bf dataset)

    # last dimension is the fidelity value
    input = x[..., :-1]
    fidelity = x[..., -1:]

    # note: this assumes that my_simulator broadcasts, i.e. accept matrix inputs.
    # If not you need to replace this by a for loop over all rows of "input"

    # observations = my_simulator(input, fidelity)
    observations = linear_simulator(input, fidelity)

    # Print only EGO phase :
    if int(tf.shape(x)[...,0]) == 1 :
        print(f"level: {fidelity[:]}")
        print(f"x: \n {input[:]}")
        print(f" value: \n {observations[:]}")

    return trieste.data.Dataset(x, observations)


#################################
# random my_seed
#################################

np.random.seed(my_seed)
tf.random.set_seed(my_seed)


#################################
# optimization
#################################
tic = time.time()

input_search_space = trieste.space.Box(lb, ub)
fidelity_search_space = trieste.space.DiscreteSearchSpace(np.array([0., 1., 2.]).reshape(-1, 1))
search_space = trieste.space.TaggedProductSearchSpace([input_search_space, fidelity_search_space],
                                                      ["input", "fidelity"])
#Non-nested initial data : ---------------------------------------------------------------------
X = input_search_space.sample_sobol(init_low)
initial_sample_low = tf.concat([X,tf.zeros([init_low,1],tf.double)],1)
X =input_search_space.sample_sobol(init_medium)
initial_sample_medium = tf.concat([X,tf.ones([init_medium,1],tf.double)],1)
X =input_search_space.sample_sobol(init_high)
initial_sample_high = tf.concat([X,2*tf.ones([init_high,1],tf.double)],1)
initial_sample = tf.concat([initial_sample_low,initial_sample_medium,initial_sample_high],0)

# #Nested initial data : --------------------------------------------------------------------------
# np.random.seed(35)
# tf.random.set_seed(35)
#
# sample_sizes = [init_low, init_medium, init_high]
# xs = [tf.linspace(0, 1, sample_sizes[0])[:, None]]
# for fidelity in range(1, n_fidelities):
#     samples = tf.Variable(
#         np.random.choice(
#             xs[fidelity - 1][:, 0], size=sample_sizes[fidelity], replace=False
#         )
#     )[:, None]
#     xs.append(samples)
# initial_samples_list = [add_fidelity_column(x, i) for i, x in enumerate(xs)]
# initial_sample = tf.concat(initial_samples_list, 0)
# #-------------------------------------------------------------------------------------------------

initial_data = observer(initial_sample)
num_fidelities=int(tf.reduce_max(initial_sample[:,-1]).numpy()+1)
# print(f"#### DOE phase :")
# print(f"level: {initial_sample[..., -1:]}")
# print(f"x: \n {initial_sample[..., :-1]}")
# print(f"observations : \n {initial_data.observations}")
likelihood_value=1e-10

lowfi_points, medfi_points, highfi_points, lowfi_mask, medfi_mask, highfi_mask, ind_lowfi, ind_medfi, ind_highfi = filter_by_fidelity(initial_data.query_points)

lf_data = Dataset(lowfi_points, tf.gather(initial_data.observations, ind_lowfi))#
low_fidelity_gpr = GaussianProcessRegression(build_gpr(lf_data, input_search_space,  likelihood_variance = likelihood_value, kernel_priors=True))

mf_data = Dataset(medfi_points, tf.gather(initial_data.observations, ind_medfi))#
medium_fidelity_gpr = GaussianProcessRegression(build_gpr(mf_data, input_search_space,  likelihood_variance = likelihood_value, kernel_priors=True))

low_virtual_query_points = tf.concat([lowfi_points,medfi_points],0)
low_virtual_observations = tf.concat([tf.gather(initial_data.observations, ind_lowfi),low_fidelity_gpr.predict(medfi_points)[0]],0)

vlf_data = Dataset(low_virtual_query_points, low_virtual_observations)#
virtual_low_fidelity_gpr = GaussianProcessRegression(build_gpr(vlf_data, input_search_space,  likelihood_variance = likelihood_value, kernel_priors=True))

medium_virtual_query_points = tf.concat([medfi_points,highfi_points],0)
medium_virtual_observations = tf.concat([tf.gather(initial_data.observations, ind_medfi),low_fidelity_gpr.predict(highfi_points)[0]],0)

vmf_data = Dataset(medium_virtual_query_points, medium_virtual_observations)#
virtual_medium_fidelity_gpr = GaussianProcessRegression(build_gpr(vmf_data, input_search_space,  likelihood_variance = likelihood_value, kernel_priors=True))

medium_residual_data = Dataset(medfi_points, tf.gather(initial_data.observations, ind_medfi) - low_fidelity_gpr.predict(medfi_points)[0])
medium_residual_gpr = GaussianProcessRegression(build_gpr(medium_residual_data, input_search_space, likelihood_variance = likelihood_value, kernel_priors=True)) # ignore this


#high_residual_data = Dataset(highfi_points, tf.gather(initial_data.observations, ind_highfi) - (medium_residual_gpr.predict(highfi_points)[0]+low_fidelity_gpr.predict(highfi_points)[0]))
high_residual_data = Dataset(highfi_points, tf.gather(initial_data.observations, ind_highfi) - medium_fidelity_gpr.predict(highfi_points)[0])
high_residual_gpr = GaussianProcessRegression(build_gpr(high_residual_data, input_search_space, likelihood_variance = likelihood_value, kernel_priors=True)) # ignore this

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

acq_builder = Product( MUMBO(search_space).using("OBJECTIVE"), CostWeighting(low_cost, medium_cost, high_cost).using("OBJECTIVE"))
optimizer =  generate_continuous_optimizer(
    num_initial_samples=10_000,
    num_optimization_runs= 10) # run multiple gradient opts from each of K best points from initial sample
rule = trieste.acquisition.rule.EfficientGlobalOptimization(builder=acq_builder)

model = MultiFidelityModel(
    low_fidelity_model = low_fidelity_gpr,
    virtual_low_fidelity_model=virtual_low_fidelity_gpr,
    medium_residual_model = medium_residual_gpr,
    virtual_medium_fidelity_model=virtual_medium_fidelity_gpr,
    high_residual_model=high_residual_gpr,
    num_fidelities=num_fidelities
)


model.update(initial_data)
model.optimize(initial_data)
#trieste.models.utils.optimize_model_and_save_result(model, initial_data)

result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)

dataset = result.try_get_final_dataset()
multifidelity_model = result.try_get_final_model()
toc = time.time()

elapsed=toc-tic
print("Elapsed time: " + time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:15], time.gmtime(elapsed)))
#################################
# New plots
#################################
data = split_dataset_by_fidelity(dataset, num_fidelities=num_fidelities)


X = tf.linspace(0, 1, 200)[:, None]
X_list = [add_fidelity_column(X, i) for i in range(num_fidelities)]
predictions = [multifidelity_model.predict(x) for x in X_list]

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

#pred_colors = ["tab:blue", "tab:orange", "tab:green"]
gt_colors = ["tab:red", "tab:brown", "tab:purple"]
pred_colors=gt_colors

#plt.figure(10)
for fidelity, prediction in enumerate(predictions):

    if fidelity==0 :
        ax.plot(
            X,
            observer(X_list[fidelity]).observations[:200],
            label=f"True function LF",
            color=gt_colors[fidelity],
        )

        ax.scatter(
            data[fidelity].query_points,
            data[fidelity].observations,
            label=f"data LF",
            color=gt_colors[fidelity],
        )

        mean, var = prediction
        ax.plot(
            X,
            mean, '--',
            label=f"Predicted LF",
            color=pred_colors[fidelity],
        )
        ax.plot(
            X,
            mean + 1.96 * tf.math.sqrt(var),
            alpha=0.2,
            color=pred_colors[fidelity],
        )
        ax.plot(
            X,
            mean - 1.96 * tf.math.sqrt(var),
            alpha=0.2,
            color=pred_colors[fidelity],

        )

    elif fidelity==1 :
        ax.plot(
            X,
            observer(X_list[fidelity]).observations[:200],
            label=f"True function MF",
            color=gt_colors[fidelity],
        )
        ax.scatter(
            data[fidelity].query_points,
            data[fidelity].observations,
            label=f"data MF",
            color=gt_colors[fidelity],
        )
        mean, var = prediction
        ax.plot(
            X,
            mean, '--',
            label=f"Predicted MF",
            color=pred_colors[fidelity],
        )
        ax.plot(
            X,
            mean + 1.96 * tf.math.sqrt(var),
            alpha=0.2,
            color=pred_colors[fidelity],
        )
        ax.plot(
            X,
            mean - 1.96 * tf.math.sqrt(var),
            alpha=0.2,
            color=pred_colors[fidelity],
        )

    else :
        ax.plot(
            X,
            observer(X_list[fidelity]).observations[:200],
            label=f"True function HF",
            color=gt_colors[fidelity],
        )
        ax.scatter(
            data[fidelity].query_points,
            data[fidelity].observations,
            label=f"data HF",
            color=gt_colors[fidelity],
        )
        mean, var = prediction
        ax.plot(
            X,
            mean, '--',
            label=f"Predicted HF",
            color=pred_colors[fidelity],
        )
        ax.plot(
            X,
            mean + 1.96 * tf.math.sqrt(var),
            alpha=0.2,
            color=pred_colors[fidelity],
        )
        ax.plot(
            X,
            mean - 1.96 * tf.math.sqrt(var),
            alpha=0.2,
            color=pred_colors[fidelity],
        )

ax.set_ylim([-11, 25])
ax.legend()#loc="center left", bbox_to_anchor=(1, 0.5))

# iterations=np.array([i for i in range(1,num_steps+1)])
# fidelities_EGO=dataset.query_points[init_low+init_medium+init_high:, 1].numpy()
# fig, ax = plt.subplots(1, 1, figsize=(10, 7))
# ax.plot(iterations,fidelities_EGO,'x')

plt.show()



#################################
# some plots
#################################
fidelities_EGO=dataset.query_points[init_low+init_medium+init_high:, 1].numpy()
print(f"EGO phase per fidelities: {fidelities_EGO}")


lowfi_points, medfi_points, highfi_points, lowfi_mask, medfi_mask, highfi_mask, ind_lowfi, ind_medfi, ind_highfi = filter_by_fidelity(dataset.query_points)
highfi_obs = tf.gather(dataset.observations, ind_highfi)
arg_min_idx = tf.squeeze(tf.argmin(highfi_obs, axis=0))

print(f"hi-fi point: {highfi_points[arg_min_idx, :]}")
print(f"observation: {highfi_obs[arg_min_idx, :]}")

medfi_obs = tf.gather(dataset.observations, ind_medfi)
arg_min_idx = tf.squeeze(tf.argmin(medfi_obs, axis=0))

print(f"mi-fi point: {medfi_points[arg_min_idx, :]}")
print(f"observation: {medfi_obs[arg_min_idx, :]}")

lowfi_obs = tf.gather(dataset.observations, ind_lowfi)
arg_min_idx = tf.squeeze(tf.argmin(lowfi_obs, axis=0))

print(f"low-fi point: {lowfi_points[arg_min_idx, :]}")
print(f"observation: {lowfi_obs[arg_min_idx, :]}")
