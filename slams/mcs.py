r"""We adapt codes from Score-based Data Assimilation (https://arxiv.org/abs/2306.10574)"""

import abc
import numpy as np
import math
import random
import seaborn as sns
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as rng
import jax_cfd.base as cfd

import torch
from torch import Tensor, Size
from torch.distributions import Normal, MultivariateNormal, Uniform
from torch_harmonics.sht import *
from torch_harmonics.examples import ShallowWaterSolver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from typing import *


class MarkovChain(abc.ABC):
    r"""Abstract first-order time-invariant Markov chain class

    Wikipedia:
        https://wikipedia.org/wiki/Markov_chain
        https://wikipedia.org/wiki/Time-invariant_system
    """

    @abc.abstractmethod
    def prior(self, shape: Size = ()) -> Tensor:
        r""" x_0 ~ p(x_0) """

        pass

    @abc.abstractmethod
    def transition(self, x: Tensor) -> Tensor:
        r""" x_i ~ p(x_i | x_{i-1}) """

        pass

    def trajectory(self, x: Tensor, length: int, last: bool = False) -> Tensor:
        r""" (x_1, ..., x_n) ~ \prod_i p(x_i | x_{i-1}) """

        if last:
            for _ in range(length):
                x = self.transition(x)

            return x
        else:
            X = []

            for _ in range(length):
                x = self.transition(x)
                X.append(x)

            return torch.stack(X)

class DiscreteODE(abc.ABC):
    r"""Discretized ordinary differential equation (ODE)

    Wikipedia:
        https://wikipedia.org/wiki/Ordinary_differential_equation
    """

    def __init__(self, dt: float = 0.01):
        super().__init__()
        self.dt = dt

    @staticmethod
    def rk4(f: Callable[[Tensor], Tensor], x: Tensor, dt: float) -> Tensor:
        r"""Performs a step of the fourth-order Runge-Kutta integration scheme.

        Wikipedia:
            https://wikipedia.org/wiki/Runge-Kutta_methods
        """

        k1 = f(x)
        k2 = f(x + dt * k1 / 2)
        k3 = f(x + dt * k2 / 2)
        k4 = f(x + dt * k3)

        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def transition(self, x: Tensor) -> Tensor:
        return self.rk4(self.f, x, self.dt)


class NoisyLorenz63(DiscreteODE):
    r"""Noisy Lorenz 1963 dynamics

    Wikipedia:
        https://wikipedia.org/wiki/Lorenz_system
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0, 
        beta: float = 8 / 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    # Define prior
    def prior(self, shape: Size = ()) -> Tensor:
        self.shape = shape
        mu = torch.tensor([0.0, 0.0, 25.0])
        sigma = torch.tensor([
            [64.0, 50.0,  0.0],
            [50.0, 81.0,  0.0],
            [ 0.0,  0.0, 75.0],
        ])

        return MultivariateNormal(mu, sigma).sample(shape)
    
    # Define transition matrix
    def f(self, x: Tensor) -> Tensor:
        return torch.stack((
            self.sigma * (x[..., 1] - x[..., 0]),
            x[..., 0] * (self.rho - x[..., 2]) - x[..., 1],
            x[..., 0] * x[..., 1] - self.beta * x[..., 2],
        ), dim=-1)
    
    
    # Generate noisy trajectory (RK-4 forward integration) given transition matrix
    def trajectory(self, x: Tensor, length: int, last: bool = False) -> Tensor:
        r""" (x_1, ..., x_n) ~ \prod_i p(x_i | x_{i-1}) """
        if last:
            for _ in range(length):
                x = self.transition(x)

            return x
        else:
            X = []

            for _ in range(length):
                x = self.transition(x)
                X.append(x)

            return torch.stack(X)

    def transition(self, x: Tensor) -> Tensor:
        # mu_x, sigma_x = super().transition(x), self.dt ** 0.5
        # return Normal(mu_x, sigma_x).sample()
        
        return super().transition(x)
    
    
    def log_prob(self, x1: Tensor, x2: Tensor) -> Tensor:
        return Normal(*self.moments(x1)).log_prob(x2).sum(dim=-1)

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        mu = x.new_tensor([0.0, 0.0, 25.0])
        sigma = x.new_tensor([8.0, 9.0, 8.6])

        return (x - mu) / sigma

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        mu = x.new_tensor([0.0, 0.0, 25.0])
        sigma = x.new_tensor([8.0, 9.0, 8.6])

        return mu + sigma * x
    

class ShallowWaterEquation(DiscreteODE):
    r"""Linearized shallow water equation as implemented with spherical spectral solver.
    
    Wikipedia:
        https://en.wikipedia.org/wiki/Shallow_water_equations
    """
    
    def __init__(
        self, 
        size = (120, 240),
        dt: int = 60,
        hamp: float = 120.,
        **kwargs
    ):

        super().__init__(**kwargs)
        
        self.size = size
        self.dt = dt
        self.hamp = hamp
        
        self.solver = ShallowWaterSolver(size[0], size[1], dt, lmax=size[0], mmax=size[0], hamp=hamp).to(device)

    def prior(self, ic='galewsky'):
        """
        Initialize solver and initial parameters
        
        - Galewsky et al, 2004
            Initial perturbation of geostrophically-balanced mid-latitude jet stream in a rotating sphere.
            It creates an IVP that gives rise to Barotropic Instability.
        
        """
        
        assert ic in ['galewsky', 'random']
        
        if ic == 'galewsky':
            self.u_spec0 = self.solver.galewsky_initial_condition()
        else:
            self.u_spec0 = self.solver.random_initial_condition()

    
    def trajectory(self, num_steps: int, nskip: int = 50) -> Tensor:
        """Perform time-step integration in the spectral domain according to specification described in torch-harmonics
        """
        u_spec = self.u_spec0.clone().to(device)
        ut_spec = torch.zeros(math.floor(num_steps // nskip) + 1, *u_spec.shape).cdouble().to(device)
        dudt_spec = torch.zeros(3, 3, self.solver.lmax, self.solver.mmax, dtype=torch.complex128, device=device)
        i_new, i_now, i_old = 0, 1, 2

        with torch.inference_mode():
            for i in range(num_steps + 1):
                t = i * self.dt

                if i % nskip == 0:
                    ut_spec[i // nskip] = u_spec

                dudt_spec[i_new] = self.solver.dudtspec(u_spec)

                # 3rd-order Adams-Bashforth time-stepping
                if i == 0:
                    dudt_spec[i_now] = dudt_spec[i_new]
                    dudt_spec[i_old] = dudt_spec[i_new]

                elif i == 1:
                    dudt_spec[i_old] = dudt_spec[i_new]

                u_spec = u_spec + self.dt * ((23./12.) * dudt_spec[i_new] - (16./12.) * dudt_spec[i_now] + (5./12.) * dudt_spec[i_old])

                # Implicit hyperdiffusion for vorticity and divergence to smooth out small structures: numerical stability
                u_spec[1:] = self.solver.hyperdiff * u_spec[1:]

                # Cycle through the indices
                i_new, i_now, i_old = (i_new - 1) % 3, (i_now - 1) % 3, (i_old - 1) % 3
                
        return ut_spec
    
    def spec2grid(self, x: Tensor) -> Tensor:
        return self.solver.spec2grid(x)
    
    def plot_sphere(self, x: Tensor) -> Tensor:
        f = plt.figure(figsize=(2,2))
        
        if x.is_cuda:
            im = self.solver.plot_griddata(x, f, cmap=sns.cm.icefire)
        else:
            im = self.solver.cpu().plot_griddata(x, f, cmap=sns.cm.icefire)
        
        return im


class KolmogorovFlow(MarkovChain):
    r"""2-D fluid dynamics with Kolmogorov forcing

    Wikipedia:
        https://wikipedia.org/wiki/Navier-Stokes_equations
    """

    def __init__(
        self,
        size: int = 256,
        dt: float = 0.01,
        reynolds = None
    ):
        super().__init__()
        
        self.size = size
        self.dt = dt
        
        reynolds = torch.tensor(1e3) if reynolds == None else reynolds

        grid = cfd.grids.Grid(
            shape=(size, size),
            domain=((0, 2 * math.pi), (0, 2 * math.pi)),
        )

        bc = cfd.boundaries.periodic_boundary_conditions(2)

        forcing = cfd.forcings.simple_turbulence_forcing(
            grid=grid,
            constant_magnitude=1.0,
            constant_wavenumber=4.0,
            linear_coefficient=-0.1,
            forcing_type='kolmogorov',
        )
        
        dt_min = cfd.equations.stable_time_step(
            grid=grid,
            max_velocity=5.0,
            max_courant_number=0.5,
            viscosity=1 / reynolds.item(),
        )

        if dt_min > dt:
            steps = 1
        else:
            steps = math.ceil(dt / dt_min)

        step = cfd.funcutils.repeated(
            f=cfd.equations.semi_implicit_navier_stokes(
                grid=grid,
                forcing=forcing,
                dt=dt / steps,
                density=1.0,
                viscosity=1 / reynolds.item(),
            ),
            steps=steps,
        )

        def prior(key: rng.PRNGKey) -> jax.Array:
            u, v = cfd.initial_conditions.filtered_velocity_field(
                key,
                grid=grid,
                maximum_velocity=3.0,
                peak_wavenumber=4.0,
            )

            return jnp.stack((u.data, v.data))

        def transition(uv: jax.Array) -> jax.Array:
            u, v = cfd.initial_conditions.wrap_variables(
                var=tuple(uv),
                grid=grid,
                bcs=(bc, bc),
            )

            u, v = step((u, v))

            return jnp.stack((u.data, v.data))

        self._prior = jax.jit(jnp.vectorize(prior, signature='(K)->(C,H,W)'))
        self._transition = jax.jit(jnp.vectorize(transition, signature='(C,H,W)->(C,H,W)'))

    def prior(self, shape: Size = ()) -> Tensor:
        seed = random.randrange(2**32)

        key = rng.PRNGKey(seed)
        keys = rng.split(key, Size(shape).numel())
        keys = keys.reshape(*shape, -1)
        
        x = self._prior(keys)
        x = torch.tensor(np.asarray(x))

        return x

    def transition(self, x: Tensor) -> Tensor:
        x = x.detach().cpu().numpy()
        x = self._transition(x)
        x = torch.tensor(np.asarray(x))

        return x

    @staticmethod
    def coarsen(x: Tensor, r: int = 2) -> Tensor:
        *batch, h, w = x.shape

        x = x.reshape(*batch, h // r, r, w // r, r)
        x = x.mean(dim=(-3, -1))

        return x

    @staticmethod
    def upsample(x: Tensor, r: int = 2, mode: str = 'bilinear') -> Tensor:
        *batch, h, w = x.shape

        x = x.reshape(-1, 1, h, w)
        x = torch.nn.functional.pad(x, pad=(1, 1, 1, 1), mode='circular')
        x = torch.nn.functional.interpolate(x, scale_factor=(r, r), mode=mode)
        x = x[..., r:-r, r:-r]
        x = x.reshape(*batch, r * h, r * w)

        return x

    @staticmethod
    def vorticity(x: Tensor) -> Tensor:
        *batch, _, h, w = x.shape

        y = x.reshape(-1, 2, h, w)
        y = torch.nn.functional.pad(y, pad=(1, 1, 1, 1), mode='circular')

        du, = torch.gradient(y[:, 0], dim=-1)
        dv, = torch.gradient(y[:, 1], dim=-2)

        y = du - dv
        y = y[:, 1:-1, 1:-1]
        y = y.reshape(*batch, h, w)

        return y
