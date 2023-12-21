r"""Markov chains"""

import abc
import numpy as np
import math
import random
import torch
import jax
import jax.numpy as jnp
import jax.random as rng
import jax_cfd.base as cfd
from torch import Tensor, Size
from torch.distributions import Normal, MultivariateNormal
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

        self.sigma, self.rho, self.beta = sigma, rho, beta
    
    # Define prior
    def prior(self, shape: Size = ()) -> Tensor:
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
        mu_x, sigma_x = super().transition(x), self.dt ** 0.5
        return Normal(mu_x, sigma_x).sample()
    
    
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

class KolmogorovFlow(MarkovChain):
    r"""2-D fluid dynamics with Kolmogorov forcing

    Wikipedia:
        https://wikipedia.org/wiki/Navier-Stokes_equations
    """

    def __init__(
        self,
        size: int = 256,
        dt: float = 0.01,
        reynolds: int = 1e3,
    ):
        super().__init__()

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
            viscosity=1 / reynolds,
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
                viscosity=1 / reynolds,
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
