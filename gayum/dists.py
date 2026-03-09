from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax


class Dist(ABC):
    """Abstract base class for distributions in GAMs"""
    
    @abstractmethod
    def log_likelihood(self, y: jax.Array, mu: jax.Array) -> jax.Array:
        pass
    
    @abstractmethod
    def link(self, mu: jax.Array) -> jax.Array:
        pass
    
    @abstractmethod
    def inverse_link(self, eta: jax.Array) -> jax.Array:
        pass
    
    @abstractmethod
    def variance(self, mu: jax.Array) -> jax.Array:
        pass
    
    def deviance(self, y: jax.Array, mu: jax.Array) -> jax.Array:
        return -2 * self.log_likelihood(y, mu)


class Normal(Dist):
    """Normal (Gaussian) distribution with identity link"""
    
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
    
    def log_likelihood(self, y: jax.Array, mu: jax.Array) -> jax.Array:
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * self.sigma**2) - 
                      (y - mu)**2 / (2 * self.sigma**2))
    
    def link(self, mu: jax.Array) -> jax.Array:
        return mu
    
    def inverse_link(self, eta: jax.Array) -> jax.Array:
        return eta
    
    def variance(self, mu: jax.Array) -> jax.Array:
        return jnp.ones_like(mu) * self.sigma**2


class Binomial(Dist):
    """Binomial distribution with logit link"""
    
    def log_likelihood(self, y: jax.Array, mu: jax.Array) -> jax.Array:
        mu = jnp.clip(mu, 1e-8, 1 - 1e-8) # clip to avoid log(0)
        return jnp.sum(y * jnp.log(mu) + (1 - y) * jnp.log(1 - mu))
    
    def link(self, mu: jax.Array) -> jax.Array:
        mu = jnp.clip(mu, 1e-8, 1 - 1e-8)
        return jnp.log(mu / (1 - mu))
    
    def inverse_link(self, eta: jax.Array) -> jax.Array:
        return 1 / (1 + jnp.exp(-eta))
    
    def variance(self, mu: jax.Array) -> jax.Array:
        return mu * (1 - mu)


class Poisson(Dist):
    """Poisson distribution with log link"""
    
    def log_likelihood(self, y: jax.Array, mu: jax.Array) -> jax.Array:
        log_factorial_y = jnp.where(y > 10, 
                                   y * jnp.log(y) - y + 0.5 * jnp.log(2 * jnp.pi * y),
                                   jax.scipy.special.gammaln(y + 1))
        return jnp.sum(y * jnp.log(mu) - mu - log_factorial_y)
    
    def link(self, mu: jax.Array) -> jax.Array:
        return jnp.log(jnp.maximum(mu, 1e-8))
    
    def inverse_link(self, eta: jax.Array) -> jax.Array:
        return jnp.exp(eta)
    
    def variance(self, mu: jax.Array) -> jax.Array:
        return mu
