import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Int, PRNGKeyArray

class CNNEmulator(eqx.Module):
    layers: list
    encoder: list
    decoder: list
    n_bottle_pixels: int

    def __init__(self, key: PRNGKeyArray, hidden_dim: Int = 4, n_layers: Int = 2):
        self.layers = []
        self.encoder = [
            eqx.nn.Conv2d(in_channels=2, out_channels=hidden_dim, kernel_size=1, key=key),
            eqx.nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, key=key)
        ]
        self.n_bottle_pixels = hidden_dim * hidden_dim

        decoder = []
        key, subkey = jax.random.split(key)

        # The decoder should expect hidden_dim channels from the encoder
        for i in range(n_layers):
            key, subkey = jax.random.split(key)
            decoder.append(eqx.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=[1,1], key=subkey))
            decoder.append(eqx.nn.Lambda(jnp.tanh))

        # Final layer to reduce back to 1 channel
        key, subkey = jax.random.split(key)
        decoder.append(eqx.nn.ConvTranspose2d(hidden_dim, 1, 3, padding=[1,1], key=subkey))
        self.decoder = decoder
        self.layers = self.encoder + self.decoder

    def __call__(self, x: Float[Array, "2 n_res n_res"]) -> Float[Array, "1 n_res n_res"]:
        x = self.encode(self.encoder, x)
        return self.decode(x)
    
    @staticmethod
    def encode(model: list, x: Float[Array, "2 n_res n_res"]) -> Float[Array, " n_bottleneck"]:
        for layer in model:
            x = layer(x)
        return x

    def decode(self, z: Float[Array, " n_bottleneck"]) -> Float[Array, " n_res n_res"]:
        for layer in self.decoder:
            z = layer(z)
        return z

    def rollout(self, x: Float[Array, "2 n_res n_res"], n_steps: int) -> Float[Array, "1 n_res n_res"]:
        for _ in range(n_steps):
            x = jnp.concatenate([x[1:], self(x)], axis=0)
        return x