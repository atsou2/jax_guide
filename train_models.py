import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Int, PRNGKeyArray
from generate_data import PendulumSimulation
import optax
from models import CNNEmulator



def loss_fn(model, batch):
    x, y = batch
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)

def train(
    model: CNNEmulator,
    dataset: Float[Array, " n_samples n_res n_res"],
    batch_size: Int,
    learning_rate: Float,
    num_epochs: Int,
    key: PRNGKeyArray,
) -> CNNEmulator:
    
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: CNNEmulator,
        opt_state: optax.OptState,
        batch: Float[Array, " n_samples n_res n_res"],
    ) -> tuple:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    print("Training...")
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            model, opt_state, loss = make_step(model, opt_state, batch)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    return model

IMAGE_SIZE = 64

pendulum = PendulumSimulation(image_size=IMAGE_SIZE)
dataset = pendulum.generate_dataset(5, 9.8, 1.0)

CNNmodel = CNNEmulator(jax.random.PRNGKey(0))
trained_CNNmodel = train(CNNmodel, dataset, 4, 1e-3, 300, jax.random.PRNGKey(1))
CNNresult = trained_CNNmodel.rollout(dataset[0][0], len(dataset[0][0]))
