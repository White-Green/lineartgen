"""Training-only conditional patch critic; the exported generator is unchanged."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class PatchCritic(nnx.Module):
    def __init__(self, *, rngs):
        widths = (3, 8, 16, 32)
        self.layers = nnx.List(
            [
                nnx.Conv(a, b, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
                for a, b in zip(widths[:-1], widths[1:], strict=True)
            ]
        )
        self.head = nnx.Conv(widths[-1], 1, (3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x):
        for layer in self.layers:
            x = jax.nn.leaky_relu(layer(x), negative_slope=0.2)
        return self.head(x).mean((1, 2, 3))


def patch_coordinates(clean, *, seed, step, size=64):
    """Two ink, one paper, one uniform patch per image; same coordinates for real/fake."""
    batch, height, width, _ = clean.shape
    if min(height, width) < size:
        raise ValueError("Patch size exceeds the training image")
    result = np.empty((batch, 4, 2), np.int32)
    for item, image in enumerate(clean):
        rng = np.random.default_rng([seed, step, item, 0x43524954])
        ink = np.argwhere(image[..., 0] < 0.9)
        paper = np.argwhere(image[..., 0] >= 0.99)
        for index, points in enumerate((ink, ink, paper, None)):
            if points is not None and len(points):
                center = points[int(rng.integers(len(points)))]
            else:
                center = np.array([rng.integers(height), rng.integers(width)])
            result[item, index] = np.clip(center - size // 2, 0, [height - size, width - size])
    return result


def extract_patches(images, coordinates, size):
    def sample(image, points):
        return jax.vmap(
            lambda p: jax.lax.dynamic_slice(image, (p[0], p[1], 0), (size, size, image.shape[-1]))
        )(points)

    patches = jax.vmap(sample)(images, coordinates)
    return patches.reshape(-1, size, size, images.shape[-1])


def critic_input(prediction, prepared, coordinates, size, min_noise):
    truth, noise, noisy = prepared.targets[0], prepared.noise_levels[0], prepared.noisy[0]
    active = noise > min_noise
    candidate = jnp.where(active, prediction, jax.lax.stop_gradient(truth))
    images = jnp.concatenate((jnp.clip(noisy, -3, 3) / 3, noise, candidate), axis=-1)
    patches = extract_patches(images, coordinates, size)
    weights = extract_patches(active.astype(jnp.float32), coordinates, size).mean((1, 2, 3))
    return patches, weights


def weighted_mean(values, weights):
    return (values * weights).sum() / jnp.maximum(weights.sum(), 1)


def generator_prior(critic, prediction, prepared, coordinates, size, min_noise):
    fake, weights = critic_input(prediction, prepared, coordinates, size, min_noise)
    return weighted_mean(jax.nn.softplus(-critic(fake)), weights)


def discriminator_loss(critic, prediction, prepared, coordinates, size, min_noise):
    fake, weights = critic_input(
        jax.lax.stop_gradient(prediction), prepared, coordinates, size, min_noise
    )
    real, _ = critic_input(prepared.targets[0], prepared, coordinates, size, min_noise)
    real_scores, fake_scores = critic(real), critic(fake)
    loss = weighted_mean(jax.nn.softplus(-real_scores) + jax.nn.softplus(fake_scores), weights)
    return loss, dict(
        critic_loss=loss,
        critic_real=weighted_mean(jax.nn.sigmoid(real_scores), weights),
        critic_fake=weighted_mean(jax.nn.sigmoid(fake_scores), weights),
    )
