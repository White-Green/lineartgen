# lineartgen-native

CPU inference extension for Python 3.10 and later. The epoch 1500 model is embedded in the extension, and image data stays in memory as tightly packed BGRA8 bytes.

```bash
python -m pip install "maturin>=1.15,<2"
maturin develop --release --locked --manifest-path crates/lineartgen-native/Cargo.toml
```

```python
import lineartgen_native

model = lineartgen_native.LineartModel()  # optional num_threads=...
output = model.infer(
    scribble=scribble,
    lineart=lineart,
    width=width,
    height=height,
    strength=strength,
    seed=seed,
    denoise_steps=denoise_steps,
)
```

Both inputs and the output contain exactly `width * height * 4` bytes in BGRA order. The output RGB channels are zero and the generated line darkness is stored in alpha.

Both dimensions must be at least 16 pixels. Pyramid depth is derived automatically: each dimension is halved until both are at most 128 pixels, unless another reduction would make a dimension smaller than 16 pixels. Non-power-of-two inputs are padded on the right and bottom and cropped back to the requested size.

`strength` must be between 0 and 1, and `denoise_steps` must be between 1 and 20. Existing lineart alpha protects that pixel from added noise. No filesystem image exchange or PNG encoding is performed.

The default wheel uses Burn Flex. The experimental CubeCL CPU backend can be built with:

```bash
maturin build --release --no-default-features --features cube-cpu --manifest-path crates/lineartgen-native/Cargo.toml
```

Flex and Vulkan inference can be benchmarked under identical conditions with:

```bash
cargo test -p lineartgen-native --release --features vulkan-benchmark benchmark_ -- --ignored --nocapture --test-threads=1
```

## Building from a submodule on Windows x64

The `diffusion_drawing` repository pins this repository at `lineartgen/` and builds
the extension from that checkout. Commit and push changes to this repository
before updating the consumer's submodule pointer. `assets/model.mpk` and
`assets/manifest.json` are checked in and embedded at compile time; no training
directory or model download is needed to build or run the extension.

From the consumer repository, with 64-bit CPython 3.10 or later, Rust and the
Visual Studio C++ build tools installed:

```powershell
git submodule update --init --recursive
python -m pip install "maturin>=1.15,<2"
python -m maturin build --release --locked --target x86_64-pc-windows-msvc --manifest-path lineartgen/crates/lineartgen-native/Cargo.toml --out wheels
```

The resulting `cp310-abi3-win_amd64.whl` uses Burn Flex and includes the model.
Install the wheel into the plugin's private package directory during packaging:

```powershell
python -m pip install --no-deps --target diffusion_drawing/_native (Get-ChildItem wheels/lineartgen_native-*.whl).FullName
```

The plugin can then use `from ._native.lineartgen_native import LineartModel`.
Krita's Python does not need pip, Rust or maturin at runtime. The wheel targets
standard, GIL-enabled CPython 3.10+ on Windows x64; other operating systems and
CPU architectures require their own builds. Maturin's
[distribution guide](https://www.maturin.rs/distribution.html) describes wheel
compatibility and the [maturin action](https://github.com/PyO3/maturin-action)
provides the GitHub Actions build step.
