# Tray Racer - a raytracing suite

## Getting started

### Dependencies

- A recent [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- The [OptiX SDK](https://developer.nvidia.com/designworks/optix/download)
- Add a `OptiX_PATH` env variable if the OptiX installer doesn't do this for you

### Examples

The first couple of examples are just a walkthrough of some of the OptiX toolkit examples.

### Usage (for now)

- First, configure the project using one of the provided cmake presets:

```shell
cmake --preset debug-msvc
```

- Then, build and run an example (the `run.*` scripts do this for you, just modify to run the
example you want).

Example for windows:

```shell
cmake --build --preset debug-msvc 
cd ./builds/debug-msvc/exampels/optix7_02
./Debug/ex02.exe
cd -
```

## Particularities

### Raylib

Raylib's function definitions clash with some symbols in windows headers
(see [here](https://github.com/raysan5/raylib/issues/1217) for a discussion).

To fix this, and to have my own accessible list of log items, I added a custom logging system, which
is compiled with the `NO_SPDLOG` flag and is made available in the `utils_no_spdlog` target library.

