# Developer Guide

Help is very much welcome and so are pull requests!

## Building

Here is how you can build AudioGridder. Note that building the server is only
supported on macOS and Windows.

```
# Checkout the code
git clone --recurse-submodules https://github.com/apohl79/audiogridder.git
cd audiogridder

# Checkout the dependencies
git clone https://github.com/apohl79/audiogridder-deps.git

# Configure and compile
python3 build.py conf --disable-signing
python3 build.py build
```

### Building with Visual Studio Code on Linux
- Install juce dependencies, build essentials along with g++
- Download and install VSCode
- Install Microsoft C/C++ extension pack and CMake tools along with ninja build
- Open AudioGridder folder and import. All configuration will be set automatically.
- Select build type, target and build

## Coding conventions

Please follow the existing coding style (*m_* notation for class member
variables, cammel case for class, member and variable names, curly braces for
single statement control blocks, etc). Also before submitting a pull request,
make sure you are running the code formater.

```
python3 build.py format
```

Thanks for your contributions!
