# Cython Compilation

## Local Development (compile in-place)

Run the following command in this directory (`src/traderalchemy_ta/`) to compile Cython extensions for local development:

```bash
python setup.py build_ext --inplace
```

This compiles `.pyx` files to `.pyd` (Windows) or `.so` (Linux/macOS) files in the same directory.

## Building Wheels (CI/CD)

Wheels are built automatically via GitHub Actions when pushing a version tag:

```bash
git tag ta-ta-v0.1.1
git push origin ta-ta-v0.1.1
```

Built wheels are published to GitHub Releases for:
- Linux x86_64 (manylinux)
- macOS Intel (x86_64)
- macOS Apple Silicon (arm64)
- Windows x64 (AMD64)

## Installation

### From GitHub Releases (pre-built)

```bash
pip install https://github.com/USERNAME/TraderAlchemy/releases/download/traderalchemy-ta-v0.1.0/traderalchemy_ta-0.1.0-cp312-cp312-win_amd64.whl
```

### From source (compiles locally)

```bash
pip install git+ssh://git@github.com/USERNAME/TraderAlchemy.git#subdirectory=pyutils/traderalchemy-ta
```