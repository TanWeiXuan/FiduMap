# Getting started

## Installation

From the repository root, create a virtual environment:

```sh
python -m venv .venv
```

On Windows:

```bat
.venv\Scripts\activate
pip install -r src\map_builder\requirements.txt
pip install pytest
```

On Linux/macOS:

```sh
source .venv/bin/activate
pip install -r src/map_builder/requirements.txt
pip install pytest
```

The main package requirements are:

```text
numpy
opencv-contrib-python
matplotlib
pyceres
```

`opencv-contrib-python` is required because marker detection uses `cv2.aruco`.

## Optional dependencies

The experimental dense reconstruction pipeline has additional optional dependencies, including **PyTorch** and the vendored **XFeat** model. These are not installed by default. Install PyTorch for your platform and ensure `vendor/xfeat` is present before running dense reconstruction stages.

If optional dependencies are missing, the dense reconstruction module should report missing packages and prevent unavailable stages from running.

## Running the GUI

On Windows, run:

```bat
run_map_builder.bat
```

Or directly from any platform with the required GUI dependencies available:

```sh
python src/map_builder/gui/app.py
```

The GUI supports selecting an image folder, indexing images, running marker detection, initialising poses, running bundle adjustment, inspecting results, visualising the marker map, and exporting the final CSV.

## Running tests

From the repository root:

```sh
pytest -q
```

The test suite exercises camera projection/unprojection, marker geometry, project persistence, synthetic bundle adjustment, GUI imports, dense reconstruction availability checks, and related helpers. The exact number of passing and skipped tests can vary depending on optional GUI/display availability and optional dependencies such as PyTorch.
