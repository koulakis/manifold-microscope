Installation
============

Python 3.10 or later is required. Install the library with:

.. code-block:: bash

   python -m pip install manifold-microscope

For the model-training and experiment scripts, add the ``benchmark`` extra:

.. code-block:: bash

   python -m pip install "manifold-microscope[benchmark]"

To use the current repository code and editable paper presets, install from
the repository root instead:

.. code-block:: bash

   python -m pip install -e ".[benchmark]"

Device selection
----------------

Grid APIs select CUDA when available, otherwise CPU. Pass ``device="cpu"``
or ``device="cuda:0"`` to choose explicitly. An explicit CUDA request raises
an error if CUDA cannot initialize.

Custom experiment configurations default to CPU. All paper presets request
``cuda:0``; pass ``--device cpu`` to the runner for an intentional CPU run.
NumPy preparation, aggregation and plotting use CPU. MPS is unvalidated.

Datasets
--------

Toy manifolds and custom dSprites are generated locally. COIL-20 requires the
processed grayscale PNGs from `Columbia's dataset page
<https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php>`_. Point
``COIL20_PATH`` to the directory containing all 20 × 72 images, including
``obj1__0.png``:

.. code-block:: bash

   export COIL20_PATH=/path/to/coil-20-proc

The original-dSprites loader also needs an external NPZ path, configured in
``microscope/datasets/original_dsprites.py``. Paper presets use generated
custom dSprites instead. See :doc:`../api_reference/datasets` for loaders and shapes.
