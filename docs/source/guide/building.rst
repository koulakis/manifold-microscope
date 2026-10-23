Building and publishing the docs
================================

The documentation uses reStructuredText and Sphinx, following the layout of
`h-nne's docs <https://github.com/koulakis/h-nne/tree/main/docs>`_:
``docs/source/guide/`` holds the user guide and ``docs/source/api_reference/``
holds the API reference. Add new pages to the toctrees in ``source/index.rst``.

Build locally
-------------

From the repository root, using Python 3.10 or later:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   make -C docs html

Open ``docs/build/html/index.html``. No scientific-package installation,
dataset loading or experiment execution is needed for the docs build.
The requirements pin Sphinx and the Read the Docs theme; keep the ``docs``
extra in ``pyproject.toml`` in sync when changing them.

Check warnings before publishing:

.. code-block:: bash

   make -C docs html SPHINXOPTS="-W --keep-going"

Without Make, the equivalent command is:

.. code-block:: bash

   python -m sphinx -b html -W --keep-going docs/source docs/build/html

Publish on Read the Docs
------------------------

Read the Docs builds from Git; there is no manual HTML upload:

1. Commit and push the docs and root ``.readthedocs.yaml`` to the public repository.
2. Import ``koulakis/manifold-microscope`` in Read the Docs, or use its existing
   project. Select a branch containing these files as the default version.
3. Trigger a build and open its documentation link after it succeeds.
4. Activate release tags under Versions when versioned documentation is needed.

The root configuration selects Ubuntu 24.04, Python 3.12,
``docs/source/conf.py`` and ``docs/requirements.txt``. Warnings fail hosted
builds too. Once repository integration is enabled, pushes trigger rebuilds.
See the official `Read the Docs tutorial
<https://docs.readthedocs.com/platform/stable/tutorial/index.html>`_ and
`configuration reference
<https://docs.readthedocs.com/platform/stable/config-file/v2.html>`_.
