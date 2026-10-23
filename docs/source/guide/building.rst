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
2. In Read the Docs Community, add ``koulakis/manifold-microscope`` as a public
   project with slug ``manifold-microscope``, or connect its existing project.
   Grant the Read the Docs GitHub App access to the repository. Select ``main``
   as the default branch once the docs changes are merged.
3. Build ``latest`` and open its documentation link after it succeeds. Confirm
   repository integration receives pushes: the GitHub App handles events
   automatically; older OAuth integrations use a repository webhook.
4. After pushing tag ``v0.2.0``, open Versions, activate that version and let
   its build finish. Keep it active and visible as the fixed paper reference.
   Verify ``/en/v0.2.0/`` serves the 0.2.0 docs. If its version slug differs,
   use the exact URL shown by Read the Docs.

``latest`` tracks the default branch; ``stable`` tracks the newest stable
release. Neither replaces the explicit ``v0.2.0`` version for paper reproduction.
Future releases can update those aliases while the paper docs stay available.

The root configuration selects Ubuntu 24.04, Python 3.12,
``docs/source/conf.py`` and ``docs/requirements.txt``. Warnings fail hosted
builds too. Once repository integration is enabled, pushes trigger rebuilds.
See the official `Read the Docs tutorial
<https://docs.readthedocs.com/platform/stable/tutorial/index.html>`_ and
`configuration reference
<https://docs.readthedocs.com/platform/stable/config-file/v2.html>`_.

README badges
-------------

The documentation badge needs a public Read the Docs project and a build of
the version in its URL (currently ``latest``). No separate badge activation
or token is needed. If it says ``unknown``, check the project slug, version
activation and build log; a successful build reports ``passing``.
See `status badges
<https://docs.readthedocs.com/platform/stable/badges.html>`_.

The Pepy badge counts total PyPI downloads automatically. Public badges need
no registration, API key, repository webhook or package instrumentation.
It counts downloads across versions, rather than paper runs or unique users;
provider and image caching can delay updates. See the `Pepy badge reference
<https://pepy.tech/pepy-api>`_.
