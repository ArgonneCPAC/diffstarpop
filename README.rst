diffstarpop
============

DiffstarPop is a python library based on JAX for generating statistical realizations 
of the `diffstar <https://diffstar.readthedocs.io/en/latest/>`_ model in
simulation-based forward modeling applications.

Installation
------------
DiffstarPop is currently a private repo that must be installed from source::

    $ cd /path/to/root/diffstarpop
    $ pip install .


Environment configuration
-------------------------
The following step is not required, but we recommend you 

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/diffstarpop
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

Some of the unit tests require that the DIFFSTARPOP_DRN environment variable is set.
The reason for this is because the data stored in this directory are too large 
to include in the repo and so tests that rely on these data must be run locally.
To create the DIFFSTARPOP_DRN environment variable with the directory 
where the dataset is stored on your disk, add the following line 
to your .bash_profile (for bash users) or .zshrc (for zshell users)::

    export DIFFSTARPOP_DRN="/path/to/drn/containing/diffstarpop/data"
