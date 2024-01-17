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

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/diffstarpop
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

