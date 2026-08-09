# idash/mpc/microbench/conftest.py
import sys, os

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))          # -> idash/mpc  (reference, dense_graph, ...)
sys.path.insert(0, os.path.join(_here, "..", "tests"))  # -> shared test_data_paths.py
