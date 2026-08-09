"""Centralized test data paths configuration.

All test files should import paths from here instead of hardcoding them.
"""
import os

# Root of the mpc directory
MPC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory: idash/mpc/data/
DATA_DIR = os.path.join(MPC_ROOT, "data")

# Test CSV files
DAVIS_TEST_CSV = os.path.join(DATA_DIR, "davis_test.csv")
KIBA_TRAIN_CSV = os.path.join(DATA_DIR, "kiba_train.csv")

# Model directory: idash/mpc/model/
MODEL_DIR = os.path.join(MPC_ROOT, "model")

# Check if files exist (for skip conditions in tests)
DAVIS_TEST_EXISTS = os.path.exists(DAVIS_TEST_CSV)
KIBA_TRAIN_EXISTS = os.path.exists(KIBA_TRAIN_CSV)
