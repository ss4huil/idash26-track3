# idash/mpc/tests/test_mpc_config.py
from reference import mpc_config as cfg

def test_fixed_point_params():
    assert cfg.BW == 32
    assert cfg.SCALE == 12
    assert cfg.NMAX == 138 and cfg.FEAT_DIM == 94 and cfg.POOL_DIM == 376

def test_share_filename_is_zero_based_no_prefix():
    assert cfg.share_filename("x", 0) == "x_share0.dat"
    assert cfg.share_filename("adj", 1) == "adj_share1.dat"
    assert cfg.share_filename("mask", 0) == "mask_share0.dat"

def test_key_filename_matches_cpp_expname():
    assert cfg.key_filename(32, 12) == "DeepDTAGen_32_12"
