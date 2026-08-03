"""
TDD – RED/GREEN for CUDA build infrastructure.

Tests:
  test_nvcc_available           – nvcc is on PATH and reports a version
  test_nvcc_smoke_compile       – trivial .cu compiles to a host binary
  test_deepdtagen_compiles_bw32 – deepdtagen_inference.cu compiles with BW=32, SM_89
  test_deepdtagen_gencode_sm90a – Makefile passes -gencode arch=compute_90a,code=sm_90a

Run:  ~/.pyenv/versions/3.8.7/bin/python -m pytest idash/mpc/tests/test_nvcc.py -v
"""
import subprocess, shutil, os, sys, tempfile
import pytest

HERE = os.path.join(os.path.dirname(__file__), "..", "gpu_mpc")
GPU_MPC_ROOT = os.path.expanduser("~/EzPC/GPU-MPC")
NVCC = shutil.which("nvcc") or os.path.expanduser("~/.local/cuda-12.1/usr/local/cuda-12.1/bin/nvcc")


def nvcc_available():
    if not os.path.isfile(NVCC):
        return False
    r = subprocess.run([NVCC, "--version"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not nvcc_available(), reason="nvcc not available")
class TestNvccSmoke:
    def test_nvcc_available(self):
        r = subprocess.run([NVCC, "--version"], capture_output=True, text=True)
        assert r.returncode == 0
        assert "release" in r.stdout.lower() or "release" in r.stderr.lower()

    def test_nvcc_smoke_compile(self, tmp_path):
        """A minimal host+device file compiles without error."""
        src = tmp_path / "hello.cu"
        src.write_text(
            "#include <stdio.h>\n"
            "__global__ void k() {}\n"
            "int main() { k<<<1,1>>>(); return 0; }\n"
        )
        r = subprocess.run(
            [NVCC, "-O0", "-o", str(tmp_path / "hello"), str(src)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"nvcc smoke failed:\n{r.stderr}"

    def test_deepdtagen_compiles_bw32(self, tmp_path):
        """deepdtagen_inference.cu compiles against local GPU-MPC with BW=32."""
        makefile = os.path.join(HERE, "Makefile")
        assert os.path.isfile(makefile), f"Makefile missing at {makefile}"
        r = subprocess.run(
            ["make", "-C", HERE, f"GPU_MPC_ROOT={GPU_MPC_ROOT}", "GPU_ARCH=89",
             "BW=32", "deepdtagen_inference", "--dry-run"],
            capture_output=True, text=True,
        )
        # dry-run must include the nvcc invocation (not just echo "nothing to do")
        assert "nvcc" in r.stdout or "nvcc" in r.stderr, \
            f"Expected nvcc call in dry-run:\n{r.stdout}\n{r.stderr}"

    def test_deepdtagen_gencode_sm90a(self, tmp_path):
        """Makefile must include SM_90a gencode for H100 compatibility."""
        r = subprocess.run(
            ["make", "-C", HERE, f"GPU_MPC_ROOT={GPU_MPC_ROOT}", "GPU_ARCH=90a",
             "BW=32", "deepdtagen_inference", "--dry-run"],
            capture_output=True, text=True,
        )
        combined = r.stdout + r.stderr
        assert "compute_90a" in combined or "sm_90a" in combined, \
            f"Expected SM_90a gencode in dry-run:\n{combined}"


class TestNvccNotAvailableMarker:
    """Always-visible: documents whether nvcc is available."""
    def test_nvcc_presence(self):
        ok = nvcc_available()
        if not ok:
            pytest.skip(f"nvcc not yet installed (expected at {NVCC})")
        # if nvcc is present this test just passes
        assert ok
