import os
import warnings
from packaging import version

import jax

from frax.core.robot import Robot
from frax.core.manipulator import Manipulator
from frax.core.humanoid import Humanoid
from frax.robots.franka_panda import load_panda
from frax.robots.unitree_g1 import load_g1


def check_env_vars():
    """
    If FRAX code is running on a single CPU, for best performance, there are
    some environment variables that should be set. Here, we'll check for if
    those env vars are set and print an info message if they're not set
    """
    devices = jax.devices()
    jax_version = jax.__version__

    if (len(devices) == 1 and devices[0].platform == "cpu") or (
        os.environ.get("JAX_PLATFORMS", "") == "cpu"
    ):
        x64_enabled = os.environ.get("JAX_ENABLE_X64", "").lower() in ("1", "true")
        xla_flags = os.environ.get("XLA_FLAGS", "").split()
        has_eigen_flag = "--xla_cpu_multi_thread_eigen=false" in xla_flags
        has_threads_flag = "intra_op_parallelism_threads=1" in xla_flags
        single_thread_enabled = has_eigen_flag and has_threads_flag
        before_jax_0_4_32 = version.parse(jax_version) < version.parse("0.4.32")

        msg = (
            "[frax] CPU backend detected but some performance settings are not configured.\n"
            + "These are optional, but should lead to better precision and speed. "
            + "See the frax README for more details."
        )
        if not before_jax_0_4_32:
            msg += (
                f"\n- Detected JAX version {jax_version}. "
                + "Consider using a version before JAX 0.4.32 for best CPU performance."
            )
        if not x64_enabled:
            msg += (
                "\n- JAX_ENABLE_X64 not detected. Recommendation: set JAX_ENABLE_X64=1"
            )
        if not single_thread_enabled:
            msg += (
                "\n- Single threaded XLA configuration not detected. "
                + "Recommendation: set XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'"
            )
        should_warn = (
            not before_jax_0_4_32 or not x64_enabled or not single_thread_enabled
        )
        if should_warn:
            warnings.warn(msg, stacklevel=2)


check_env_vars()
