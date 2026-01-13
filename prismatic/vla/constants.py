"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
# Note: ACTION_DIM now includes an extra dimension for EOS flag (dimension 8)
# Base action dimensions (without EOS): xyz(3) + rotation(3) + gripper(1) = 7
# Full action dimensions (with EOS): base(7) + eos_flag(1) = 8
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "BASE_ACTION_DIM": 7,  # Original action dimensions (xyz, rotation, gripper)
    "ACTION_DIM": 8,  # Extended with EOS flag as 8th dimension
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "BASE_ACTION_DIM": 14,  # Original action dimensions
    "ACTION_DIM": 15,  # Extended with EOS flag
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "BASE_ACTION_DIM": 7,  # Original action dimensions
    "ACTION_DIM": 8,  # Extended with EOS flag
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
BASE_ACTION_DIM = constants["BASE_ACTION_DIM"]  # Original action dim (without EOS)
ACTION_DIM = constants["ACTION_DIM"]  # Full action dim (with EOS flag as last dimension)
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  BASE_ACTION_DIM = {BASE_ACTION_DIM} (without EOS)")
print(f"  ACTION_DIM = {ACTION_DIM} (with EOS flag at dimension {ACTION_DIM})")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("Note: ACTION_DIM now includes EOS flag. Last dimension = 0 (continue) or 1 (end of substep)")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
