import os
import time
import copy
import torch
import numpy as np

from validate import validate
from util import Logger, printSet
from test_options import TestOptions

# Import the two separate model implementations
from freqnet1 import freqnet as original_freqnet
from freqnet import freqnet as modified_freqnet


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CHECKPOINT_PATH = "/content/4-classes-freqnet-v2.pth"

DetectionTests = {
    "ForenSynths": {
        "dataroot": "/content/dataset/ForenSynths/",
        "no_resize": False,
        "no_crop": True,
    },
    "GANGen-Detection": {
        "dataroot": "/content/dataset/GANGen-Detection/",
        "no_resize": True,
        "no_crop": True,
    },
}


# --------------------------------------------------
# DEVICE AND CHECKPOINT
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Checkpoint:", CHECKPOINT_PATH)

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

# Support either a direct state_dict or a wrapped checkpoint.
if isinstance(checkpoint, dict) and "model" in checkpoint:
    checkpoint = checkpoint["model"]

# Handle checkpoints saved with DataParallel's "module." prefix.
if checkpoint and all(
    key.startswith("module.") for key in checkpoint.keys()
):
    checkpoint = {
        key[len("module."):]: value
        for key, value in checkpoint.items()
    }


# --------------------------------------------------
# CREATE BOTH MODELS WITH THE SAME WEIGHTS
# --------------------------------------------------

models = {
    "Original FreqNet": original_freqnet(num_classes=1),
    "Modified FreqNet": modified_freqnet(num_classes=1),
}

for name, model in models.items():
    print(f"\nLoading weights into {name}...")

    # Strict loading catches architecture/key mismatches.
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    print(f"{name}: checkpoint loaded successfully.")


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

opt_base = TestOptions().parse(print_options=False)

all_results = {}

for testSet, config in DetectionTests.items():

    dataroot = config["dataroot"]

    # Skip datasets that have not been downloaded.
    if not os.path.isdir(dataroot):
        print(f"\nSkipping {testSet}: directory not found:")
        print(dataroot)
        continue

    # Only evaluate immediate subdirectories, as in your
    # original test.py.
    subfolders = sorted([
        folder for folder in os.listdir(dataroot)
        if os.path.isdir(os.path.join(dataroot, folder))
    ])

    if not subfolders:
        print(f"\nSkipping {testSet}: no subdirectories found.")
        continue

    print("\n" + "=" * 65)
    printSet(testSet)
    print("=" * 65)

    all_results[testSet] = {}

    for model_name, model in models.items():

        print(f"\nEvaluating: {model_name}")
        print(time.strftime("%Y-%m-%d %H:%M:%S"))

        accuracies = []
        average_precisions = []

        for index, folder in enumerate(subfolders):

            opt = copy.deepcopy(opt_base)

            opt.dataroot = os.path.join(dataroot, folder)
            opt.classes = ""
            opt.no_resize = config["no_resize"]
            opt.no_crop = config["no_crop"]

            result = validate(model, opt)

            acc, ap, *_ = result

            accuracies.append(float(acc))
            average_precisions.append(float(ap))

            print(
                f"({index} {folder:20}) "
                f"ACC: {acc * 100:.2f}% | "
                f"AP: {ap * 100:.2f}%"
            )

        mean_acc = float(np.mean(accuracies))
        mean_ap = float(np.mean(average_precisions))

        all_results[testSet][model_name] = {
            "accuracy": mean_acc,
            "ap": mean_ap,
            "per_folder_acc": accuracies,
            "per_folder_ap": average_precisions,
            "folders": subfolders,
        }

        print("-" * 55)
        print(f"{model_name} mean ACC: {mean_acc * 100:.2f}%")
        print(f"{model_name} mean AP : {mean_ap * 100:.2f}%")
        print("-" * 55)


# --------------------------------------------------
# FINAL COMPARISON
# --------------------------------------------------

print("\n\n" + "=" * 75)
print("ORIGINAL vs MODIFIED FREQNET — PERFORMANCE COMPARISON")
print("=" * 75)

for testSet, results in all_results.items():

    if (
        "Original FreqNet" not in results
        or "Modified FreqNet" not in results
    ):
        continue

    original = results["Original FreqNet"]
    modified = results["Modified FreqNet"]

    acc_diff = (
        modified["accuracy"] - original["accuracy"]
    ) * 100

    ap_diff = (
        modified["ap"] - original["ap"]
    ) * 100

    print(f"\nDataset: {testSet}")
    print(
        f"Original ACC : {original['accuracy'] * 100:.2f}%"
    )
    print(
        f"Modified ACC : {modified['accuracy'] * 100:.2f}%"
    )
    print(f"ACC change   : {acc_diff:+.2f} percentage points")

    print(
        f"Original AP  : {original['ap'] * 100:.2f}%"
    )
    print(
        f"Modified AP  : {modified['ap'] * 100:.2f}%"
    )
    print(f"AP change    : {ap_diff:+.2f} percentage points")

    if acc_diff > 0:
        print("Accuracy improved.")
    elif acc_diff < 0:
        print("Accuracy decreased.")
    else:
        print("Accuracy unchanged.")

    if ap_diff > 0:
        print("Average precision improved.")
    elif ap_diff < 0:
        print("Average precision decreased.")
    else:
        print("Average precision unchanged.")

print("\nEvaluation complete.")
