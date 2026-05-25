#!/usr/bin/env python3

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
OSRT_PYTHON_DIR = SCRIPT_DIR.parent
REPO_ROOT = OSRT_PYTHON_DIR.parent.parent

sys.path.insert(0, str(OSRT_PYTHON_DIR))
from model_configs_fashionmist import models_configs  # noqa: E402


FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def parse_label_from_filename(filename: str) -> int:
    """
    Expected format:
      00000_label_9.png
      09999_label_6.png
    """
    match = re.search(r"_label_(\d+)\.", filename)
    if not match:
        raise ValueError(f"Could not parse label from filename: {filename}")
    return int(match.group(1))


def load_fashionmnist_image(
    image_path: Path,
    height: int,
    width: int,
    input_mean,
    input_scale,
) -> np.ndarray:
    """
    Loads a FashionMNIST PNG image as grayscale and returns NCHW float32 tensor.
    Output shape: (1, 1, H, W)
    """
    img = Image.open(image_path).convert("L").resize((width, height), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0  # H, W in [0,1]

    # Apply simple channel-wise normalization if provided
    mean = float(input_mean[0]) if input_mean else 0.0
    scale = float(input_scale[0]) if input_scale else 1.0
    arr = (arr - mean) * scale

    # NCHW
    arr = arr[np.newaxis, np.newaxis, :, :]
    return arr.astype(np.float32)


def get_model_paths(config: dict) -> tuple[Path, Path]:
    model_path = Path(config["session"]["model_path"]).resolve()

    artifact_dir = config["session"].get("artifacts_folder", None)
    if artifact_dir is None:
        model_stem = model_path.stem
        artifact_dir = REPO_ROOT / "model-artifacts" / model_stem
    else:
        artifact_dir = Path(artifact_dir).resolve()

    return model_path, artifact_dir


def get_image_dir(config: dict) -> Path:
    # Prefer explicit config entry
    image_dir = None

    if "extra_info" in config and isinstance(config["extra_info"], dict):
        image_dir = config["extra_info"].get("input_images", None)

    if image_dir is None:
        image_dir = config.get("input_images", None)

    print(image_dir)
    if image_dir is None:
        # Default repo-local location for your FashionMNIST export
        image_dir = REPO_ROOT / "test_data" / "fashionmnist"
    else:
        image_dir = Path(image_dir)

    return Path(image_dir).resolve()


def create_session(
    model_path: Path,
    artifact_dir: Path,
    disable_offload: bool,
    compile_mode: bool,
) -> ort.InferenceSession:
    if disable_offload:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]

    elif compile_mode:
        tidl_tools_path = os.environ.get("TIDL_TOOLS_PATH")

        if not tidl_tools_path:
            raise RuntimeError("TIDL_TOOLS_PATH is not set in environment")

        providers = ["TIDLCompilationProvider"]
        provider_options = [
            {
                "tidl_tools_path": tidl_tools_path,
                "artifacts_folder": str(artifact_dir),
                "debug_level": "0",
                "advanced_options:c7x_firmware_version":"11_00_06_00",
                "advanced_options:core_number": 1,
            },
        ]
    else:
        tidl_tools_path = os.environ.get("TIDL_TOOLS_PATH")

        providers = ["TIDLExecutionProvider", "CPUExecutionProvider"]
        provider_options = [
            {
                "tidl_tools_path": tidl_tools_path,
                "artifacts_folder": str(artifact_dir),
                "debug_level": "0",
                "advanced_options:core_number": 1,
            },
            {},
        ]
    print("Using providers:", providers)

    print("Creating ONNX Runtime session...", flush=True)
    return ort.InferenceSession(
        str(model_path),
        providers=providers,
        provider_options=provider_options,
    )

def run_model(config_name: str, disable_offload: bool, compile_mode: bool) -> int:
    if config_name not in models_configs:
        print(f"Model config '{config_name}' not found.")
        print("Available models:")
        for name in sorted(models_configs.keys()):
            print(" ", name)
        return 1

    config = models_configs[config_name]
    model_path, artifact_dir = get_model_paths(config)
    image_dir = get_image_dir(config)

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return 1

    if compile_mode:
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for path in artifact_dir.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)

            elif not disable_offload and not artifact_dir.exists():
                print(f"Artifact directory not found: {artifact_dir}")
                print("Run compile first, for example:")
                print(f"  python {Path(__file__).name} -c -m {config_name}")
                return 1
	        
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        return 1

    image_files = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() == ".png"]
    )
    if not image_files:
        print(f"No PNG images found in: {image_dir}")
        return 1

    session = create_session(
        model_path,
        artifact_dir,
        disable_offload,
        compile_mode,
    )
    print("Session Created")
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_name = input_meta.name
    input_shape = input_meta.shape

    if len(input_shape) != 4:
        raise RuntimeError(f"Expected 4D input, got shape: {input_shape}")

    _, channels, height, width = input_shape
    if channels != 1:
        raise RuntimeError(
            f"This custom script expects a 1-channel FashionMNIST model. "
            f"Model input shape is {input_shape}."
        )

    input_mean = config["session"].get("input_mean", [0.0])
    input_scale = config["session"].get("input_scale", [1.0])

    print("\nRunning_Model :", config_name)
    print("Model path    :", model_path)
    print("Artifacts dir :", artifact_dir)
    print("Image dir     :", image_dir)
    print("Input name    :", input_name)
    print("Input shape   :", input_shape)
    print("Output name   :", output_meta.name)
    print("Output shape  :", output_meta.shape)
    if disable_offload:
        mode = "CPU only"
    elif compile_mode:
        mode = "TIDL compilation"
    else:
        mode = "TIDL + CPU fallback"
    print("Execution mode:", mode)
    print()

    correct = 0
    total = 0
    total_wall_time = 0.0
    misclassified = []

    for idx, image_path in enumerate(image_files, start=1):
        true_label = parse_label_from_filename(image_path.name)

        input_tensor = load_fashionmnist_image(
            image_path=image_path,
            height=height,
            width=width,
            input_mean=input_mean,
            input_scale=input_scale,
        )

        t0 = time.time()
        output = session.run([output_meta.name], {input_name: input_tensor})[0]
        t1 = time.time()

        probs = softmax(output)
        pred = int(np.argmax(probs, axis=1)[0])

        wall_ms = (t1 - t0) * 1000.0
        total_wall_time += wall_ms
        total += 1

        if pred == true_label:
            correct += 1
        elif len(misclassified) < 10:
            misclassified.append(
                (
                    image_path.name,
                    true_label,
                    pred,
                    float(probs[0][pred]),
                )
            )

        if idx % 1000 == 0 or idx == len(image_files):
            acc = 100.0 * correct / total
            print(f"Processed {idx}/{len(image_files)} | Accuracy so far: {acc:.2f}%")

    accuracy = 100.0 * correct / total
    avg_wall_ms = total_wall_time / total

    print("\n=== Final Results ===")
    print(f"Accuracy          : {accuracy:.2f}%")
    print(f"Correct           : {correct}/{total}")
    print(f"Avg wall time/img : {avg_wall_ms:.3f} ms")

    if misclassified:
        print("\n=== Sample Misclassifications ===")
        for name, truth, pred, conf in misclassified:
            print(
                f"{name}: true={truth} ({FASHION_CLASSES[truth]}), "
                f"pred={pred} ({FASHION_CLASSES[pred]}), conf={conf:.4f}"
            )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="Run TIDL model compilation/import mode",
    )

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model config name from model_configs.py",
    )

    parser.add_argument(
        "-d",
        "--disable_offload",
        action="store_true",
        help="Run with CPUExecutionProvider only",
    )

    args = parser.parse_args()

    return run_model(
        args.model,
        args.disable_offload,
        args.compile,
    )

if __name__ == "__main__":
    raise SystemExit(main())
