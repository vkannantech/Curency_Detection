"""Inference engine registry and helpers for multi-backend YOLO deployment.

This module keeps runtime selection, deployment recommendations, capability
reporting, and export planning in one place so the rest of the project can stay
small and focused on detection logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


SUPPORTED_ENGINES = {
    "auto",
    "pytorch",
    "onnx",
    "openvino",
    "torchscript",
    "tensorrt",
    "tflite",
    "edgetpu",
    "pb",
    "saved_model",
    "ncnn",
    "mnn",
    "imx",
    "rknn",
    "coreml",
    "paddle",
}


EXTENSION_TO_ENGINE = {
    ".pt": "pytorch",
    ".onnx": "onnx",
    ".torchscript": "torchscript",
    ".engine": "tensorrt",
    ".tflite": "tflite",
    ".mlpackage": "coreml",
    ".mnn": "mnn",
    ".rknn": "rknn",
    ".pb": "pb",
    ".xml": "openvino",
    ".param": "ncnn",
    ".pdmodel": "paddle",
}


@dataclass(frozen=True, slots=True)
class EngineCapability:
    """Describe what an engine is best at and where it usually runs."""

    name: str
    priority: int
    label: str
    typical_devices: tuple[str, ...]
    typical_formats: tuple[str, ...]
    speed_hint: str
    quality_hint: str
    pi_friendly: bool
    desktop_friendly: bool
    web_friendly: bool
    quantization_friendly: bool
    notes: str
    export_name: str | None = None


@dataclass(frozen=True, slots=True)
class EngineSpec:
    """Concrete engine selection for a specific weights path."""

    requested: str
    detected: str
    normalized_weights: str
    capability: EngineCapability


ENGINE_CAPABILITIES: dict[str, EngineCapability] = {
    "pytorch": EngineCapability(
        name="pytorch",
        priority=100,
        label="PyTorch",
        typical_devices=("desktop CPU", "NVIDIA GPU", "development laptop"),
        typical_formats=(".pt",),
        speed_hint="Best for training and debugging. Good baseline inference speed.",
        quality_hint="Usually the easiest way to preserve full training behavior.",
        pi_friendly=False,
        desktop_friendly=True,
        web_friendly=False,
        quantization_friendly=False,
        notes="Use this when iterating fast or when you still need training and validation inside one workflow.",
        export_name=None,
    ),
    "onnx": EngineCapability(
        name="onnx",
        priority=90,
        label="ONNX Runtime",
        typical_devices=("desktop CPU", "edge CPU", "Jetson", "Raspberry Pi"),
        typical_formats=(".onnx",),
        speed_hint="Very strong cross-platform runtime with broad tooling support.",
        quality_hint="Often the safest portable deployment format after training.",
        pi_friendly=True,
        desktop_friendly=True,
        web_friendly=False,
        quantization_friendly=True,
        notes="Good default export target when you want one format that can travel across many devices.",
        export_name="onnx",
    ),
    "openvino": EngineCapability(
        name="openvino",
        priority=85,
        label="OpenVINO",
        typical_devices=("Intel CPU", "Intel iGPU", "Intel NPU"),
        typical_formats=(".xml", ".bin"),
        speed_hint="Excellent on Intel hardware, especially CPU-heavy deployments.",
        quality_hint="Stable production backend for optimized desktop or kiosk inference.",
        pi_friendly=False,
        desktop_friendly=True,
        web_friendly=False,
        quantization_friendly=True,
        notes="Best when the target system is Intel-based and you care about CPU efficiency.",
        export_name="openvino",
    ),
    "torchscript": EngineCapability(
        name="torchscript",
        priority=75,
        label="TorchScript",
        typical_devices=("desktop CPU", "embedded Linux", "PyTorch serving"),
        typical_formats=(".torchscript",),
        speed_hint="Useful when you want a PyTorch-adjacent compiled artifact.",
        quality_hint="Close to the original PyTorch behavior with simpler packaging.",
        pi_friendly=True,
        desktop_friendly=True,
        web_friendly=False,
        quantization_friendly=False,
        notes="A good middle-ground if you want deployment simplification without fully changing ecosystems.",
        export_name="torchscript",
    ),
    "tensorrt": EngineCapability(
        name="tensorrt",
        priority=95,
        label="TensorRT",
        typical_devices=("NVIDIA GPU", "Jetson"),
        typical_formats=(".engine",),
        speed_hint="Usually the fastest inference backend on supported NVIDIA hardware.",
        quality_hint="Great for production throughput after tuning the exported engine.",
        pi_friendly=False,
        desktop_friendly=True,
        web_friendly=False,
        quantization_friendly=True,
        notes="Use this for maximum GPU inference speed when the target hardware is NVIDIA.",
        export_name="engine",
    ),
    "tflite": EngineCapability(
        name="tflite",
        priority=80,
        label="TensorFlow Lite",
        typical_devices=("Raspberry Pi", "Android", "edge ARM CPU"),
        typical_formats=(".tflite",),
        speed_hint="Lightweight runtime for embedded and mobile CPU inference.",
        quality_hint="Very practical for compact deployment when accuracy remains acceptable.",
        pi_friendly=True,
        desktop_friendly=False,
        web_friendly=False,
        quantization_friendly=True,
        notes="Strong deployment choice for Raspberry Pi when you want smaller runtime dependencies.",
        export_name="tflite",
    ),
    "edgetpu": EngineCapability(
        name="edgetpu",
        priority=82,
        label="Edge TPU",
        typical_devices=("Coral USB", "Coral Dev Board"),
        typical_formats=(".tflite",),
        speed_hint="Very fast on compatible Coral accelerators.",
        quality_hint="Works best after careful quantized export and supported ops alignment.",
        pi_friendly=True,
        desktop_friendly=False,
        web_friendly=False,
        quantization_friendly=True,
        notes="Use when pairing Raspberry Pi with Coral Edge TPU hardware.",
        export_name="edgetpu",
    ),
    "pb": EngineCapability(
        name="pb",
        priority=55,
        label="TensorFlow GraphDef",
        typical_devices=("TensorFlow serving", "legacy TF pipelines"),
        typical_formats=(".pb",),
        speed_hint="More legacy than cutting-edge, but still useful for some TF ecosystems.",
        quality_hint="Best reserved for compatibility with older TensorFlow infrastructure.",
        pi_friendly=False,
        desktop_friendly=True,
        web_friendly=False,
        quantization_friendly=False,
        notes="Choose only if your downstream toolchain explicitly needs GraphDef.",
        export_name="pb",
    ),
    "saved_model": EngineCapability(
        name="saved_model",
        priority=60,
        label="TensorFlow SavedModel",
        typical_devices=("TensorFlow serving", "TF pipelines", "cloud inference"),
        typical_formats=("saved_model.pb",),
        speed_hint="Good interoperability for TensorFlow-native deployment stacks.",
        quality_hint="Useful when your cloud or app pipeline expects SavedModel assets.",
        pi_friendly=False,
        desktop_friendly=True,
        web_friendly=False,
        quantization_friendly=True,
        notes="Prefer this when the rest of your serving stack is already TensorFlow-based.",
        export_name="saved_model",
    ),
    "ncnn": EngineCapability(
        name="ncnn",
        priority=78,
        label="NCNN",
        typical_devices=("mobile CPU", "Raspberry Pi", "small ARM boards"),
        typical_formats=(".param", ".bin"),
        speed_hint="Efficient on low-power CPU devices.",
        quality_hint="A strong edge deployment option when lightweight native inference matters.",
        pi_friendly=True,
        desktop_friendly=False,
        web_friendly=False,
        quantization_friendly=True,
        notes="Useful for ARM-heavy deployments where minimal runtime size matters.",
        export_name="ncnn",
    ),
    "mnn": EngineCapability(
        name="mnn",
        priority=70,
        label="MNN",
        typical_devices=("mobile device", "edge ARM board"),
        typical_formats=(".mnn",),
        speed_hint="Compact runtime focused on embedded and mobile inference.",
        quality_hint="Good for highly constrained deployment environments.",
        pi_friendly=True,
        desktop_friendly=False,
        web_friendly=False,
        quantization_friendly=True,
        notes="Best used when your downstream runtime specifically prefers MNN artifacts.",
        export_name="mnn",
    ),
    "imx": EngineCapability(
        name="imx",
        priority=68,
        label="IMX",
        typical_devices=("NXP i.MX edge device",),
        typical_formats=("vendor-specific",),
        speed_hint="Hardware-focused backend for supported embedded SoCs.",
        quality_hint="Most useful when you already know the deployment board family.",
        pi_friendly=False,
        desktop_friendly=False,
        web_friendly=False,
        quantization_friendly=True,
        notes="Specialized export path for NXP embedded targets.",
        export_name="imx",
    ),
    "rknn": EngineCapability(
        name="rknn",
        priority=74,
        label="RKNN",
        typical_devices=("Rockchip NPU", "RK3588 edge board"),
        typical_formats=(".rknn",),
        speed_hint="Good accelerator-backed speed on Rockchip devices.",
        quality_hint="Solid choice if the final device is explicitly Rockchip-based.",
        pi_friendly=False,
        desktop_friendly=False,
        web_friendly=False,
        quantization_friendly=True,
        notes="Use only when you have a Rockchip deployment target.",
        export_name="rknn",
    ),
    "coreml": EngineCapability(
        name="coreml",
        priority=72,
        label="CoreML",
        typical_devices=("iPhone", "iPad", "Apple Silicon"),
        typical_formats=(".mlpackage",),
        speed_hint="Strong Apple-device deployment path.",
        quality_hint="Useful for polished mobile demos in the Apple ecosystem.",
        pi_friendly=False,
        desktop_friendly=False,
        web_friendly=False,
        quantization_friendly=True,
        notes="Use this if you want an iOS or Apple Silicon showcase version.",
        export_name="coreml",
    ),
    "paddle": EngineCapability(
        name="paddle",
        priority=58,
        label="PaddlePaddle",
        typical_devices=("Paddle serving", "Baidu ecosystem", "CPU server"),
        typical_formats=(".pdmodel", ".pdiparams"),
        speed_hint="Compatibility-focused backend for Paddle-based stacks.",
        quality_hint="Not the first choice for this project unless the deployment stack needs it.",
        pi_friendly=False,
        desktop_friendly=True,
        web_friendly=False,
        quantization_friendly=True,
        notes="Select when integration requirements point to Paddle rather than PyTorch or ONNX.",
        export_name="paddle",
    ),
}


PI_RECOMMENDED_ENGINES = ("onnx", "tflite", "ncnn", "mnn", "torchscript")
DESKTOP_RECOMMENDED_ENGINES = ("pytorch", "onnx", "openvino", "tensorrt", "torchscript")
MOBILE_RECOMMENDED_ENGINES = ("tflite", "ncnn", "mnn", "coreml")


def _require_supported_engine(engine: str) -> str:
    normalized = engine.lower().strip()
    if normalized not in SUPPORTED_ENGINES:
        allowed = ", ".join(sorted(SUPPORTED_ENGINES))
        raise ValueError(f"Unsupported engine '{engine}'. Supported values: {allowed}")
    return normalized


def _detect_engine_from_directory(path: Path) -> tuple[str, Path]:
    xml_files = sorted(path.glob("*.xml"))
    if xml_files:
        return "openvino", xml_files[0]

    if (path / "saved_model.pb").exists():
        return "saved_model", path

    param_files = sorted(path.glob("*.param"))
    if param_files:
        return "ncnn", param_files[0]

    pdmodel_files = sorted(path.glob("*.pdmodel"))
    if pdmodel_files:
        return "paddle", pdmodel_files[0]

    mlpackage_dirs = sorted(path.glob("*.mlpackage"))
    if mlpackage_dirs:
        return "coreml", mlpackage_dirs[0]

    return "pytorch", path


def _detect_engine_from_path(path: Path) -> tuple[str, Path]:
    if path.is_dir():
        return _detect_engine_from_directory(path)

    detected = EXTENSION_TO_ENGINE.get(path.suffix.lower(), "pytorch")
    return detected, path


def _capability_for_engine(engine: str) -> EngineCapability:
    if engine == "auto":
        raise ValueError("'auto' does not have a concrete capability profile.")
    return ENGINE_CAPABILITIES[engine]


def list_supported_engines() -> list[str]:
    """Return a stable sorted engine list for CLI and documentation."""
    return sorted(SUPPORTED_ENGINES)


def list_exportable_engines() -> list[str]:
    """Return engines that map cleanly to export formats."""
    exportables = [name for name, capability in ENGINE_CAPABILITIES.items() if capability.export_name]
    return sorted(exportables)


def get_engine_capability(engine: str) -> EngineCapability:
    """Return capability metadata for a concrete engine."""
    normalized = _require_supported_engine(engine)
    if normalized == "auto":
        raise ValueError("Use a concrete engine name when requesting capability metadata.")
    return _capability_for_engine(normalized)


def iter_engine_capabilities(engines: Iterable[str] | None = None) -> list[EngineCapability]:
    """Return capability objects ordered by deployment priority."""
    selected = engines or ENGINE_CAPABILITIES.keys()
    capabilities = [get_engine_capability(engine) for engine in selected if engine != "auto"]
    return sorted(capabilities, key=lambda item: (-item.priority, item.name))


def get_recommended_engines(target: str) -> tuple[str, ...]:
    """Recommend engines for a deployment target."""
    normalized = target.lower().strip()
    if normalized in {"pi", "raspberry-pi", "raspberry_pi", "arm"}:
        return PI_RECOMMENDED_ENGINES
    if normalized in {"desktop", "laptop", "workstation"}:
        return DESKTOP_RECOMMENDED_ENGINES
    if normalized in {"mobile", "android", "ios"}:
        return MOBILE_RECOMMENDED_ENGINES
    return DESKTOP_RECOMMENDED_ENGINES


def resolve_engine(weights: str, engine: str = "auto") -> EngineSpec:
    """Resolve the runtime engine from either the requested engine or the weights path."""
    requested = _require_supported_engine(engine)
    path = Path(weights)
    detected, normalized_path = _detect_engine_from_path(path)

    if requested != "auto" and detected != requested:
        raise ValueError(
            f"Engine mismatch: weights '{weights}' look like '{detected}', but '--engine {requested}' was requested."
        )

    concrete = detected if requested == "auto" else requested
    return EngineSpec(
        requested=requested,
        detected=concrete,
        normalized_weights=str(normalized_path),
        capability=_capability_for_engine(concrete),
    )


def load_model(weights: str, engine: str = "auto") -> tuple[YOLO, EngineSpec]:
    """Load a YOLO model using a resolved runtime engine."""
    spec = resolve_engine(weights, engine)
    return YOLO(spec.normalized_weights), spec


def engine_support_matrix() -> str:
    """Build a human-readable backend summary table."""
    lines = [
        "Engine | Devices | Pi | Desktop | Quantization | Notes",
        "--- | --- | --- | --- | --- | ---",
    ]
    for capability in iter_engine_capabilities():
        lines.append(
            " | ".join(
                [
                    capability.label,
                    ", ".join(capability.typical_devices),
                    "yes" if capability.pi_friendly else "no",
                    "yes" if capability.desktop_friendly else "no",
                    "yes" if capability.quantization_friendly else "no",
                    capability.notes,
                ]
            )
        )
    return "\n".join(lines)


def build_runtime_hint(engine: str) -> str:
    """Return a short deployment hint for the selected engine."""
    capability = get_engine_capability(engine)
    return (
        f"{capability.label}: {capability.speed_hint} "
        f"{capability.quality_hint} Recommended devices: {', '.join(capability.typical_devices)}."
    )


def build_export_plan(target: str) -> list[tuple[str, str]]:
    """Return a prioritized list of engines and reasons for a deployment target."""
    plans: list[tuple[str, str]] = []
    for engine in get_recommended_engines(target):
        capability = get_engine_capability(engine)
        plans.append((engine, capability.notes))
    return plans


def export_formats_for_target(target: str) -> list[str]:
    """Return recommended export formats for a target environment."""
    formats: list[str] = []
    for engine in get_recommended_engines(target):
        capability = get_engine_capability(engine)
        if capability.export_name and capability.export_name not in formats:
            formats.append(capability.export_name)
    return formats


def backend_cli_help() -> str:
    """Produce a compact multi-line help block for CLI output."""
    lines = ["Available engines:"]
    for capability in iter_engine_capabilities():
        lines.append(
            f"- {capability.name}: {capability.speed_hint} "
            f"Pi={'yes' if capability.pi_friendly else 'no'}, "
            f"Desktop={'yes' if capability.desktop_friendly else 'no'}"
        )
    return "\n".join(lines)
