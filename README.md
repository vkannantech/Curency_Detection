# The Blind Assist: Currency Detection For the Visually Impaired

This project is a YOLO-based real-time currency detector built for a college project and designed to work on both:

- a stronger desktop or laptop for training
- a Raspberry Pi for real-time camera inference

It is structured for Indian currency by default and includes both paper notes and coins.

## Project Goal

The app detects currency through a live camera feed and can speak the detected denomination aloud for visually impaired users.

## Important Accuracy Note

`90%+ accuracy` is possible, but it depends much more on the dataset quality than on the model name alone.

To reach strong accuracy in a real project, you should collect and label:

- clean front and back images of each note and coin
- different lighting conditions
- partial occlusions
- folded notes
- worn notes and shiny coins
- multiple backgrounds
- different camera distances and angles
- Raspberry Pi camera images, not only phone images

## Model Recommendation

Ultralytics currently documents `YOLO26` as the newest model family, while also noting `YOLO11` as a stable production option for many workloads:

- Ultralytics docs: https://docs.ultralytics.com/

For this project:

- train on desktop with `yolo11m.pt` or `yolo26m.pt`
- deploy on Raspberry Pi with exported `ONNX` or a smaller trained model
- if the Pi is slow, use the `n` or `s` model size for real-time use

## Default Classes

This scaffold is prepared for these Indian currency classes:

- `inr_1_coin`
- `inr_2_coin`
- `inr_5_coin`
- `inr_10_coin`
- `inr_20_coin`
- `inr_10_note`
- `inr_20_note`
- `inr_50_note`
- `inr_100_note`
- `inr_200_note`
- `inr_500_note`

You can edit the class list later in [dataset/data.yaml](/d:/Code%20Space/Curency_Detection/dataset/data.yaml) and [labels.py](/d:/Code%20Space/Curency_Detection/src/currency_detection/labels.py).

## Folder Structure

```text
Curency_Detection/
├── currency/                     # requested virtual environment
├── dataset/
│   ├── data.yaml
│   └── README.md
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── export_pi.py
│   └── run_camera.py
├── src/
│   └── currency_detection/
│       ├── camera.py
│       ├── config.py
│       ├── labels.py
│       └── speech.py
├── requirements.txt
└── requirements-pi.txt
```

## Virtual Environment

You asked for a venv named `currency`, and it has been created in the project folder. On this machine, Python `3.14` created the environment but failed while bootstrapping `pip` because of a Windows permission problem during `ensurepip`.

That means:

- `currency\Scripts\python.exe` exists
- `pip` is not available yet inside that environment

Recommended fix:

1. Install Python `3.11`
2. Recreate the venv with Python `3.11`
3. Install dependencies

Example:

```powershell
Remove-Item -Recurse -Force currency
py -3.11 -m venv currency
.\currency\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you stay on the current machine and `pip` is repaired later, use:

```powershell
.\currency\Scripts\python.exe -m pip install -r requirements.txt
```

## Desktop Training Setup

Install desktop dependencies:

```powershell
pip install -r requirements.txt
```

Train:

```powershell
python scripts/dataset_audit.py
python scripts/train.py --model yolo11m.pt --profile max-accuracy --epochs 120 --imgsz 960
```

If your Ultralytics version supports the newest weights:

```powershell
python scripts/train.py --model yolo26m.pt --epochs 120 --imgsz 960
```

Evaluate:

```powershell
python scripts/evaluate.py --weights runs/detect/currency_yolo/weights/best.pt
```

Export for Raspberry Pi:

```powershell
python scripts/export_pi.py --weights runs/detect/currency_yolo/weights/best.pt --format onnx
```

## Raspberry Pi Setup

Install Pi dependencies:

```bash
python3 -m venv currency
source currency/bin/activate
pip install -r requirements-pi.txt
```

Run the live camera assistant:

```bash
python scripts/run_camera.py --weights models/best.pt --source 0 --profile pi --speak
```

If you use the Raspberry Pi Camera Module, make sure OpenCV can access the camera on your OS image. If needed, use `libcamera`-enabled images and test camera access first.

## Web Demo Deployment

This repo also includes a simple Gradio demo in [app.py](/d:/Code%20Space/Curency_Detection/app.py) so you can publish an online image-upload version of the project.

Local run:

```powershell
pip install -r requirements-web.txt
python app.py
```

Good hosting options:

- Hugging Face Spaces
- Render
- Railway

For Hugging Face Spaces:

1. Create a new `Gradio` Space
2. Upload this repo
3. Use [requirements-web.txt](/d:/Code%20Space/Curency_Detection/requirements-web.txt)
4. Keep `app.py` as the main entry file

Note: the full real-time webcam + speech assistant is best for desktop or Raspberry Pi. The web app is mainly for online demo and project presentation.

## Multi-Engine Runtime Support

The camera assistant can now run with multiple exported model engines through the same Python code path.

Supported `--engine` values:

- `auto`
- `pytorch`
- `onnx`
- `openvino`
- `torchscript`
- `tensorrt`
- `tflite`
- `edgetpu`
- `pb`
- `saved_model`
- `ncnn`
- `mnn`
- `imx`
- `rknn`
- `coreml`
- `paddle`

Examples:

```powershell
python scripts/run_camera.py --weights last.pt --engine auto --source 0 --speak
python scripts/run_camera.py --weights exports\\best.onnx --engine onnx --source 0 --profile pi
python scripts/run_camera.py --weights exports\\openvino_model --engine openvino --source 0 --profile pi-lite
```

This keeps the same preprocessing, temporal smoothing, speech, and total-value logic across all supported runtimes.

Additional backend tools:

- [list_engines.py](/d:/Code%20Space/Curency_Detection/scripts/list_engines.py)
- [export_all.py](/d:/Code%20Space/Curency_Detection/scripts/export_all.py)
- [benchmark_backends.py](/d:/Code%20Space/Curency_Detection/scripts/benchmark_backends.py)

Examples:

```powershell
python scripts/list_engines.py --target pi
python scripts/export_all.py --weights last.pt --target pi
python scripts/benchmark_backends.py --weights last.pt exports\best.onnx --image sample.jpg --runs 10
python scripts/run_camera.py --weights exports\best.onnx --engine onnx --source 0 --profile pi
```

## Dataset Layout

Follow standard YOLO detection format:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image needs a matching `.txt` file with:

```text
class_id x_center y_center width height
```

All values must be normalized between `0` and `1`.

## Practical Advice For 90%+

- collect at least `800 to 1500` labeled images per class if possible
- capture both old and new note designs
- include real coin glare and motion blur
- train with `imgsz 960` or `1280` if your GPU allows
- use class-balanced data
- keep validation images separate from training images
- test on Raspberry Pi camera frames before final submission

## Real-Time Assistant Features

The camera app includes:

- live webcam or Pi camera processing
- confidence threshold control
- spoken denomination announcements
- duplicate announcement suppression
- optional headless mode for Pi deployments without display
- confidence-weighted temporal consensus across frames
- low-light and low-contrast preprocessing
- desktop and Raspberry Pi inference profiles

## Next Best Upgrade

If you want, the next step after this scaffold is to:

1. add your real dataset
2. train the model
3. measure precision, recall, and mAP50
4. tune for Raspberry Pi speed

That is the step that will decide whether the project actually crosses `90%+`.
