# Work-Safety-Monitoring Using YOLOv10

This project leverages the cutting-edge capabilities of YOLOv10 for real-time end-to-end object detection to enhance work safety monitoring. By identifying safety equipment such as helmets and vests, the system aims to ensure compliance with safety regulations in various work environments.

## Features

- **Real-Time Detection**: Utilizes YOLOv10 for fast and accurate detection of safety equipment in work environments.
- **Custom Model Support**: Ability to use pretrained models or train custom models tailored to specific safety equipment.
- **Multi-Platform Support**: Includes Docker support for easy deployment across different platforms, including ARM64 and NVIDIA Jetson devices.
- **Comprehensive Documentation**: Detailed examples and tutorials to help users get started with the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Docker (for Docker deployment)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repository/Work-Safety-Monitoring.git
   cd Work-Safety-Monitoring
   ```
2. Install the required pakages:
   ```sh
   pip install -r yolov10/requirements.txt
   ```

## Usage
To use a pretrained model for detection:
```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
model.predict()
```

For exporting models to different formats (ONNX, TensorRT):
```python
model.export(model='jameslahm/yolov10{n/s/m/b/l/x}', format='onnx', opset=13, simplify=True)
```

To run Streamlit demo:
```sh
streamlit run app.py
```

## Docker Deployment
Refer to the Dockerfiles in the yolov10/docker directory for deploying the application using Docker.

## Documentation
For more detailed information on usage and deployment, please refer to the mkdocs documentation and the examples provided in the yolov10/examples directory.

## Citation
If you find this project useful in your research, please consider citing:
```python
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```
