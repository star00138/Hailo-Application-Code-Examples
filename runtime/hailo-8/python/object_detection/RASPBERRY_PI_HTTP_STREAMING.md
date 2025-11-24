# Raspberry Pi 5 HTTP streaming with Hailo-8 object detection

This guide shows how to run YOLO-style HEF files (e.g., `yolov8s.hef`) on a Raspberry Pi 5 with an IMX708 camera and expose the annotated frames over HTTP.

## Prerequisites
- Raspberry Pi OS (Bookworm) with the IMX708 camera enabled.
- Hailo-8 card installed and the Hailo PCIe driver + PyHailoRT installed.
- HEF compiled for Hailo-8 that includes the NMS post-process (already verified with `hailortcli parse-hef`).
- Python 3.9+ virtual environment.

## Enable the camera as a V4L2 source
The HTTP streamer expects the camera to appear as `/dev/video*`.

```bash
# Ensure the V4L2 shim is loaded (Bookworm libcamera stack)
sudo modprobe bcm2835-v4l2
# Optional: confirm the IMX708 node that will be passed to --camera-index
v4l2-ctl --list-devices
```

## Install dependencies
From the repository root:
```bash
cd runtime/hailo-8/python/object_detection
pip install -r requirements.txt
```

## Run HTTP streaming inference
```bash
python http_streaming_detection.py \
  -n /path/to/yolov8s.hef \
  -l /path/to/labels.txt \
  --resolution hd \
  --fps 30 \
  --camera-index 0 \
  --port 8080
```

- `--resolution`: `sd` (640×480), `hd` (1280×720), or `fhd` (1920×1080). The IMX708 sensor can be configured with `libcamera-vid --list-cameras` if you want different formats.
- `--fps`: Requested frame rate; the driver will cap at the sensor limit.
- `--camera-index`: V4L2 device index returned by `v4l2-ctl --list-devices`.

Browse to `http://<pi-ip>:8080/` to view the MJPEG stream. The `/video` endpoint exposes the raw multipart MJPEG feed for use with players such as VLC or ffplay (`ffplay -fflags nobuffer http://<pi-ip>:8080/video`).

## Notes
- The script keeps only the latest annotated frames in a bounded queue to avoid backlog; if your network viewer is slower than the inference loop, frames will be dropped instead of building latency.
- Stop the server with `Ctrl+C`; the signal handler will close the camera and Hailo device cleanly.
- To change the rendering thresholds or visualization style, edit `config.json` in the same directory.
