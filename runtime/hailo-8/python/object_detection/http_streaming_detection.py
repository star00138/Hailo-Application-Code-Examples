#!/usr/bin/env python3
"""HTTP streaming example for running Hailo8 object detection on Raspberry Pi cameras.

This script is intended for Raspberry Pi 5 + IMX708 camera setups. It runs
YOLO-style HEF models (e.g., yolov8s.hef) and serves an MJPEG stream over HTTP
so clients can view annotated frames in real time.
"""
import argparse
import os
import queue
import signal
import threading
from functools import partial
from pathlib import Path

import cv2
from flask import Flask, Response
from loguru import logger
import numpy as np

from common.hailo_inference import HailoInfer
from common.toolbox import (
    CAMERA_RESOLUTION_MAP,
    default_preprocess,
    get_labels,
    load_json_file,
)
from object_detection_post_process import inference_result_handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve Hailo8 object detection results over HTTP (MJPEG)."
    )
    parser.add_argument(
        "-n",
        "--net",
        type=str,
        required=True,
        help="Path to HEF file (e.g., yolov8s.hef).",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        default=str(Path(__file__).parent.parent / "common" / "coco.txt"),
        help="Path to label file for class names.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=str,
        choices=["sd", "hd", "fhd"],
        default="hd",
        help="Camera capture resolution preset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target camera FPS (the sensor may cap the actual rate).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Video capture index exposed by v4l2 (IMX708 often appears as /dev/video0).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="HTTP bind address for the MJPEG server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port for the MJPEG server.",
    )

    args = parser.parse_args()
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"HEF not found: {args.net}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")
    return args


def configure_camera(camera_index: int, resolution: str, fps: int) -> cv2.VideoCapture:
    width, height = CAMERA_RESOLUTION_MAP.get(resolution, (1280, 720))
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open camera index {camera_index}. Ensure v4l2 is enabled for IMX708."
        )
    logger.info(f"Camera opened at {width}x{height} targeting {fps} FPS")
    return cap


def inference_callback(
    completion_info,
    bindings_list: list,
    input_frame: np.ndarray,
    labels: list,
    config_data: dict,
    frame_queue: queue.Queue,
):
    if completion_info.exception:
        logger.error(f"Inference error: {completion_info.exception}")
        return

    for bindings in bindings_list:
        if len(bindings._output_names) == 1:
            infer_results = bindings.output().get_buffer()
        else:
            infer_results = {
                name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                for name in bindings._output_names
            }
        annotated = inference_result_handler(
            input_frame.copy(), infer_results, labels, config_data
        )
        if not frame_queue.full():
            frame_queue.put(annotated)


def run_inference_loop(
    cap: cv2.VideoCapture,
    hailo_infer: HailoInfer,
    labels: list,
    config_data: dict,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
):
    height, width, _ = hailo_infer.get_input_shape()
    logger.info(f"Model expects {width}x{height} input.")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Camera frame grab failed; stopping inference loop.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = default_preprocess(rgb, width, height)
        cb = partial(
            inference_callback,
            input_frame=frame,
            labels=labels,
            config_data=config_data,
            frame_queue=frame_queue,
        )
        hailo_infer.run([processed], cb)

    hailo_infer.close()
    cap.release()
    frame_queue.put(None)


def mjpeg_generator(frame_queue: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        frame = frame_queue.get()
        if frame is None:
            break
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue
        jpg_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
        )


def main():
    args = parse_args()
    labels = get_labels(args.labels)
    config_data = load_json_file(Path(__file__).with_name("config.json"))

    cap = configure_camera(args.camera_index, args.resolution, args.fps)
    hailo_infer = HailoInfer(args.net, batch_size=1)

    frame_queue: queue.Queue = queue.Queue(maxsize=8)
    stop_event = threading.Event()

    infer_thread = threading.Thread(
        target=run_inference_loop,
        args=(cap, hailo_infer, labels, config_data, frame_queue, stop_event),
        daemon=True,
    )
    infer_thread.start()

    app = Flask(__name__)

    @app.route("/video")
    def video():
        return Response(
            mjpeg_generator(frame_queue, stop_event),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/")
    def index():
        return (
            "<html><body><h2>Hailo8 Object Detection Stream</h2>"
            '<img src="/video" /></body></html>'
        )

    def handle_shutdown(signum, frame):
        logger.info("Received shutdown signal; stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info(f"Serving MJPEG on http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, threaded=True)
    stop_event.set()
    infer_thread.join()


if __name__ == "__main__":
    main()
