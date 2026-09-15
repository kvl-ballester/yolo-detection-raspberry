from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time


MODEL_PATH = "model_ncnn"

CONF = 0.4
IOU = 0.4
IMG_SIZE = 320


# Camera
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (1280, 720),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()


# Model
model = YOLO(MODEL_PATH)


# Warm-up
frame = picam2.capture_array()

model.predict(
    source=frame,
    imgsz=IMG_SIZE,
    conf=CONF,
    iou=IOU,
    verbose=False
)

# Inference
prev_time = time.time()

while True:

    # Picamera2 returns RGB
    frame = picam2.capture_array()

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        verbose=False
    )

    # plot() returns an image suitable for OpenCV
    annotated = results[0].plot()

    # FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO NCNN", annotated)

    # ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


picam2.stop()
cv2.destroyAllWindows()
