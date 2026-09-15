from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time


MODEL_PATH = "model_ncnn"

CONF = 0.4
IOU = 0.4
IMG_SIZE = 320

# Ejecutar YOLO cada N frames
INFERENCE_INTERVAL = 5


# -----------------------------
# Prioridades
# -----------------------------

priorities = {
    1: ["botella"],
    2: ["lata"],
    3: ["vaso"]
}


# -----------------------------
# Función de prioridades
# -----------------------------

def get_class_by_priority(boxes, priorities, class_names):

    # nombre -> ID
    name_to_id = {v: k for k, v in class_names.items()}

    for _, class_names_priority in sorted(priorities.items()):

        class_ids = {
            name_to_id[name]
            for name in class_names_priority
            if name in name_to_id
        }

        mask = [
            int(cls) in class_ids
            for cls in boxes.cls
        ]

        if any(mask):
            return boxes[mask]

    # No se ha encontrado ninguna clase
    return boxes[:0]


# -----------------------------
# Cámara
# -----------------------------

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (1280, 720),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()


# -----------------------------
# Modelo NCNN
# -----------------------------

model = YOLO(MODEL_PATH)

class_names = model.names


# -----------------------------
# Warm-up
# -----------------------------

frame = picam2.capture_array()

model.predict(
    source=frame,
    imgsz=IMG_SIZE,
    conf=CONF,
    iou=IOU,
    verbose=False
)

# -----------------------------
# Bucle
# -----------------------------

frame_count = 0
filtered_boxes = []

prev_time = time.time()


while True:

    # RGB
    frame = picam2.capture_array()

    # -------------------------
    # Inferencia cada 5 frames
    # -------------------------

    if frame_count % INFERENCE_INTERVAL == 0:

        result = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=IOU,
            verbose=False
        )[0]

        boxes = result.boxes

        filtered_boxes = get_class_by_priority(
            boxes,
            priorities,
            class_names
        )


    # -------------------------
    # Dibujar detección
    # -------------------------

    for box in filtered_boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = class_names[class_id]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{class_name} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # -------------------------
    # FPS
    # -------------------------

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # -------------------------
    # Mostrar
    # -------------------------

    # frame es RGB, OpenCV necesita BGR
    display = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    cv2.imshow("YOLO NCNN", display)


    # ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


    frame_count += 1


picam2.stop()
cv2.destroyAllWindows()
