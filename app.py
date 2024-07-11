import cv2
import torch
import time as tm
from ultralytics import YOLO
import supervision as sv


def process_frame(frame, model, shape_annotator, text_annotator, object_dict, device):
    # Analyse the frame using the model
    model_result = model.predict(frame, device=device)[0]

    # convert the model output into a supervision.detections object
    detections = sv.Detections.from_ultralytics(model_result)

    # build a set of labels for the objects detected
    labels = [
        f"{model.model.names[class_id]} {int(confidence * 100)}%"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # build a frame using the vanilla image annotated with the text labels built above
    annotated_frame_1 = text_annotator.annotate(
        scene=frame, labels=labels,
        detections=detections)

    # build a frame using the frame annotated with text using the shape annotator to outline the identified object
    annotated_frame_2 = shape_annotator.annotate(
        scene=annotated_frame_1, detections=detections
    )

    # count the objects in the frame
    # reset the dictionary
    for class_name, count in object_dict.items():
        object_dict[class_name] = 0

    for label in labels:
        class_name, confidence = label.split(" ")
        if class_name in object_dict:
            object_dict[class_name] += 1
        else:
            object_dict[class_name] = 1

    count_text = ""
    for class_name, count in object_dict.items():
        count_text += f", {class_name} {count:2} "

    if count_text != "":
        count_text = count_text[2:]

    # Add this text to the frame
    cv2.putText(annotated_frame_2, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated_frame_2, labels, detections


def get_video_stream(filename):
    print("Opening the video stream...")
    video_capture = cv2.VideoCapture(filename)

    if not video_capture.isOpened():
        print("ERROR: Unable to open video feed")
        Exception("Unable to open video feed")
    return video_capture


def get_annotators():
    print("Setting up annotators...")
    text_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT, text_thickness=1)
    shape_annotator = sv.BoundingBoxAnnotator()
    return text_annotator, shape_annotator


def process_video_stream(video_capture, model, shape_annotator, text_annotator, device):
    print("Processing video stream...")

    prev_frame_time = 0
    frame_period = int(1e9 / video_capture.get(cv2.CAP_PROP_FPS))

    object_dict = {}
    capture_success, frame = video_capture.read()
    while capture_success:
        annotated_frame, labels, detections = process_frame(frame, model, shape_annotator, text_annotator, object_dict, device)

        time_diff = frame_period - (tm.time_ns() - prev_frame_time)
        if time_diff > 0:
            tm.sleep(time_diff / 1e9)
        prev_frame_time = tm.time_ns() + 1e3

        cv2.imshow("Python AI Video Viewer", annotated_frame)
        if cv2.waitKey(1) == 27:
            break
        capture_success, frame = video_capture.read()


def main(filename, model):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    video_capture = get_video_stream(filename)

    try:
        text_annotator, shape_annotator = get_annotators()
        start = tm.time_ns()
        process_video_stream(video_capture, model, shape_annotator, text_annotator, device)

    except Exception as e:
        print(f"Error occurred during processing: {e}")

    finally:
        end = tm.time_ns()
        print("vid_time:", (end - start) / 1e9)
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model = YOLO(".venv/yolov8n.pt")
    main(".venv/demo2.mp4", model)
