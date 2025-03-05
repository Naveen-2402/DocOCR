import cv2
from ultralytics import YOLO

def initialize_camera(cam_number):
    """Initialize the webcam and YOLO model."""
    model = YOLO('C:/Users/Naveen/Desktop/Pattukunte Vadhilestha Syndicate/model-1/model-1.pt')
    cap = cv2.VideoCapture(cam_number)
    return model, cap

def capture_frame(model, cap):
    """Capture a frame, perform YOLO detection, and save the image if 's' is pressed."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                if conf > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {cls}, Conf: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('YOLO Detection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit the program
            cap.release()
            cv2.destroyAllWindows()
            return None, None
        elif key == ord('s'):  # Save the image and return it with the bounding box
            bbox = (x1, y1, x2, y2)  # Bounding box coordinates
            return original_frame, bbox

    cap.release()
    cv2.destroyAllWindows()
    return None, None