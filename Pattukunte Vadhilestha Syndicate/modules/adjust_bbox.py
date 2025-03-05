import cv2
import numpy as np

def adjust_bbox(image, bbox):
    """Adjust the bounding box by dragging and save the warped perspective image."""
    points = []
    dragging_point = None

    def mouse_callback(event, x, y, flags, param):
        """Mouse callback function to adjust bounding box using draggable points."""
        nonlocal points, dragging_point

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (px, py) in enumerate(points):
                if abs(x - px) < 10 and abs(y - py) < 10:
                    dragging_point = i
                    break

        elif event == cv2.EVENT_MOUSEMOVE and dragging_point is not None:
            points[dragging_point] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            dragging_point = None

    # Initialize points using the bounding box
    x1, y1, x2, y2 = bbox
    points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    cv2.namedWindow("Adjust Bounding Box")
    cv2.setMouseCallback("Adjust Bounding Box", mouse_callback)

    while True:
        display_frame = image.copy()
        for (px, py) in points:
            cv2.circle(display_frame, (px, py), 5, (0, 0, 255), -1)

        cv2.line(display_frame, points[0], points[1], (255, 0, 0), 2)
        cv2.line(display_frame, points[1], points[3], (255, 0, 0), 2)
        cv2.line(display_frame, points[3], points[2], (255, 0, 0), 2)
        cv2.line(display_frame, points[2], points[0], (255, 0, 0), 2)

        cv2.imshow("Adjust Bounding Box", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save the warped image
            x_min = min(points[0][0], points[1][0], points[2][0], points[3][0])
            y_min = min(points[0][1], points[1][1], points[2][1], points[3][1])
            x_max = max(points[0][0], points[1][0], points[2][0], points[3][0])
            y_max = max(points[0][1], points[1][1], points[2][1], points[3][1])

            pts1 = np.float32(points)
            target_width = x_max - x_min
            target_height = y_max - y_min
            pts2 = np.float32([
                [0, 0], [target_width, 0], [0, target_height], [target_width, target_height]
            ])

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(image, matrix, (target_width, target_height))

            cv2.destroyAllWindows()
            return warped

        elif key == ord('q'):  # Quit without saving
            cv2.destroyAllWindows()
            return None