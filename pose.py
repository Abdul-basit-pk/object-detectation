from ultralytics import YOLO
import cv2
import torch

# ----------------------------
# Load models
# ----------------------------
pose_model = YOLO("yolov8n-pose.pt")       # small pose model (wrist/hand keypoints)
object_model = YOLO("yolov8n.pt")          # general object detection
face_model = YOLO("yolov8n-face.pt")       # face detection (forked face model)

# ----------------------------
# Setup device
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
pose_model.to(device)
object_model.to(device)
face_model.to(device)

# ----------------------------
# Open webcam
# ----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # Object detection
    # ----------------------------
    obj_results = object_model(frame)
    annotated_obj = obj_results[0].plot()

    # ----------------------------
    # Face detection
    # ----------------------------
    face_results = face_model(frame)
    annotated_face = face_results[0].plot()

    # ----------------------------
    # Pose detection (hands)
    # ----------------------------
    pose_results = pose_model(frame)
    annotated_pose = annotated_obj.copy()  # start from object annotated frame

    for result in pose_results:
        if result.keypoints is not None:
            keypoints = result.keypoints.cpu().numpy()  # [num_people, num_keypoints, 3]
            for person in keypoints:
                # make sure keypoints exist (wrist: 9=left, 10=right in COCO format)
                if person.shape[0] >= 11:
                    left_wrist = person[9][:2]
                    right_wrist = person[10][:2]

                    # draw circles on wrists (hands)
                    cv2.circle(annotated_pose, tuple(left_wrist.astype(int)), 5, (0,255,0), -1)
                    cv2.circle(annotated_pose, tuple(right_wrist.astype(int)), 5, (0,0,255), -1)

    # ----------------------------
    # Combine annotations
    # ----------------------------
    final_frame = annotated_pose.copy()

    # Draw faces in blue rectangles
    for face in face_results:
        boxes = face.boxes.xyxy.cpu().numpy() if hasattr(face.boxes, "xyxy") else []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(final_frame, (x1,y1), (x2,y2), (255,0,0), 2)
    
    # ----------------------------
    # Show result
    # ----------------------------
    cv2.imshow("Live Detection: Objects + Face + Hands", final_frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
