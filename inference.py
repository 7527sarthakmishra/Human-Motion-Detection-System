"""
Aegis Human Risk Architecture: Biomechanical Keypoint Analysis Pipeline
Implements 17-Joint Human Pose Regression with Spatial Angular Tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO

def calculate_joint_angle(pointA, pointB, pointC):
    """
    Calculates the internal angle between three spatial keypoint vertices (B is the vertex).
    Implements the arctan2 coordinate displacement subtraction formula from Section IV-C.
    """
    xA, yA = pointA
    xB, yB = pointB
    xC, yC = pointC

    # Compute vector angles relative to the horizon
    angle_BA = np.arctan2(yA - yB, xA - xB)
    angle_BC = np.arctan2(yC - yB, xC - xB)

    # Calculate absolute inner deviation angle in degrees
    angle = np.abs(np.degrees(angle_BC - angle_BA))
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

# 1. Instantiate the optimal institutional model variant outlined in the research paper
# YOLOv8m-pose strikes the finest speed-accuracy trade-off on standard hardware frames
model_path = "yolov8m-pose.pt"  
model = YOLO(model_path)

# 2. Configure video processing stream for custom college data collection
video_input_path = "college_capture.mp4"  # Place your college camera video file here
video_output_path = "biomechanical_analysis_output.mp4"

cap = cv2.VideoCapture(video_input_path)

# Extract tracking dimensions from the video file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize output stream writer using standard high-definition compression codecs
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

# COCO 17-Keypoint Index Map for reference during visual overlays
KEYPOINT_LABELS = {
    0: "Nose", 5: "L_Sho", 6: "R_Sho", 7: "L_Elb", 8: "R_Elb",
    9: "L_Wri", 10: "R_Wri", 11: "L_Hip", 12: "R_Hip",
    13: "L_Kne", 14: "R_Kne", 15: "L_Ank", 16: "R_Ank"
}

print(f"🎬 Processing pipeline initialized. Analyzing frames from {video_input_path}...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference pass utilizing the decoupled anchor-free regression layout
    results = model(frame, verbose=False)[0]

    if results.keypoints is not None:
        # Iterate over every single human instance tracked within the capture layer
        for person in results.keypoints:
            # Convert raw coordinates to localized pixel coordinate tensors
            kpts = person.xy.cpu().numpy()[0]  # Array of shape (17, 2)
            confidences = person.conf.cpu().numpy()[0] if person.conf is not None else None

            # Enforce the paper's heuristic visibility threshold to drop background noise
            visible_indices = [i for i in range(len(kpts)) if (confidences[i] > 0.5 if confidences is not None else True)]

            # Left Knee Angle require: Hip (11), Knee (13), Ankle (15)
            # Right Knee Angle requires: Hip (12), Knee (14), Ankle (16)
            
            if 11 in visible_indices and 13 in visible_indices and 15 in visible_indices:
                left_knee_angle = calculate_joint_angle(kpts[11], kpts[13], kpts[15])
                
                # Render explicit numerical overlay on the dynamic Left Knee vertex joint
                cv2.putText(frame, f"L_Knee: {left_knee_angle:.1f} DEG", 
                            (int(kpts[13][0]) + 15, int(kpts[13][1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Ground-truth highlight anchor rendering directly on the knee joint center
                cv2.circle(frame, (int(kpts[13][0]), int(kpts[13][1])), 8, (0, 0, 255), -1)

            if 12 in visible_indices and 14 in visible_indices and 16 in visible_indices:
                right_knee_angle = calculate_joint_angle(kpts[12], kpts[14], kpts[16])
                
                # Render explicit numerical overlay on the dynamic Right Knee vertex joint
                cv2.putText(frame, f"R_Knee: {right_knee_angle:.1f} DEG", 
                            (int(kpts[14][0]) + 15, int(kpts[14][1]) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Ground-truth highlight anchor rendering directly on the knee joint center
                cv2.circle(frame, (int(kpts[14][0]), int(kpts[14][1])), 8, (0, 0, 255), -1)

            # Map standard color-coded connecting paths across joints
            for idx in visible_indices:
                x, y = int(kpts[idx][0]), int(kpts[idx][1])
                # Draw small tracking point on every visible anatomical joint landmark
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                
                # Label joint name strings for portfolio demonstration clarity
                if idx in KEYPOINT_LABELS:
                    cv2.putText(frame, KEYPOINT_LABELS[idx], (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Save the transformed metrics frame live to the disk stream output
    out.write(frame)

# Clean up interfaces cleanly upon sequence completion
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Biomechanical analysis successfully compiled! File stored safely at: {video_output_path}")