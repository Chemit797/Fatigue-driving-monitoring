import cv2
import time
from datetime import datetime
import numpy as np
import mediapipe as mp

# 初始化 MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)

# EAR 和 MAR 的阈值设定
EAR_THRESH = 0.23
MAR_THRESH = 0.78
EYE_AR_CONSEC_FRAMES = 48
print(f"  EAR_THRESH（闭眼阈值）: {EAR_THRESH}")
print(f"  MAR_THRESH（打哈欠阈值）: {MAR_THRESH}")
print(f"  EYE_AR_CONSEC_FRAMES（连续帧数）: {EYE_AR_CONSEC_FRAMES}")
# 特征索引
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [[61, 291], [39, 181], [0, 17], [269, 405]]

# 初始化参数
counter = 0
yawns = 0
score = 0
prev_yawn = False
status_start = time.time()
log_start = time.time()

# 分数项缓存
status_flags = {
    "blink": False,
    "yawn": False,
    "gaze": False
}

# 日志函数
def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("fatigue_log.txt", "a") as f:
        f.write(f"{timestamp} - {message}\n")
    print(f"{timestamp} - {message}")

# EAR 计算
def get_ear(landmarks, eye_idx, w, h):
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (A + B) / (2.0 * C)

# MAR 计算
def get_mar(landmarks, w, h):
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for pair in MOUTH for i in pair]
    N1 = np.linalg.norm(np.array(coords[2]) - np.array(coords[3]))
    N2 = np.linalg.norm(np.array(coords[4]) - np.array(coords[5]))
    N3 = np.linalg.norm(np.array(coords[6]) - np.array(coords[7]))
    D = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
    return (N1 + N2 + N3) / (3 * D)

# 头部方向判断
def check_gaze(pitch, yaw):
    return abs(pitch) > 10 or abs(yaw) > 10

# 视频捕捉
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # 重置分数项
    score = 0
    reasons = []

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        ear_left = get_ear(face_landmarks, LEFT_EYE, w, h)
        ear_right = get_ear(face_landmarks, RIGHT_EYE, w, h)
        avg_ear = (ear_left + ear_right) / 2.0

        mar = get_mar(face_landmarks, w, h)

        # 眨眼判断
        if avg_ear < EAR_THRESH:
            counter += 1
            if counter >= EYE_AR_CONSEC_FRAMES:
                score += 10
                reasons.append("眨眼情况异常")
                status_flags["blink"] = True
        else:
            counter = 0
            status_flags["blink"] = False

        # 打哈欠判断
        if mar > MAR_THRESH:
            if not prev_yawn:
                yawns += 1
                prev_yawn = True
            score += 3
            reasons.append("频繁打哈欠")
            status_flags["yawn"] = True
        else:
            prev_yawn = False
            status_flags["yawn"] = False
        # # 头部方向（视线）判断
        # nose = face_landmarks[1]
        # pitch = face_landmarks[1].y - face_landmarks[199].y
        # yaw_angle = face_landmarks[33].x - face_landmarks[263].x
        # if check_gaze(pitch * 100, yaw_angle * 100):
        #     score += 2
        #     reasons.append("视线偏移")
        #     status_flags["gaze"] = True
        # else:
        #     status_flags["gaze"] = False

    # 每2秒判断是否立即预警
    if time.time() - status_start >= 2:
        if score >= 10:
            log_event("状况异常：" + "+".join(reasons))
        status_start = time.time()

    # 每30秒状态记录
    if time.time() - log_start >= 30:
        if any(status_flags.values()):
            reasons_log = []
            if status_flags["blink"]:
                reasons_log.append("眨眼情况异常")
            if status_flags["yawn"]:
                reasons_log.append("频繁打哈欠")
            if status_flags["gaze"]:
                reasons_log.append("视线偏移")
            log_event("状况异常（周期性统计）：" + "+".join(reasons_log))
        else:
            log_event("正常驾驶中，情况良好")
        log_start = time.time()

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
