import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# -----------------------------
# 配置参数
# -----------------------------
DATA_DIR = 'handeye_data'
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.json')
CHESSBOARD_SIZE = (3, 3)  # 4x4 方格 → 内角点为 3x3
CHESSBOARD_SQUARE_SIZE = 0.025  # 每格 2.5cm
CAMERA_FOVY = 60  # 与XML中一致
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

# 标定板在世界坐标系下的位姿（用于验证）
T_m2w = np.array([
    [1, 0, 0, 0.35],
    [0, 1, 0, 0],
    [0, 0, 1, 0.31],
    [0, 0, 0, 1]
])

# 计算相机内参
focal_length = (IMAGE_HEIGHT / 2) / np.tan(np.radians(CAMERA_FOVY / 2))
K = np.array([
    [focal_length, 0, IMAGE_WIDTH / 2],
    [0, focal_length, IMAGE_HEIGHT / 2],
    [0, 0, 1]
])


def detect_chessboard(image_path):
    """检测棋盘格并返回位姿 (相机坐标系 -> 标定板坐标系)"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告: 无法读取图像 {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FAST_CHECK
    )

    if not ret:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= CHESSBOARD_SQUARE_SIZE

    result = cv2.solvePnP(objp, corners, K, None)

    if len(result) == 3:
        success, rvec, tvec = result
    else:
        success, rvec, tvec, _ = result

    if not success:
        return None

    R_c2m, _ = cv2.Rodrigues(rvec)
    T_c2m = np.eye(4)
    T_c2m[:3, :3] = R_c2m
    T_c2m[:3, 3] = tvec.squeeze()

    return T_c2m


def main():
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    # 存储机器人运动 (gripper to base)
    R_gripper2base = []
    t_gripper2base = []

    # 存储标定板在相机坐标系中的位姿 (target to camera)
    R_target2cam = []
    t_target2cam = []

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= CHESSBOARD_SQUARE_SIZE

    for sample in metadata:
        # 1. 获取末端执行器位姿（MuJoCo 四元数顺序: [w, x, y, z]）
        ee = sample['ee_pose']
        pos = np.array(ee['position'])
        quat_wxyz = np.array(ee['quaternion'])  # [w, x, y, z]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        R_mat = R.from_quat(quat_xyzw).as_matrix()

        R_gripper2base.append(R_mat)
        t_gripper2base.append(pos)

        # 2. 从图像获取 target -> camera
        rgb_path = sample['rgb_path']
        img = cv2.imread(rgb_path)
        if img is None:
            print(f"跳过 {rgb_path}: 图像不存在")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if not ret:
            print(f"跳过 {rgb_path}: 棋盘格未检测到")
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, None)
        if not ret:
            continue

        R_cam, _ = cv2.Rodrigues(rvec)
        R_target2cam.append(R_cam)
        t_target2cam.append(tvec.flatten())

    if len(R_gripper2base) < 3 or len(R_target2cam) < 3:
        raise RuntimeError("有效样本不足")

    # 3. 执行手眼标定（Eye-to-hand）
    try:
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
    except Exception as e:
        print("手眼标定失败:", e)
        raise

    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base.flatten()

    print("\n===== 手眼标定结果 (相机 -> 基座) =====")
    print("Rotation:\n", R_cam2base)
    print("Translation:", t_cam2base.flatten())

    # 4. 验证：将标定板从相机坐标转换到基座
    errors = []
    for i in range(len(R_target2cam)):
        T_target2cam = np.eye(4)
        T_target2cam[:3, :3] = R_target2cam[i]
        T_target2cam[:3, 3] = t_target2cam[i]

        T_target2base_est = T_cam2base @ T_target2cam
        error = np.linalg.norm(T_target2base_est[:3, 3] - T_m2w[:3, 3])
        errors.append(error)
        print(f"Sample {i}: 重投影误差 = {error:.4f} m")

    mean_error = np.mean(errors)
    print(f"平均误差: {mean_error:.4f} m")

    # 5. 保存结果
    result = {
        "T_camera_to_base": T_cam2base.tolist(),
        "mean_reprojection_error_m": float(mean_error),
        "num_valid_samples": len(errors)
    }
    result_json_path = os.path.join(DATA_DIR, 'handeye_result.json')
    with open(result_json_path, 'w') as f:
        json.dump(result, f, indent=2)

    # 6. 可视化：使用正确的变量
    plt.figure(figsize=(10, 5))

    angles = R.from_matrix(R_cam2base).as_euler('xyz', degrees=True)
    plt.subplot(1, 2, 1)
    plt.bar(['X', 'Y', 'Z'], angles)
    plt.title('Rotation Angles (degree)')
    plt.xlabel('Axis')
    plt.ylabel('Angle (degree)')

    plt.subplot(1, 2, 2)
    plt.bar(['X', 'Y', 'Z'], T_cam2base[:3, 3])
    plt.title('Translation Vector (m)')
    plt.xlabel('Axis')
    plt.ylabel('Distance (m)')

    plt.tight_layout()
    result_img_path = os.path.join(DATA_DIR, 'handeye_result.png')
    plt.savefig(result_img_path)

    print(f"\n标定完成！结果已保存至:")
    print(f"   - JSON: {result_json_path}")
    print(f"   - 图像: {result_img_path}")


if __name__ == "__main__":
    main()