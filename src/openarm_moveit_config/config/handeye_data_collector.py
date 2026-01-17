import mujoco
import numpy as np
import json
import os
from PIL import Image

# -----------------------------
# 配置
# -----------------------------
XML_PATH = 'openarm_bimanual.xml'
CAMERA_NAME = 'd435_front'
OUTPUT_DIR = 'handeye_data'
NUM_SAMPLES = 5
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_ee_pose(model, data, ee_body_name='left_ee'):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    pos = data.xpos[body_id].copy()
    quat = data.xquat[body_id].copy()
    return {"position": pos.tolist(), "quaternion": quat.tolist()}

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Renderer 高层 API
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)

    sample_configs = [
        [0.0, -0.5, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.2, -0.7, 0.2, -0.8, 0.1, 0.0, 0.0],
        [-0.2, -0.6, -0.2, -0.9, -0.1, 0.0, 0.0],
        [0.0, -0.4, 0.0, -1.2, 0.0, 0.1, 0.0],
        [0.1, -0.8, 0.1, -0.7, 0.0, -0.1, 0.0],
    ]

    metadata = []

    for i in range(NUM_SAMPLES):
        print(f"采集样本 {i+1}/{NUM_SAMPLES}...")

        if len(data.qpos) >= 7:
            data.qpos[:7] = sample_configs[i]
        else:
            print(f"警告：只有 {len(data.qpos)} 个关节")

        mujoco.mj_forward(model, data)

        # === RGB 渲染===
        renderer.update_scene(data, camera=cam_id)
        rgb_img = renderer.render()  # 直接返回 (H, W, 3) uint8 RGB 图像

        # === Depth 渲染===
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_id)
        depth_img = renderer.render()  # 返回 (H, W) float32 深度图（单位：米）
        renderer.disable_depth_rendering()

        # 保存图像
        rgb_path = os.path.join(OUTPUT_DIR, f"rgb_{i:03d}.png")
        depth_path = os.path.join(OUTPUT_DIR, f"depth_{i:03d}.png")
        Image.fromarray(rgb_img).save(rgb_path)

        # 将深度图转换为可视化图像
        depth_vis = ((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-6) * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(depth_path)

        ee_pose = get_ee_pose(model, data, 'left_ee')
        metadata.append({
            "sample_id": i,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "ee_pose": ee_pose
        })

    # 保存元数据
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n采集完成！数据保存在 '{OUTPUT_DIR}'")

if __name__ == "__main__":
    # 确保环境变量已设置
    import os
    os.environ["MUJOCO_GL"] = "egl"
    
    main()