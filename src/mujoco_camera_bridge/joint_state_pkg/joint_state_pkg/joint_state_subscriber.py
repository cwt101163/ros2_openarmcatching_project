#!/usr/bin/env python3
# openarm_vis_min.py  —— 保证能转视角
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import mujoco
import glfw
import queue
import threading
import signal
import sys

# ---------- ROS 2 ----------
class JointSub(Node):
    def __init__(self, q):
        super().__init__('joint_sub')
        self.q = q
        self.create_subscription(JointState, '/joint_states', self.cb, 10)

    def cb(self, msg: JointState):
        idx = [i for i, n in enumerate(msg.name) if n.startswith('openarm_')]
        if idx:
            self.q.put([msg.position[i] for i in idx])


# ---------- MuJoCo + GLFW ----------
def main():
    rclpy.init()
    q = queue.Queue(maxsize=3)
    node = JointSub(q)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    # 1. 模型
    model = mujoco.MjModel.from_xml_path('/home/ldp/ros2_ws/openarm_mujoco/v1/scene.xml')
    data  = mujoco.MjData(model)

    # 2. GLFW 初始化
    if not glfw.init():
        raise RuntimeError('glfw.init() failed')
    window = glfw.create_window(1200, 900, 'openarm_vis', None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError('glfw window failed')

    # ★★★ 关键顺序：先 make_context，再注册回调 ★★★
    glfw.make_context_current(window)

    # 3. 相机
    cam, opt, scene, con = mujoco.MjvCamera(), mujoco.MjvOption(), \
                           mujoco.MjvScene(model, 1000), \
                           mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    cam.distance = 2.5

    # 4. 交互变量
    button_left = button_right = False
    last_x, last_y = 0, 0

    def mouse_button_cb(win, button, act, mods):
        nonlocal button_left, button_right
        button_left  = (button == glfw.MOUSE_BUTTON_LEFT  and act == glfw.PRESS)
        button_right = (button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS)

    def cursor_pos_cb(win, x, y):
        nonlocal last_x, last_y
        dx, dy = x - last_x, y - last_y
        last_x, last_y = x, y
        if button_left:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx, dy, scene, cam)
        elif button_right:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_PAN_H,    dx, dy, scene, cam)

    def scroll_cb(win, xoff, yoff):
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM_H, 0, -yoff, scene, cam)


    # ★★★ 注册回调 ★★★
    glfw.set_mouse_button_callback(window, mouse_button_cb)
    glfw.set_cursor_pos_callback(window, cursor_pos_cb)
    glfw.set_scroll_callback(window, scroll_cb)

    # 5. 主循环
    while not glfw.window_should_close(window):
        # 读最新角度
        try:
            angles = q.get_nowait()
            data.qpos[:len(angles)] = angles
        except queue.Empty:
            pass
        mujoco.mj_forward(model, data)

        # 渲染
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, con)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()
    rclpy.shutdown()


# 新增入口函数，供 setup.py 调用
def main_entry():
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    main()