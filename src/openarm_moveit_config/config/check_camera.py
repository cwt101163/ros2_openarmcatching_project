import mujoco

model = mujoco.MjModel.from_xml_path('openarm_bimanual.xml')

print("Success！")
print(f"Num: {model.ncam}")
for i in range(model.ncam):
    print(f"   - Camera {i}: '{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)}'")