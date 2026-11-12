"""Builds the Blender scene for the low-poly 3D portrait from the files written by `python -m algofilms.lowpoly3d`.

  Blender -b --python blender/lowpoly_scene.py -- <lowpoly3d_out_dir> [accent_hex]

Saves <dir>/scene.blend: mesh with one colour per triangle, the fly-in / inflate animation as a Mesh Cache (PC2),
a key light, a coloured rim light and a camera that orbits once the picture has become 3D. EEVEE, 1080x1920."""
import math
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
D = os.path.abspath(argv[0])
ACC = argv[1] if len(argv) > 1 else "#d7263d"
acc = tuple(int(ACC.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
z = np.load(os.path.join(D, "mesh.npz"))
verts, cols = z["verts"], z["colors"]
FPS, F, aspect = int(z["fps"]), int(z["frames"]), float(z["aspect"])
N = len(verts)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
me = bpy.data.meshes.new("Portrait")
me.from_pydata(verts.tolist(), [], [(i, i + 1, i + 2) for i in range(0, N, 3)])
me.update()
ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
lin = np.where(cols <= 0.04045, cols / 12.92, ((cols + 0.055) / 1.055) ** 2.4)           # sRGB -> linear
ca.data.foreach_set("color", np.concatenate([lin, np.ones((N, 1))], 1).astype(np.float32).ravel())
ob = bpy.data.objects.new("Portrait", me)
sc.collection.objects.link(ob)

mat = bpy.data.materials.new("Facets"); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes["Principled BSDF"]
attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "Col"
nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
nt.links.new(attr.outputs["Color"], bsdf.inputs["Emission Color"])
bsdf.inputs["Emission Strength"].default_value = 0.35      # keeps the photo's colours readable in the dark
bsdf.inputs["Roughness"].default_value = 0.5
mat.use_backface_culling = False
me.materials.append(mat)

mc = ob.modifiers.new("Assemble", "MESH_CACHE")
mc.cache_format = "PC2"; mc.filepath = os.path.join(D, "assemble.pc2")
mc.frame_start = 1; mc.time_mode = "FRAME"; mc.play_mode = "SCENE"

# camera on a pivot so it can orbit the portrait
pivot = bpy.data.objects.new("Pivot", None); sc.collection.objects.link(pivot)
pivot.location = (0, 0, 0.05)
cam_d = bpy.data.cameras.new("Cam"); cam_d.lens = 85; cam_d.sensor_fit = "VERTICAL"; cam_d.sensor_height = 36
cam = bpy.data.objects.new("Cam", cam_d); sc.collection.objects.link(cam); sc.camera = cam
cam.parent = pivot; cam.rotation_euler = (math.pi / 2, 0, 0)
fr = lambda s: int(round(s * FPS)) + 1
for t, dist in ((0, 11.6), (5.0, 10.2), (F / FPS, 9.6)):
    cam.location = (0, -dist, 0); cam.keyframe_insert("location", frame=fr(t))
for t, yaw, pitch in ((0, 0, 0), (5.3, 0, 0), (6.9, 26, -4), (8.6, -22, 3), (F / FPS, -8, 0)):
    pivot.rotation_euler = (math.radians(pitch), 0, math.radians(yaw)); pivot.keyframe_insert("rotation_euler", frame=fr(t))

def light(name, kind, loc, energy, color, size=3.0):
    d = bpy.data.lights.new(name, kind); d.energy = energy; d.color = color
    if kind == "AREA":
        d.size = size
    o = bpy.data.objects.new(name, d); sc.collection.objects.link(o); o.location = loc
    c = o.constraints.new("TRACK_TO"); c.target = pivot; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    return o

light("Key", "AREA", (-3.2, -5.0, 3.0), 900, (1.0, 0.96, 0.9), 4.0)
light("Rim", "AREA", (3.6, 1.2, 1.5), 650, acc, 2.5)
light("Fill", "AREA", (2.5, -5.5, -1.5), 180, (0.75, 0.82, 1.0), 5.0)

w = bpy.data.worlds.new("W"); w.use_nodes = True; sc.world = w
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.004, 0.006, 1)

for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
    try:
        sc.render.engine = eng; break
    except TypeError:
        pass
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1080, 1920, 100
sc.render.fps = FPS; sc.frame_start, sc.frame_end = 1, F
sc.view_settings.view_transform = "Standard"
sc.render.image_settings.file_format = "PNG"
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(D, "scene.blend"))
print("SAVED", os.path.join(D, "scene.blend"), "engine", sc.render.engine, "tris", N // 3, "frames", F)
