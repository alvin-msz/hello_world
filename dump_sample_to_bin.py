
import os
import pickle
import json
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

NUSCENES_ROOT = '/data01/chenmu/data/nuscenes-full'
PKL_FILE      = os.path.join(NUSCENES_ROOT, 'bevdetv2-nuscenes_infos_val.pkl')
OUTPUT_DIR    = '/data01/chenmu/FlashOCC/sample_bin'
SAMPLE_INDEX  = 0 

FH, FW    = 256, 704
SRC_H, SRC_W = 900, 1600
CAM_NAMES = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',
]
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SCALE_IMGS       = 0.0078125  
SCALE_INTRINS    = 9.81564      
SCALE_EGO2SENS   = 0.00784314  

N_PTS = 30000

os.makedirs(OUTPUT_DIR, exist_ok=True)

def quantize_s8(arr, scale, zero_point=0):
    """Quantize float32 array to int8 using given scale."""
    q = np.round(arr.astype(np.float32) / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)

def save_bin(arr, name, dtype=None):
    path = os.path.join(OUTPUT_DIR, name)
    out = arr.astype(dtype) if dtype else arr
    out.tofile(path)
    print(f'  {name:42s}  shape={str(arr.shape):25s}  dtype={out.dtype}')

print(f'Loading {PKL_FILE} ...')
with open(PKL_FILE, 'rb') as f:
    data = pickle.load(f)
infos = data['infos'] if isinstance(data, dict) else data
info  = infos[SAMPLE_INDEX]
print(f'Sample token : {info.get("token", "N/A")}')
print(f'Timestamp    : {info.get("timestamp", "N/A")}')

resize        = float(FW) / float(SRC_W)          # 704/1600 = 0.44
resize_w      = int(SRC_W * resize)               # 704
resize_h      = int(SRC_H * resize)               # 396
crop_h_offset = int((1 - 0.0) * resize_h) - FH   # 396 - 256 = 140
crop_w_offset = int(max(0, resize_w - FW) / 2)   # 0
crop          = (crop_w_offset, crop_h_offset,
                 crop_w_offset + FW, crop_h_offset + FH)  # (0,140,704,396)
print(f'\nResize {SRC_W}x{SRC_H} -> {resize_w}x{resize_h}  scale={resize:.4f}')
print(f'Crop   {crop}  ->  {FW}x{FH}')

imgs_list        = []
sensor2egos_list = []
ego2globals_list = []
intrins_list     = []

for cam_name in CAM_NAMES:
    cam_data = info['cams'][cam_name]

    img_path = cam_data['data_path']
    if not os.path.exists(img_path):
        rel = img_path.split('nuscenes/')[-1] if 'nuscenes/' in img_path else img_path
        img_path = os.path.join(NUSCENES_ROOT, rel)
    assert os.path.exists(img_path), f'Image not found: {img_path}'

    img = Image.open(img_path).convert('RGB')
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    img = img.crop(crop)
    assert img.size == (FW, FH)

    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - IMG_MEAN) / IMG_STD   # (H, W, 3)
    img_np = img_np.transpose(2, 0, 1)       # (3, H, W)
    imgs_list.append(img_np)

    K     = np.array(cam_data['cam_intrinsic'], dtype=np.float32)
    K_adj = K.copy()
    K_adj[0, 0] *= resize
    K_adj[1, 1] *= resize
    K_adj[0, 2]  = K[0, 2] * resize - crop_w_offset
    K_adj[1, 2]  = K[1, 2] * resize - crop_h_offset
    intrins_list.append(K_adj)

    w, x, y, z = cam_data['sensor2ego_rotation']
    rot  = Quaternion(w, x, y, z).rotation_matrix.astype(np.float32)
    tran = np.array(cam_data['sensor2ego_translation'], dtype=np.float32)
    s2e  = np.eye(4, dtype=np.float32)
    s2e[:3, :3] = rot;  s2e[:3, 3] = tran
    sensor2egos_list.append(s2e)
    
    w, x, y, z = cam_data['ego2global_rotation']
    rot  = Quaternion(w, x, y, z).rotation_matrix.astype(np.float32)
    tran = np.array(cam_data['ego2global_translation'], dtype=np.float32)
    e2g  = np.eye(4, dtype=np.float32)
    e2g[:3, :3] = rot;  e2g[:3, 3] = tran
    ego2globals_list.append(e2g)

imgs_f32    = np.stack(imgs_list,        axis=0)[np.newaxis]   
sensor2egos = np.stack(sensor2egos_list, axis=0)[np.newaxis]   # (1,6,4,4)
ego2globals = np.stack(ego2globals_list, axis=0)[np.newaxis]   # (1,6,4,4)
intrins_f32 = np.stack(intrins_list,     axis=0)[np.newaxis]   # (1,6,3,3)
post_rots   = np.tile(np.eye(3, dtype=np.float32), (1,6,1,1)) # (1,6,3,3)
post_trans  = np.zeros((1, 6, 3), dtype=np.float32)            # (1,6,3)
bda_rot     = np.eye(3, dtype=np.float32)[np.newaxis]          # (1,3,3)
ego2sensors_f32 = np.linalg.inv(sensor2egos).astype(np.float32) # (1,6,4,4)

lidar_path = info['lidar_path']
if not os.path.exists(lidar_path):
    rel = lidar_path.split('nuscenes/')[-1] if 'nuscenes/' in lidar_path else lidar_path
    lidar_path = os.path.join(NUSCENES_ROOT, rel)
assert os.path.exists(lidar_path), f'LiDAR not found: {lidar_path}'

pts_raw     = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
pts_xyz     = pts_raw[:, :3]
pts_int     = pts_raw[:, 3:4]
l2e_rot     = Quaternion(info['lidar2ego_rotation']).rotation_matrix.astype(np.float32)
l2e_tran    = np.array(info['lidar2ego_translation'], dtype=np.float32)
pts_xyz_ego = pts_xyz @ l2e_rot.T + l2e_tran
pts_ego_raw = np.concatenate([pts_xyz_ego, pts_int], axis=1).astype(np.float32)

N_raw = pts_ego_raw.shape[0]
print(f'\nLiDAR raw pts : {N_raw}  ->  padded/truncated to {N_PTS}')
if N_raw >= N_PTS:
    pts_ego = pts_ego_raw[:N_PTS]
else:
    pad     = np.zeros((N_PTS - N_raw, 4), dtype=np.float32)
    pts_ego = np.concatenate([pts_ego_raw, pad], axis=0)

imgs_s8       = quantize_s8(imgs_f32,       SCALE_IMGS) 
intrins_s8    = quantize_s8(intrins_f32,    SCALE_INTRINS)  
ego2sensors_s8= quantize_s8(ego2sensors_f32,SCALE_EGO2SENS)

# ─── Dump ─────────────────────────────────────────────────────────────────────
print(f'\nDumping to {OUTPUT_DIR}/')
save_bin(imgs_s8,           'imgs.bin')           # S8   (1,6,3,256,704)
save_bin(sensor2egos,       'sensor2egos.bin')    # F32  (1,6,4,4)
save_bin(ego2globals,       'ego2globals.bin')    # F32  (1,6,4,4)
save_bin(intrins_s8,        'intrins.bin')        # S8   (1,6,3,3)
save_bin(post_rots,         'post_rots.bin')      # F32  (1,6,3,3)
save_bin(post_trans,        'post_trans.bin')     # F32  (1,6,3)
save_bin(bda_rot,           'bda_rot.bin')        # F32  (1,3,3)
save_bin(ego2sensors_s8,    'ego2sensors.bin')    # S8   (1,6,4,4)
save_bin(pts_ego,           'pts_ego.bin')        # F32  (30000,4)

# ─── Metadata JSON ────────────────────────────────────────────────────────────
metadata = {
    'sample_token'    : info.get('token', ''),
    'timestamp'       : info.get('timestamp', ''),
    'sample_index'    : SAMPLE_INDEX,
    'input_size'      : [FH, FW],
    'src_size'        : [SRC_H, SRC_W],
    'resize_scale'    : float(resize),
    'resize_dims'     : [resize_h, resize_w],
    'crop'            : list(crop),
    'img_mean'        : IMG_MEAN.tolist(),
    'img_std'         : IMG_STD.tolist(),
    'cam_names'       : CAM_NAMES,
    'n_pts_fixed'     : N_PTS,
    'n_pts_raw'       : int(N_raw),
    'quantisation'    : {
        'imgs'       : {'type': 'S8', 'scale': SCALE_IMGS,     'zero_point': 0},
        'intrins'    : {'type': 'S8', 'scale': SCALE_INTRINS,  'zero_point': 0},
        'ego2sensors': {'type': 'S8', 'scale': SCALE_EGO2SENS, 'zero_point': 0},
    },
    'bin_shapes' : {
        'imgs'        : list(imgs_s8.shape),
        'sensor2egos' : list(sensor2egos.shape),
        'ego2globals' : list(ego2globals.shape),
        'intrins'     : list(intrins_s8.shape),
        'post_rots'   : list(post_rots.shape),
        'post_trans'  : list(post_trans.shape),
        'bda_rot'     : list(bda_rot.shape),
        'ego2sensors' : list(ego2sensors_s8.shape),
        'pts_ego'     : list(pts_ego.shape),
    },
    'bin_dtypes' : {
        'imgs': 'int8', 'sensor2egos': 'float32', 'ego2globals': 'float32',
        'intrins': 'int8', 'post_rots': 'float32', 'post_trans': 'float32',
        'bda_rot': 'float32', 'ego2sensors': 'int8', 'pts_ego': 'float32',
    },
    'lidar_path'           : lidar_path,
    'lidar2ego_rotation'   : list(info['lidar2ego_rotation']),
    'lidar2ego_translation': list(info['lidar2ego_translation']),
    'intrins_original'     : {cam: np.array(info['cams'][cam]['cam_intrinsic']).tolist() for cam in CAM_NAMES},
    'intrins_adjusted_f32' : {cam: intrins_list[i].tolist() for i, cam in enumerate(CAM_NAMES)},
    'sensor2egos'          : {cam: sensor2egos_list[i].tolist() for i, cam in enumerate(CAM_NAMES)},
    'ego2globals'          : {cam: ego2globals_list[i].tolist() for i, cam in enumerate(CAM_NAMES)},
    'note': ('post_rots=eye(3), post_trans=zeros, bda=eye(3) — no augmentation. '
             'Intrinsics adjusted for resize+crop. LiDAR in ego coords. '
             'S8 inputs quantized: q=clip(round(x/scale), -128, 127).')
}
with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
print('  metadata.json')

print('\nDone.')
print(f'\nSummary ({OUTPUT_DIR}):')
print('  input[0] imgs.bin          (1,6,3,256,704)  int8    scale=0.0078125')
print('  input[1] sensor2egos.bin   (1,6,4,4)        float32')
print('  input[2] ego2globals.bin   (1,6,4,4)        float32')
print('  input[3] intrins.bin       (1,6,3,3)        int8    scale=9.81564')
print('  input[4] post_rots.bin     (1,6,3,3)        float32 identity')
print('  input[5] post_trans.bin    (1,6,3)          float32 zeros')
print('  input[6] bda_rot.bin       (1,3,3)          float32 identity')
print('  input[7] ego2sensors.bin   (1,6,4,4)        int8    scale=0.00784314')
print('  input[8] pts_ego.bin       (30000,4)        float32 x,y,z,intensity ego')