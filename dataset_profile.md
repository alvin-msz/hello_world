## Q1-20261008

占用高度远小于 8 米时，把 hat.vt_z_follows_occ 设为 true，**vt_virtual_bins，vt_z_span，vt_z_inset**怎么设置，有什么含义？

`samples/ai_toolchain/horizon_model_train_sample/scripts/tools/dataset_profile.py`
`samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevformer_vt_virtual_z.py`
`samples/ai_toolchain/horizon_model_train_sample/scripts/tools/ego_occ_rotate.py`

## A1-20261008

### 四个参数各自的含义

先理解核心机制。HAT 的 `BevFormerViewTransformer.export_reference_points` 会**把 `z_max - z_min`（物理 Z 跨度，单位米）直接塞给 `gen_reference_points` 当作"高度采样层数（bins）"**。官方 nuScenes 雷达高度带正好是 8 米，所以 bins 恰好 = 8、`0.5` 的 inset 恰好也是半层 = 0.5 米，一切自洽。一旦 Z 跨度不是 8 米，这套"米＝层数"就崩了。

`SingleBevFormerViewTransformerVirtualZ`（`scripts/tools/bevformer_vt_virtual_z.py`）就是来解决这个问题的——它改成把 `vt_virtual_bins`（而不是物理米数）传给 `gen_reference_points`。

各参数：

| 参数 | 含义 | 单位 | 默认 |
|---|---|---|---|
| `vt_z_follows_occ` | 开关：物理高度是否直接跟随 OCC 的 Z | bool | `false` |
| `vt_virtual_bins` | Z 方向上采样高度的**虚拟层数**（不是米） | 层 | `8` |
| `vt_z_span` | 目标高度带跨度 | 米 | `8.0` |
| `vt_z_inset` | 采样时两端各内缩的距离 | 层（换算前是"半层"） | `0.5` |

关键点：**`0.5` 是半层，不是 0.5 米**（配置注释里也强调了）。在 `gen_reference_points` 里高度是按归一化 `linspace(0.5, Z-0.5, Z)` 采样的，再反归一化回 `pc_range` 的物理 Z。只有当 bins=8 且物理跨度=8 米时，"半层"才恰好等于 0.5 米。

### 两种模式下的具体作用

去看 dataset_profile.py 的 `_derive()`（第 219–253 行）：

**`vt_z_follows_occ: false`（默认，nuScenes 场景）**

```python
margin = (vt_z_span - occ_z_span) / 2      # (8.0 - 6.4)/2 = 0.8
vt_z = (occ_z_min - margin, occ_z_max + margin)  # (-1.8, 6.2)
```

- 物理高度被**垫**到 8 米（就是把 OCC 的 6.4 米上下各补 0.8）。
- 此分支强制要求 `vt_z_span == 8.0` **且** `vt_virtual_bins == 8`，否则直接报错（第 228–231 行）。
- `vt_z_inset` 只用于校验：`cover = vt_z_span - 2*inset = 8 - 1 = 7`，即 OCC 的 Z 跨度不能超过 7 米。

**`vt_z_follows_occ: true`（占用高度远小于 8 米的场景）**

```python
vt_z = (occ_range[2], occ_range[5])   # 直接用 OCC 的 Z，不垫
```

- 物理高度 = OCC 的实际 Z（比如 `[-1.0, 5.4]`，6.4 米，也可以是任意更矮的）。
- 采样层数由 `vt_virtual_bins` 提供，**不再把米当层数**，所以即使 Z 只有 2 米、4 米也一样正确。
- 此时 `vt_z_span` 和 `vt_z_inset` 的值**不再影响结果**（代码跳过 else 分支的校验），但它们在 yaml 里必须存在（`_need` 无条件读取），`vt_z_inset >= 0` 仍会被检查。

### `vt_z_follows_occ: true` 时怎么设置

推荐保持默认不动即可：

```yaml
hat:
  vt_z_follows_occ: true     # 改成 true
  vt_virtual_bins: 8         # 保持 8：这是采样层数，与物理高度无关
  vt_z_span: 8.0             # 保留（必须写，但 true 时值被忽略）
  vt_z_inset: 0.5            # 保留（必须写，>=0 即可）
```

要点：

1. **唯一需要改的就是 `vt_z_follows_occ: true`**，另外确认 `vt_virtual_bins` 是你想要的采样层数（通常保持 8，与 nuScenes 官方采样行为一致）。
2. `vt_virtual_bins` 语义是"层数"：越大 Z 方向参考点越密（计算量上升），至少为 2。和你 OCC 的 Z 高度（几米）没有绑定关系。
3. `vt_z_span` / `vt_z_inset` 别删（会触发 `_need` 报错），值留默认就行，真正生效的是 `occ.pc_range` 里的 Z。

⚠️ 一个连带影响：`follows_occ: true` 后，LiDAR 的 Z 也随之缩短（`voxel_size` 的 Z 维 = `vt_z[1] - vt_z[0]`，不再是 8），因为代码里 `lidar_range` / `voxel_size`（第 304–313 行）都复用了 `vt_z`。所以旧 checkpoint 的 pillar/voxel Z 尺寸会对不上，需要重训（这和 yaml 顶部注释说的一致）。