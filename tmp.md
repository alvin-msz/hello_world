https://developer.horizon.auto/blog/10003 我将训练坐标系（预训练也进行了同样的操作）从lidar坐标系统一到ego坐标系进行训练，发现路上方占用格顶部大片误检测的现象更明显了，这是为什么？这套框架为什么偏偏要在lidar坐标系而不是在ego坐标系下训练？

是的，这个现象很符合**坐标系只改了标签或只改了部分流程、但没有把整个几何链路严格同步迁移**的情况。这里的“在 LiDAR 坐标系训练”并非 OCC 理论上的硬性要求；地平线参考方案这样做，是为了让相机流、LiDAR 流、检测头和 Occ Head 共享同一个、且与 CenterPoint 点云栅格天然一致的 BEV 坐标框架。你切到 ego 后，上方大块误检更明显，通常说明 \(z\) 向定义或变换链路出现了不一致。 [developer.horizon](https://developer.horizon.auto/blog/10003)

## 为什么原框架选 LiDAR 坐标系

该方案的 LiDAR 流采用 CenterPoint：点云经过 pillar 化、scatter 后直接形成 LiDAR 坐标系下的 BEV 特征；相机 BEVFormer 流也被专门“转换到 LiDAR 坐标系”，随后二者融合，检测和 OCC 共用融合特征。OCC 的原始标签在 ego 坐标系，因此文档明确要求将其转换到 LiDAR 坐标系。 [developer.horizon](https://developer.horizon.auto/blog/10003)

其核心不是“LiDAR 坐标系更正确”，而是**特征、标签、网格和解码结果必须在同一个坐标系里**：

\[
\text{Camera BEV}_{L}
\;\oplus\;
\text{LiDAR BEV}_{L}
\rightarrow
\text{Occ prediction}_{L}
\overset{\text{loss}}{\longleftrightarrow}
\text{Occ GT}_{L}
\]

若统一成 ego，则需要完整变为：

\[
\text{Camera BEV}_{E}
\;\oplus\;
\text{LiDAR BEV}_{E}
\rightarrow
\text{Occ prediction}_{E}
\overset{\text{loss}}{\longleftrightarrow}
\text{Occ GT}_{E}
\]

而不是只把 Occ GT 改到 ego。特别是 CenterPoint 的 voxelization、point-cloud range、pillar index、检测框标注、BEV augmentation、相机 projection/reference points、时序 ego-motion 以及 Occ 可视化，都必须一起转换。

另外，LiDAR 坐标系通常更贴近点云传感器的原始测量坐标；因此它能避免在点云进入 voxel/pillar 前再引入一次固定外参变换与数值误差，也便于复用该框架已预训练的 CenterPoint 权重。地平线文档也建议分别加载相机流和 LiDAR 流预训练权重后联合训练。 [developer.horizon](https://developer.horizon.auto/blog/10003)

## 为什么会出现“上方大片误检”

最值得关注的是：**ego 原点不一定与 LiDAR 原点重合，而且两者一般存在固定平移，尤其是 \(z\) 方向。**

设 ego 到 LiDAR 的刚体变换为：

\[
\mathbf p_L=\mathbf R_{L\leftarrow E}\mathbf p_E+\mathbf t_{L\leftarrow E}
\]

如果车辆标定中 LiDAR 的安装高度为 \(h\)，那么即便旋转近似单位阵，也会有：

\[
z_L=z_E+t_z
\]

对 OCC 而言，\(z\) 轴被离散为多个高度 bin。若你的标签已转为 ego，但网络输出网格、LiDAR pillar 特征或 decode 仍按 LiDAR 的 \(z_{\min}\)、\(z_{\max}\) 和 voxel index 解释，则整张 \(z\)-slice 会发生错位。模型为了拟合错位的监督，最常见的结果不是“整体目标平移得非常整齐”，而是物体顶面、边缘和弱证据区域被扩散到更高的体素层，视觉上就像道路上方出现大片其他类误占据。

常见具体原因如下：

| 可能不一致处 | 为什么会导致上方伪占据 |
|---|---|
| 仅将 Occ GT 转至 ego | 融合 BEV 与预测仍隐含 LiDAR 原点，GT 与 logits 在空间上错位 |
| 未变更 LiDAR 点云预处理 | pillar/voxel index 仍以 LiDAR range 和 LiDAR 原点建立，特征不在 ego 网格 |
| 未更新 BEVFormer reference points | 相机特征投影仍对应 LiDAR-BEV query，却用 ego-OCC 标签监督 |
| \(z_{\min},z_{\max},\Delta z\) 未同步 | 固定高度偏移会被量化为一个或多个错误 \(z\)-bin |
| 标签变化但继续加载 LiDAR 预训练 | 预训练空间语义先验对应 LiDAR BEV，微调数据监督突然移位，早期或弱监督体素更易发散 |
| 时序变换混用 | prev-BEV 的 ego-motion 可能是 \(E_t\leftarrow E_{t-1}\)，而特征或标签却在 \(L_t\) 定义，导致跨帧堆叠出现空间重影 |
| 旋转增强变换顺序错误 | 增强前后 ego↔LiDAR 外参应严格随点、框、标签和相机投影共同处理，否则会有高度或平面偏差 |

地平线方案中，相机流经过 BEVFormer 得到 BEV 特征，LiDAR 流使用 CenterPoint，二者在融合模块中 concat 并做通道注意力融合；若两流并非同一空间定义，融合会把本来正确的单流证据变成冲突特征。 [developer.horizon](https://developer.horizon.auto/blog/10003)

## 一个容易忽略的关键点

对于**纯 BEV 任务**，ego 与 LiDAR 往往在平面上几乎平行，因此很多问题在俯视图中不明显；但 OCC 是 \(x,y,z\) 的三维体素任务，固定的传感器安装高度会在侧视图中被直接放大。

例如：

- LiDAR 高度相对于 ego 原点存在约 1.5 m 偏移；
- OCC 垂直体素分辨率是 0.2 m；
- 若变换方向错了或漏掉平移，则会产生约 \(1.5/0.2=7.5\) 个高度 bin 的误差。

即便最终只残留 1–2 个 bin 的偏差，也足以让物体顶部或路面上方产生明显“浮层”；而由 C2H 直接将 BEV channel 重塑成高度层的 Occ Head，会让这种垂直层错位尤其直观。该参考实现的 Occ Head 正是以 FlashOcc 的通道转高度重塑方式设计。 [developer.horizon](https://developer.horizon.auto/blog/10003)

## 如何正确迁移到 ego

如果你的目的只是改善 OCC 表示，**不建议先改坐标系**。优先保持官方 LiDAR 坐标系，先修复误检；它是和该预训练模型、CenterPoint 流、BEVFormer-to-LiDAR 转换以及部署配置一致的低风险路径。 [developer.horizon](https://developer.horizon.auto/blog/10003)

若必须全流程使用 ego 坐标系，建议按以下顺序完成迁移：

1. **从原始点开始变换。** 在 voxelization / pillarization 前，把当前帧点云由 LiDAR 转到 ego：
   \[
   \mathbf p_E=\mathbf T_{E\leftarrow L}\mathbf p_L
   \]
   然后在 ego 的 `point_cloud_range` 与 voxel size 下建立 pillar/grid；不能只变 Occ label。

2. **重定义全局 BEV 网格。** 统一 ego 原点、`pc_range`、\(x/y/z\) 轴正方向、BEV resolution、Occ ROI、`z_min/z_max` 和每层 voxel center。检测框、OCC GT、LiDAR pillar 特征、相机 BEV query 的物理 cell center 必须一一相同。

3. **重新生成相机几何。** 不能只替换一个 `lidar2ego` 矩阵名称。应检查图像投影用的链路是否真正是：
   \[
   \mathbf p_C
   =
   \mathbf T_{C\leftarrow E}\mathbf p_E
   \]
   原来若是：
   \[
   \mathbf p_C
   =
   \mathbf T_{C\leftarrow L}\mathbf p_L
   \]
   则需要用外参复合得到：
   \[
   \mathbf T_{C\leftarrow E}
   =
   \mathbf T_{C\leftarrow L}\mathbf T_{L\leftarrow E}
   \]
   并且对每个 camera、每帧都验证矩阵方向。

4. **同步时序坐标。** 相邻帧的 LiDAR 点或历史 BEV 在 ego 表示下，应使用对应时刻的 ego 位姿进行 \(E_t \leftarrow E_{t-1}\) 变换；不要混用传感器外参与全局车体位姿。

5. **重新初始化或充分微调。** 由于预训练的相机/LiDAR BEV 表示带有 LiDAR 网格先验，坐标系迁移后建议至少重置 Occ Head 和融合层；若变更了 grid 定义或投影模块，LiDAR encoder、相机 BEV encoder 也应重新训练或采用足够长的低学习率迁移训练。仅“小 epoch 微调”常会保留旧坐标先验。

6. **先做几何单元测试再训练。** 任取同一个三维点、3D box center、Occ voxel center，验证它在 LiDAR、ego、各相机像素和最终 BEV index 中的对应关系。重点输出并核验：
   - `T_ego_from_lidar` 与 `T_lidar_from_ego` 是否互逆；
   - 变换前后点的 \(z\) 值；
   - GT occupied voxel 的世界中心；
   - 预测 grid cell 的物理中心；
   - 可视化工具采用的 frame 与 `origin`。

## 最快的定位实验

在不改网络的情况下，对同一个 batch 做以下三组实验：

- **A：官方链路**：输入、特征、预测、标签均在 LiDAR frame  
- **B：只把 GT 转 ego**：若误检明显增加，基本可确认 supervision frame 与输出 frame 不一致  
- **C：输入点云、相机 reference points、BEV grid、GT 全部改 ego**：若仍出现固定高度偏差，优先检查 \(T_{E\leftarrow L}\) 的方向及 \(t_z\)、\(z\)-bin 原点和可视化 frame

再画出 GT 与 prediction 的侧视 \(x\)-\(z\) 切片，并按高度计算每层的 FP。如果误检峰值相对真实表面稳定高出固定的 \(\Delta z\)，则近似可反推错误来源：

\[
\Delta z_{\mathrm{observed}}
\approx
t_z^{\mathrm{missed}}
+
\Delta z_{\mathrm{grid-origin}}
\]

换言之：这不是 ego 坐标系本身导致精度差，而是该参考工程以 LiDAR 为融合锚点实现和预训练；改成 ego 后，只要任意一个模块仍按 LiDAR frame 理解张量，OCC 的垂直层就最先、最明显地暴露错误。