# Mirror Cup & Saucer Art — 全面修复计划

## 问题诊断

经过仔细对比论文和代码，发现以下关键问题：

### 问题1：相机参数与论文不符
论文 Section 4 明确说明：
- 相机位置: `(0, -5.5, 5)`
- 直接视角 look-at: `(0, 0, -0.8)`，FOV = **4.5°**
- 反射视角 look-at: `(0, 0, 0.1)`，FOV = **3.0°**
- up 方向应该是 z 轴 `[0, 0, 1]`（z 轴是垂直向上方向）

当前代码中直接视角 FOV=12°, look-at=(0,-1,0.15)，完全不正确，导致渲染视角不对。

### 问题2：texture.png 和 saucer.obj 是空/损坏文件
- `texture.png` 只有 162 字节（损坏/空白）
- `saucer.obj` 是 173KB，但来自旧的默认输出，不是优化结果

### 问题3：3D viewer 中碟子和圆柱坐标系不一致
- Python 坐标系：碟子在 `y∈[-1.5,-0.5]`，圆柱在原点
- Three.js 坐标转换：`v x height -y`，碟子的 z 坐标应在 `[0.5, 1.5]`
- 但圆柱被放在 y=1.0（Three.js），这样碟子会在圆柱"后面"而不是"前面"
- 圆柱应该在场景原点，碟子在圆柱前方(正 Z 轴方向)

### 问题4：Three.js 中 saucer 使用 `side: THREE.DoubleSide` 但纹理映射 UV 可能错误
- OBJ 中 UV 坐标格式没问题，但 Three.js 加载纹理时如果 texture.png 是空白的，什么都看不到

### 问题5：cylinder 没有被添加到 Three.js 场景（只用 cubeCamera 反射，但 saucer 没有纹理）
- viewer.html 中 cylinder.obj 文件没有被加载，只是手工创建了圆柱几何体
- 圆柱内侧反射使用的是 THREE.BackSide，但由于 cubeCamera 和圆柱本身位置/方向混淆，反射无法正确显示场景中的碟子图像

## 修复方案

### 修复1：更正 solver.py 中的相机参数
按论文精确设置：
- `Pd`: cam=(0,-5.5,5), look_at=(0,0,-0.8), fov=4.5°, up=[0,0,1]
- `Pr`: cam=(0,-5.5,5), look_at=(0,0,0.1), fov=3.0°, up=[0,0,1]

### 修复2：修复 viewer.html 的 3D 场景坐标
Three.js Y-up 坐标，Python→Three.js 转换：`(px, py, pz) → (px, pz_height, -py)`
- 碟子 Python y ∈ [-1.5, -0.5] → Three.js z ∈ [0.5, 1.5]（在Z正方向）
- 圆柱在 Three.js 的 (0, 0, 0) 处（从 y=0 到 y=h）
- 摄像机在 Python 的 (0,-5.5,5) → Three.js 的 (0, 5, 5.5)

### 修复3：viewer.html 增加实际 OBJ+texture 加载逻辑修复
- 修正碟子在 Three.js 中的位置和方向
- 添加调试信息（在没有纹理时，至少显示网格）
- 增加 fallback：当 texture.png 加载失败时用顶点色代替

### 修复4：export_utils.py 坐标修正
确保 OBJ 文件里顶点坐标与 Three.js 期望的一致：
- Three.js Y-up，碟子在 Z=[0.5,1.5], X=[-0.5,0.5], Y=height
- 圆柱在 (0,0,0)，高度沿 Y 轴，半径 0.4

### 修复5：添加运行验证脚本
创建 `validate_examples.py` 来验证所有示例（合成字母A/B，Paper Exp 1/2）

## 文件修改清单

### [MODIFY] core/solver.py
- 修正 Pd, Pr 相机矩阵参数

### [MODIFY] viewer.html
- 修正 Three.js 场景坐标系
- 修复摄像机默认位置
- 修复碟子 OBJ 加载后的位置
- 增加 cylinder.obj 加载并赋予反射材质  
- 修复 cubeCamera 位置
- 当纹理加载失败时降级到顶点色

### [MODIFY] export_utils.py
- 确认坐标转换逻辑正确(`v x height -y`)

### [NEW] validate_examples.py
- 快速验证脚本，不跑完整优化，只用 5 次迭代验证 pipeline 能否运行

## 验证计划

1. 运行 `python validate_examples.py` 检查渲染管线
2. 运行 `python main.py --exp 0 --shape plane --res 50 --render_size 128 --iters1 10 --iters2 5 --iters3 5 --no_viewer` 快速测试
3. 启动 `python viewer.py` 并在浏览器中打开检查 3D 场景
