import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建网格点
x = np.linspace(-2.0, 2.0, 100)
z = np.linspace(-2.0, 2.0, 100)
X, Z = np.meshgrid(x, z)

# 碗形曲面方程：碗底最低点在 (0, -1.0, 0)
# 使用抛物面 Y = X^2 + Z^2 - 1.0
Y = X**2 + Z**2 - 1.0

# 创建图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

# 标记原点 (0,0,0)
ax.scatter(0, 0, 0, color='red', s=50, label='Origin (0,0,0)')

# 标记碗底中心 (0, -1.0, 0)
ax.scatter(0, -1.0, 0, color='blue', s=50, label='Bowl center (0, -1.0, 0)')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Radial Bowl Shape with Center at (0, -1.0, 0)')

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Height (Y)')

# 设置图例
ax.legend()

# 调整视角，便于观察碗底偏移
ax.view_init(elev=25, azim=-60)

plt.show()
# 若需保存图片，取消下一行注释
# plt.savefig('radial_bowl.png', dpi=150)