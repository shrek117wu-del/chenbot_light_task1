import sys
from vedo import Plotter, load

def main():
    # 检查命令行参数，若未提供文件则提示
    if len(sys.argv) < 2:
        print("用法: python view_3d.py <模型文件.obj>")
        print("支持 .obj / .mtl 等格式，vedo 会自动加载关联的材质文件。")
        sys.exit(1)

    file_path = sys.argv[1]

    # 创建绘图器
    plt = Plotter(title="3D 模型查看器 (旋转/缩放/平移)")

    # 加载模型（自动处理 .mtl 材质）
    mesh = load(file_path)
    if mesh is None:
        print(f"错误：无法加载文件 {file_path}")
        sys.exit(1)

    # 将模型添加到场景
    plt += mesh

    # 显示说明文字
    plt += __doc__  # 或者自定义文本

    # 启动交互窗口（默认支持鼠标拖拽旋转、滚轮缩放、Shift+拖拽平移）
    plt.show(interactive=True)

if __name__ == "__main__":
    main()