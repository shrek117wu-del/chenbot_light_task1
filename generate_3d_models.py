import numpy as np

def export_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Exported {filename} | Vertices: {len(vertices)}, Faces: {len(faces)}")


def generate_cup(radius_bottom=0.4, radius_top=0.4, height=1.0, segments=128):
    """
    生成一个支持直筒/倒圆台形（圆锥截面）的平滑镜面杯子模型。
    """
    vertices = []
    faces = []
    
    vertices.append([0.0, 0.0, 0.0])
    vertices.append([0.0, 0.0, height])
    bottom_center_idx = 0
    top_center_idx = 1
    
    base_idx = len(vertices)
    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        
        # Bottom ring
        xb = radius_bottom * np.cos(angle)
        yb = radius_bottom * np.sin(angle)
        vertices.append([xb, yb, 0.0])
        
        # Top ring
        xt = radius_top * np.cos(angle)
        yt = radius_top * np.sin(angle)
        vertices.append([xt, yt, height])
        
    for i in range(segments):
        next_i = (i + 1) % segments
        
        b1 = base_idx + i * 2
        t1 = base_idx + i * 2 + 1
        b2 = base_idx + next_i * 2
        t2 = base_idx + next_i * 2 + 1
        
        faces.append([b1, t1, b2])
        faces.append([b2, t1, t2])
        
        faces.append([b1, b2, bottom_center_idx])
        faces.append([t1, top_center_idx, t2])
        
    return vertices, faces


def generate_saucer(pattern_type="stepped", radius_inner=0.42, radius_outer=1.6, rad_segments=180, r_segments=120):
    """
    生成各种风格起伏的茶碟。
    pattern_type: "stepped" (图2阶梯), "wavy" (图3图4连续波浪), "smooth_rim" (图1平滑边缘)
    """
    vertices = []
    faces = []
    
    def profile_z(r):
        if r <= radius_inner:
            return 0.0
            
        t = (r - radius_inner) / (radius_outer - radius_inner)
        base_h = 0.25 * t  # 基本向上的倾斜度
        
        if pattern_type == "stepped":
            # 图2：阶梯状（数量少，有平台感）
            steps = 4 
            ripple = 0.02 * np.sin(t * steps * 2 * np.pi - np.pi/2) + 0.02
        elif pattern_type == "wavy":
            # 图3、4：高频连续波浪状（像涟漪，非常密集）
            waves = 13
            ripple = 0.04 * np.sin(t * waves * 2 * np.pi)
        elif pattern_type == "smooth_rim":
            # 图1：平坦且仅有边缘上翘
            ripple = 0.0
            if t > 0.6:
                ripple = 0.5 * (t - 0.6)**2
        else:
            ripple = 0.0
            
        return base_h + ripple
        
    vertices.append([0.0, 0.0, profile_z(0.0)])
    center_idx = 0
    
    radii = np.linspace(0.01, radius_outer, r_segments)
    
    for r in radii:
        z = profile_z(r)
        for t_idx in range(rad_segments):
            angle = 2.0 * np.pi * t_idx / rad_segments
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            vertices.append([x, y, z])
            
    for t_idx in range(rad_segments):
        next_t = (t_idx + 1) % rad_segments
        faces.append([1 + t_idx, 1 + next_t, center_idx])
        
    for r_idx in range(r_segments - 1):
        for t_idx in range(rad_segments):
            next_t = (t_idx + 1) % rad_segments
            
            p1 = 1 + r_idx * rad_segments + t_idx
            p2 = 1 + r_idx * rad_segments + next_t
            p3 = 1 + (r_idx + 1) * rad_segments + t_idx
            p4 = 1 + (r_idx + 1) * rad_segments + next_t
            
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])
            
    return vertices, faces


if __name__ == "__main__":
    # 1. 杯子类型
    # (a) 标准圆柱直线杯
    v, f = generate_cup(radius_bottom=0.4, radius_top=0.4)
    export_obj("photo_cup_straight.obj", v, f)
    
    # (b) 圆柱圆台形杯 (图1)
    v, f = generate_cup(radius_bottom=0.4, radius_top=0.46)
    export_obj("photo_cup_conical.obj", v, f)
    
    # 2. 碟子起伏类型
    # (a) 阶梯形碟子 (图2. 郁金香款)
    v, f = generate_saucer("stepped")
    export_obj("photo_saucer_stepped.obj", v, f)
    
    # (b) 高频波浪形碟子 (图3/4. 蓝猫/绿鸟款)
    v, f = generate_saucer("wavy")
    export_obj("photo_saucer_wavy.obj", v, f)
    
    # (c) 平坦上坡边缘碟子 (图1. 金色马款)
    v, f = generate_saucer("smooth_rim")
    export_obj("photo_saucer_smooth_rim.obj", v, f)
