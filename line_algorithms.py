import numpy as np

def compute_hom_lines(start_hom_points, end_hom_points):
    hom_lines = np.cross(start_hom_points, end_hom_points)
    return hom_lines / np.linalg.norm(hom_lines)

def to_hom_points(points, cam):
    hom_points = np.ones((len(points), 3))
    hom_points[:,0] = (points[:,0] - cam.cx) / cam.fx
    hom_points[:,1] = (points[:,1] - cam.cy) / cam.fy
    return hom_points

def to_world_planes(hom_lines, transform):
    R = transform[:3,:3]
    t = transform[:3,3:]
    world_planes = np.zeros((len(hom_lines), 4))
    normal = hom_lines.dot(np.linalg.inv(R).T)
    offset = normal.dot(t)
    world_planes[:,:3] = normal
    world_planes[:,3:] = offset
    return world_planes

def plane_intersection(planes1, planes2):
    directions = np.cross(planes1[:,:3], planes2[:,:3])
    l = len(planes1)

    # Find the point closest to the origin using Lagrange Multipliers, Lx = b
    L = np.zeros((l,5,5))
    L[:,:3,:3] = np.eye(3) * 2
    L[:,:3,3] = -planes1[:,:3]
    L[:,:3,4] = -planes2[:,:3]
    L[:,3,:3] = planes1[:,:3]
    L[:,4,:3] = planes2[:,:3]

    b = np.zeros((l,5))
    b[:,3] = planes1[:,3]
    b[:,4] = planes2[:,3]

    L_inv = np.linalg.inv(L)
    x = np.zeros((l, 5))
    for i in range(l):
        x[i] = L_inv[i].dot(b[i])
    closest_points = x[:,:3]

    lines = np.zeros((l,6))
    lines[:,:3] = directions
    lines[:,3:] = closest_points
    return lines

def snap_points_to_lines(hom_points, lines, transform, cam):
    K_inv = np.linalg.inv(cam.intrinsic)
    l = len(lines)
    direction = lines[:,:3]
    offset = lines[:,3:]
    R = transform[:3,:3]
    t = transform[:3,3:]

    A = direction.dot(R)
    B = offset.dot(R) + t[:,0]
    C = hom_points.dot(K_inv)

    # lambda * A = mu * B + C, we want mu
    # Mx=C, where M=[A|B], x = [lambda, -mu]
    M = np.zeros((l, 3, 2))
    M[:,:,0] = A
    M[:,:,1] = B
    x = np.zeros((l, 2))
    for i in range(l):
        x[i] = np.linalg.lstsq(M[i], C[i,0:])[0]
    mu = -x[:,1:]
    return direction * mu + offset

def triangulate_lines(lines1, lines2, transform1, transform2, cam):
    hom_start_1 = to_hom_points(lines1[:,:2], cam)
    hom_end_1 = to_hom_points(lines1[:,2:], cam)
    hom_start_2 = to_hom_points(lines2[:,:2], cam)
    hom_end_2 = to_hom_points(lines2[:,2:], cam)

    hom_lines_1 = compute_hom_lines(hom_start_1, hom_end_1)
    hom_lines_2 = compute_hom_lines(hom_start_2, hom_end_2)

    world_planes_1 = to_world_planes(hom_lines_1, transform1)
    world_planes_2 = to_world_planes(hom_lines_2, transform2)

    world_lines = plane_intersection(world_planes_1, world_planes_2)

    snapped_start_1 = snap_points_to_lines(hom_start_1, world_lines, transform1, cam)
    snapped_end_1 = snap_points_to_lines(hom_end_1, world_lines, transform1, cam)
    snapped_start_2 = snap_points_to_lines(hom_start_2, world_lines, transform2, cam)
    snapped_end_2 = snap_points_to_lines(hom_end_2, world_lines, transform2, cam)

    start_3d = (snapped_start_1 + snapped_start_2) / 2
    end_3d = (snapped_end_1 + snapped_end_2) / 2

    lines_3d = np.zeros((len(lines1), 6))
    lines_3d[:,:3] = start_3d
    lines_3d[:,3:] = end_3d

    return lines_3d