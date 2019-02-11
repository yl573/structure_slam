import numpy as np

def compute_hom_lines(start_hom_points, end_hom_points):
    hom_lines = np.cross(start_hom_points, end_hom_points)
    return hom_lines

def to_hom_points(points, cam):
    hom_points = np.ones((len(points), 3))
    hom_points[:,0] = (points[:,0] - cam.cx) / cam.fx
    hom_points[:,1] = (points[:,1] - cam.cy) / cam.fy
    hom_points = hom_points / hom_points[:,2:]
    return hom_points

def to_world_planes(hom_lines, transform):
    R = transform[:3,:3]
    t = transform[:3,3:]
    world_planes = np.zeros((len(hom_lines), 4))
    normal = hom_lines.dot(np.linalg.inv(R).T)
    normal = normal / np.linalg.norm(normal, axis=1)[:,None]
    offset = normal.dot(t)
    world_planes[:,:3] = normal
    world_planes[:,3:] = offset

    return world_planes

def plane_intersection(planes1, planes2):
    directions = np.cross(planes1[:,:3], planes2[:,:3])
    directions = directions / np.linalg.norm(directions, axis=1)[:,None]
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

    x = np.zeros((l, 5))
    for i in range(l):
        x[i] = np.linalg.inv(L[i]).dot(b[i])
    closest_points = x[:,:3]

    lines = np.zeros((l,6))
    lines[:,:3] = directions
    lines[:,3:] = closest_points

    return lines

def snap_points_to_lines(hom_points, lines, transform):
    l = len(lines)
    direction = lines[:,:3]
    offset = lines[:,3:]
    R = transform[:3,:3]
    t = transform[:3,3:]
    
    # line is A * mu + B
    # sigma * hom = mu * RA + RB + t
    # M = [RA|hom], x = RB + t, M * [mu, sigma]^T = x
    
    M = np.zeros((l, 3, 2))
    M[:,:,0] = -direction.dot(R.T)
    M[:,:,1] = hom_points
    B = offset.dot(R.T) - t.T

    x = np.zeros((l, 2))
    for i in range(l):
        x[i] = np.linalg.lstsq(M[i], B[i], rcond=1)[0]

    mu = x[:,0:1]
    return direction * mu + offset, mu

def invert_transform(transform):
    new = np.array(transform)
    R_inv = np.linalg.inv(transform[:3,:3])
    new[:3,:3] = R_inv
    new[:3,3:] = -R_inv.dot(transform[:3,3:])
    return new

def triangulate_lines(lines1, lines2, inv_transform1, inv_transform2, cam, max_end_diff=10):

    # for some reason stereo-ptam uses the inverse transformation matrix, need to reserse it
    transform1 = invert_transform(inv_transform1)
    transform2 = invert_transform(inv_transform2)

    hom_start_1 = to_hom_points(lines1[:,:2], cam)
    hom_end_1 = to_hom_points(lines1[:,2:], cam)
    hom_start_2 = to_hom_points(lines2[:,:2], cam)
    hom_end_2 = to_hom_points(lines2[:,2:], cam)

    hom_lines_1 = compute_hom_lines(hom_start_1, hom_end_1)
    hom_lines_2 = compute_hom_lines(hom_start_2, hom_end_2)

    world_planes_1 = to_world_planes(hom_lines_1, transform1)
    world_planes_2 = to_world_planes(hom_lines_2, transform2)

    world_lines = plane_intersection(world_planes_1, world_planes_2)

    snapped_start_1, mu_start_1 = snap_points_to_lines(hom_start_1, world_lines, transform1)
    snapped_end_1, mu_end_1 = snap_points_to_lines(hom_end_1, world_lines, transform1)
    snapped_start_2, mu_start_2 = snap_points_to_lines(hom_start_2, world_lines, transform2)
    snapped_end_2, mu_end_2 = snap_points_to_lines(hom_end_2, world_lines, transform2)

    is_good = np.logical_and(
        abs(mu_start_1 - mu_start_2) < max_end_diff,
        abs(mu_end_1 - mu_end_2) < max_end_diff,
    ).reshape(-1)

    start_3d = (snapped_start_1 + snapped_start_2) / 2
    end_3d = (snapped_end_1 + snapped_end_2) / 2

    lines_3d = np.zeros((len(lines1), 6))
    lines_3d[:,:3] = start_3d
    lines_3d[:,3:] = end_3d

    return lines_3d, is_good