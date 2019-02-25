from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from sklearn.neighbors import BallTree
import numpy as np

class Map(object):
    def __init__(self, ):
        self.kfs = []
        self.pts = set()
        self.lns = set()
        
        self.kfs_set = set()
        self.meas_lookup = dict()

    def keyframes(self):
        return self.kfs.copy()

    def mappoints(self):
        return self.pts.copy()

    def add_keyframe(self, kf):
        self.kfs.append(kf)
        self.kfs_set.add(kf)

    def add_mappoint(self, pt):
        self.pts.add(pt)

    def add_mapline(self, ln):
        self.lns.add(ln)

    def search_adjust_keyframes(self):
        return self.kfs[-3:]

    def add_point_measurement(self, kf, pt, meas):
        if kf not in self.kfs_set or pt not in self.pts:
            return

        meas.keyframe = kf
        meas.mappoint = pt

        self.meas_lookup[meas.id] = meas

    def add_line_measurement(self, kf, ln, meas):
        if kf not in self.kfs_set or ln not in self.lns:
            return
        meas.keyframe = kf
        meas.mapline = ln

        self.meas_lookup[meas.id] = meas

    def ground_points(self):
        return [pt for pt in self.pts if pt.best_seg() == 0]

    def triangulate_mesh_simplices(self, points_3d, invalid_indices):
        """
        Use valid_indices to create a non-convex mesh
        """
        pca = PCA(n_components=2)
        pca.fit(points_3d)
        points = pca.transform(points_3d)
        
        invalid_indices = set(invalid_indices)
        triangles = Delaunay(points)
        boundary_points = set()
        valid_triangles = []
        
        # find inlier points on the boundary
        for simp in triangles.simplices:
            invalid_pts = [s for s in simp if s in invalid_indices]
            
            if len(invalid_pts) <= 1:
                valid_triangles.append(simp)

            if len(invalid_pts) == 1:
                in_points = set([s for s in set(simp) if s not in invalid_indices])
                boundary_points = boundary_points | in_points
                
        concave_triangles = []        
        
        for simp in valid_triangles:
            boundary_pts = [s for s in simp if s in boundary_points]
            if len(boundary_pts) < 3:
                concave_triangles.append(simp)
        
        return np.array(concave_triangles)



    def compute_ground_mesh(self):
        gnd_pts = self.ground_points()
        pos_3d = np.array([pt.position for pt in gnd_pts])

        import time

        t0 = time.time()

        N_NEIGHBOURS = 10
        TOP_CUTOFF = 70
        BOT_CUTOFF = 40

        bt = BallTree(pos_3d, leaf_size=30, metric='euclidean')
        dist, nbrs = bt.query(pos_3d, k=N_NEIGHBOURS + 1, return_distance=True)
        avg_nbr_dist = np.mean(dist[:,1:], axis=1)

        top_lim = np.percentile(avg_nbr_dist, TOP_CUTOFF)
        bot_lim = np.percentile(avg_nbr_dist, BOT_CUTOFF)
        invalid = np.bitwise_or(avg_nbr_dist < bot_lim, avg_nbr_dist > top_lim)
        invalid_indices = nbrs[invalid][:,0]

        simplices = self.triangulate_mesh_simplices(pos_3d, invalid_indices)

        # lines for drawing
        lines = []
        for tri in pos_3d[simplices]:
            lines.append(np.hstack((tri[0], tri[1])))
            lines.append(np.hstack((tri[1], tri[2])))
            lines.append(np.hstack((tri[2], tri[0])))
        lines = np.array(lines)

        print(f'Time: {time.time() - t0}')

        return lines



