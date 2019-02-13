import numpy as np
import g2o
from measurements import MeasurementType


class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, ):
        super().__init__()

        # Higher confident (better than CHOLMOD, according to 
        # paper "3-D Mapping With an RGB-D Camera")
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        # Convergence Criterion
        terminate = g2o.SparseOptimizerTerminateAction()
        terminate.set_gain_threshold(1e-6)
        super().add_post_iteration_action(terminate)

        # Robust cost Function (Huber function) delta
        self.delta = np.sqrt(5.991)   
        self.aborted = False

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)
        try:
            return not self.aborted
        finally:
            self.aborted = False

    def add_pose(self, pose_id, pose, cam, fixed=False):
        sbacam = g2o.SBACam(
            pose.orientation(), pose.position())
        sbacam.set_cam(
            cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3) 

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_marginalized(marginalized)
        v_p.set_estimate(point)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    # def add_edge_from_raw(self, id, point_id, pose_id, meas_type: MeasurementType, meas_data):
    #     if meas_type == MeasurementType.STEREO:
    #         edge = self.stereo_edge(meas_data)
    #     elif meas_type == MeasurementType.LEFT:
    #         edge = self.mono_edge(meas_data)
    #     elif meas_type == MeasurementType.RIGHT:
    #         edge = self.mono_edge_right(meas_data)
    #     self._add_edge(id, point_id, pose_id, edge)        

    def add_edge(self, id, point_id, pose_id, meas):
        if meas.is_stereo():
            edge = self.stereo_edge(meas.xyx)
        elif meas.is_left():
            edge = self.mono_edge(meas.xy)
        elif meas.is_right():
            edge = self.mono_edge_right(meas.xy)
        self._add_edge(id, point_id, pose_id, edge)

    def _add_edge(self, id, point_id, pose_id, edge):
        edge.set_id(id)
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        kernel = g2o.RobustKernelHuber(self.delta)
        edge.set_robust_kernel(kernel)
        super().add_edge(edge)

    def stereo_edge(self, projection, information=np.identity(3)):
        e = g2o.EdgeProjectP2SC()
        e.set_measurement(projection)
        e.set_information(information)
        return e

    def mono_edge(self, projection, 
            information=np.identity(2) * 0.5):
        e = g2o.EdgeProjectP2MC()
        e.set_measurement(projection)
        e.set_information(information)
        return e

    def mono_edge_right(self, projection, 
            information=np.identity(2) * 0.5):
        e = g2o.EdgeProjectP2MCRight()
        e.set_measurement(projection)
        e.set_information(information)
        return e

    def get_pose(self, id):
        return self.vertex(id * 2).estimate()

    def get_point(self, id):
        return self.vertex(id * 2 + 1).estimate()

    def abort(self):
        self.aborted = True

class LocalBA(object):
    def __init__(self, ):
        self.optimizer = BundleAdjustment()
        self.measurements = []
        self.keyframes = []
        self.mappoints = set()

        # threshold for confidence interval of 95%
        self.huber_threshold = 5.991

    def add_keyframe(self, id, pose, cam, fixed=False):
        self.optimizer.add_pose(id, pose, cam, fixed=fixed)

    def add_mappoint(self, point_id, position):
        self.optimizer.add_point(point_id, position)

    def add_measurement(self, meas_id, point_id, keyframe_id, meas_type, measured_point):
        self.optimizer.add_edge(meas_id, point_id, keyframe_id, m)

    # def set_data(self, adjust_keyframes, fixed_keyframes, mappoints, observations):

    def set_data(self, adjust_keyframes, fixed_keyframes):
        self.clear()
        for kf in adjust_keyframes:
            self.optimizer.add_pose(kf.id, kf.pose, kf.cam, fixed=False)
            self.keyframes.append(kf)

            for m in kf.point_measurements():

                pt = m.mappoint
                if pt not in self.mappoints:
                    self.optimizer.add_point(pt.id, pt.position)
                    self.mappoints.add(pt)

                edge_id = len(self.measurements)
                self.optimizer.add_edge(edge_id, pt.id, kf.id, m)
                self.measurements.append(m)

        for kf in fixed_keyframes:
            self.optimizer.add_pose(kf.id, kf.pose, kf.cam, fixed=True)
            for m in kf.point_measurements():
                if m.mappoint in self.mappoints:
                    edge_id = len(self.measurements)
                    self.optimizer.add_edge(edge_id, m.mappoint.id, kf.id, m)
                    self.measurements.append(m)

    def update_points(self):
        for mappoint in self.mappoints:
            mappoint.update_position(self.optimizer.get_point(mappoint.id))

    def update_poses(self):
        for keyframe in self.keyframes:
            keyframe.update_pose(self.optimizer.get_pose(keyframe.id))

    def get_bad_measurements(self):
        bad_measurements = []
        for edge in self.optimizer.active_edges():
            if edge.chi2() > self.huber_threshold:
                bad_measurements.append(self.measurements[edge.id()])
        return bad_measurements

    def clear(self):
        self.optimizer.clear()
        self.keyframes.clear()
        self.mappoints.clear()
        self.measurements.clear()

    def abort(self):
        self.optimizer.abort()

    def optimize(self, max_iterations):
        return self.optimizer.optimize(max_iterations)


