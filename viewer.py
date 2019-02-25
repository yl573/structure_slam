import numpy as np
import cv2

import OpenGL.GL as gl
import pangolin

import dill

import time
from multiprocessing import Process, Queue

class DynamicArray(object):
    def __init__(self, shape=3):
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, tuple)

        self.data = np.zeros((1000, *shape))
        self.shape = shape
        self.ind = 0

    def clear(self):
        self.ind = 0

    def append(self, x):
        self.extend([x])
    
    def extend(self, xs):
        if len(xs) == 0:
            return
        assert np.array(xs[0]).shape == self.shape

        if self.ind + len(xs) >= len(self.data):
            self.data.resize(
                (2 * len(self.data), *self.shape) , refcheck=False)

        if isinstance(xs, np.ndarray):
            self.data[self.ind:self.ind+len(xs)] = xs
        else:
            for i, x in enumerate(xs):
                self.data[self.ind+i] = x
        self.ind += len(xs)

    def array(self):
        return self.data[:self.ind]

    def __len__(self):
        return self.ind

    def __getitem__(self, i):
        assert i < self.ind
        return self.data[i]

    def __iter__(self):
        for x in self.data[:self.ind]:
            yield x




class MapViewer(object):
    def __init__(self, system=None, config=None):
        self.system = system
        self.config = config

        self.saved_keyframes = set()

        # data queue
        self.q_pose = Queue()
        self.q_active = Queue()
        self.q_active_lines = Queue()
        self.q_points = Queue()
        self.q_lines = Queue()
        self.q_colors = Queue()
        self.q_graph = Queue()
        self.q_camera = Queue()
        self.q_image = Queue()

        self.q_ground = Queue()

        # message queue
        self.q_refresh = Queue()
        # self.q_quit = Queue()

        self.view_thread = Process(target=self.view)
        self.view_thread.start()

    def save_points(self):
        points = []
        colors = []
        
        print('Saving point cloud')
        for pt in self.system.map.mappoints():
            points.append(pt.position)
            colors.append(pt.best_seg_color())
        if len(points) > 0:
            with open('map.pkl', 'wb') as f:
                dill.dump({
                    'points': points,
                    'colors': colors
                }, f)

            self.load_points()

    def load_points(self):
        with open('map.pkl', 'rb') as f:
            data = dill.load(f)
        points = data['points']
        colors = data['colors']
        self.q_points.put((points, 0))
        self.q_colors.put((colors, 0))

    def update(self, refresh=False):
        while not self.q_refresh.empty():
            refresh = self.q_refresh.get()

        self.q_image.put(self.system.current.image)
        self.q_pose.put(self.system.current.pose.matrix())

        points = []
        lines = []
        line_colors = []
        for m in self.system.preceding.measurements():

            primitive = m.get_map_primitive()

            if m.is_point() and m.from_triangulation():
                points.append(m.mappoint.position) 

            if m.is_line() and m.from_triangulation():
                lines.append(m.mapline.endpoints)
                line_colors.append(m.mapline.color)

        self.q_active.put(points)
        if len(lines) > 0:
            self.q_active_lines.put((lines, line_colors))

        lines = []
        for kf in self.system.map.keyframes():
            if kf.get_preceding():
                lines.append(([*kf.position, *kf.get_preceding().position], 1))
            if kf.loop_keyframe is not None:
                lines.append(([*kf.position, *kf.loop_keyframe.position], 2))
        self.q_graph.put(lines)

        
        if refresh:
            print('****************************************************************', 'refresh')
            cameras = []
            for kf in self.system.map.keyframes():
                cameras.append(kf.pose.matrix())
            self.q_camera.put(cameras)

            # mesh = self.system.map.compute_ground_mesh()
            # self.q_ground.put(mesh)

            points = []
            colors = []
            # for pt in self.system.map.mappoints():
            for pt in self.system.map.mappoints():
                points.append(pt.position)
                colors.append(pt.color)
            if len(points) > 0:
                self.q_points.put((points, 0))
                self.q_colors.put((colors, 0))
        else:
            cameras = []
            points = []
            lines = []
            colors = []
            for kf in self.system.map.keyframes()[-20:]:
                if kf.id not in self.saved_keyframes:
                    cameras.append(kf.pose.matrix())
                    self.saved_keyframes.add(kf.id)
                    for m in kf.measurements():
                        if m.from_triangulation():
                            if m.is_point():
                                points.append(m.mappoint.position)
                                colors.append(m.mappoint.color)
                            if m.is_line():
                                lines.append(m.mapline.endpoints)
            if len(cameras) > 0:
                self.q_camera.put(cameras)
            if len(points) > 0:
                self.q_points.put((points, 1))
                self.q_colors.put((colors, 1))
            if len(lines) > 0:
                self.q_lines.put((lines, 1))


    def stop(self):
        self.update(refresh=True)
        self.view_thread.join()

        qtype = type(Queue())
        for x in self.__dict__.values():
            if isinstance(x, qtype):
                while not x.empty():
                    _ = x.get()
        print('viewer stopped')


    def view(self):
        pangolin.CreateWindowAndBind('Viewer', 1024, 768)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc (gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        panel = pangolin.CreatePanel('menu')
        panel.SetBounds(0.5, 1.0, 0.0, 175 / 1024.)

        # checkbox
        m_follow_camera = pangolin.VarBool('menu.Follow Camera', value=True, toggle=True)
        m_show_points = pangolin.VarBool('menu.Show Points', True, True)
        m_show_lines = pangolin.VarBool('menu.Show Lines', True, True)
        m_show_keyframes = pangolin.VarBool('menu.Show KeyFrames', True, True)
        m_show_graph = pangolin.VarBool('menu.Show Graph', True, True)
        m_show_image = pangolin.VarBool('menu.Show Image', True, True)

        # button
        m_replay = pangolin.VarBool('menu.Replay', value=False, toggle=False)
        m_refresh = pangolin.VarBool('menu.Refresh', False, False)  
        m_reset = pangolin.VarBool('menu.Reset', False, False)

        if self.config is None:
            width, height = 400, 250
            viewpoint_x = 0
            viewpoint_y = -500   # -10
            viewpoint_z = -100   # -0.1
            viewpoint_f = 2000
            camera_width = 1.
        else:
            width = self.config.view_image_width
            height = self.config.view_image_height
            viewpoint_x = self.config.view_viewpoint_x
            viewpoint_y = self.config.view_viewpoint_y
            viewpoint_z = self.config.view_viewpoint_z
            viewpoint_f = self.config.view_viewpoint_f
            camera_width = self.config.view_camera_width

        proj = pangolin.ProjectionMatrix(
            1024, 768, viewpoint_f, viewpoint_f, 512, 389, 0.1, 5000)
        look_view = pangolin.ModelViewLookAt(
            viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)

        # Camera Render Object (for view / scene browsing)
        scam = pangolin.OpenGlRenderState(proj, look_view)

        # Add named OpenGL viewport to window and provide 3D Handler
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 175 / 1024., 1.0, -1024 / 768.)
        dcam.SetHandler(pangolin.Handler3D(scam))


        # image
        # width, height = 400, 130
        dimg = pangolin.Display('image')
        dimg.SetBounds(0, height / 768., 0.0, width / 1024., 1024 / 768.)
        dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        texture = pangolin.GlTexture(width, height, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image = np.ones((height, width, 3), 'uint8')



        pose = pangolin.OpenGlMatrix()   # identity matrix
        following = True

        active = []
        replays = []
        graph = []
        loops = []
        mappoints = DynamicArray(shape=(3,))
        maplines = DynamicArray(shape=(6,))
        colors = DynamicArray(shape=(3,))
        cameras = DynamicArray(shape=(4, 4))

        active_lines = [] 
        line_colors = []   
        gnd_mesh = None

        while not pangolin.ShouldQuit():

            if not self.q_pose.empty():
                pose.m = self.q_pose.get()

            follow = m_follow_camera.Get()
            if follow and following:
                scam.Follow(pose, True)
            elif follow and not following:
                scam.SetModelViewMatrix(look_view)
                scam.Follow(pose, True)
                following = True
            elif not follow and following:
                following = False


            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)


            # show graph
            if not self.q_graph.empty():
                graph = self.q_graph.get()
                loops = np.array([_[0] for _ in graph if _[1] == 2])
                graph = np.array([_[0] for _ in graph if _[1] < 2])
            if m_show_graph.Get():
                if len(graph) > 0:
                    gl.glLineWidth(1)
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawLines(graph, 3)
                if len(loops) > 0:
                    gl.glLineWidth(2)
                    gl.glColor3f(0.0, 0.0, 0.0)
                    pangolin.DrawLines(loops, 4)

                gl.glPointSize(4)
                gl.glColor3f(1.0, 0.0, 0.0)
                gl.glBegin(gl.GL_POINTS)
                gl.glVertex3d(pose[0, 3], pose[1, 3], pose[2, 3])
                gl.glEnd()

            # if not self.q_ground.empty():
            #     gnd_mesh = self.q_ground.get()

            if gnd_mesh is not None:
                gl.glLineWidth(2)
                gl.glColor3f(0.5, 0.25, 0.5)
                pangolin.DrawLines(gnd_mesh, 2)


            # Show mappoints
            if not self.q_points.empty():
                pts, code = self.q_points.get()
                cls, code = self.q_colors.get()
                if code == 1:     # append new points
                    mappoints.extend(pts)
                    colors.extend(cls)
                elif code == 0:   # refresh all points
                    mappoints.clear()
                    mappoints.extend(pts)
                    colors.clear()
                    colors.extend(cls)

            if m_show_points.Get():
                gl.glPointSize(4)
                 # easily draw millions of points
                pangolin.DrawPoints(mappoints.array(), colors.array())

                if not self.q_active.empty():
                    active = self.q_active.get()

                gl.glPointSize(5)
                gl.glBegin(gl.GL_POINTS)
                gl.glColor3f(1.0, 0.0, 0.0)
                for point in active:
                    gl.glVertex3f(*point)
                gl.glEnd()

            if not self.q_lines.empty():
                lines, code = self.q_lines.get()
                maplines.extend(lines)       

            if m_show_lines.Get():
                
                gl.glLineWidth(1)
                gl.glColor3f(1.0, 0.0, 0.5)
                pangolin.DrawLines(maplines.array(), 2)

                if not self.q_active_lines.empty():
                    active_lines, line_colors = np.array(self.q_active_lines.get())

                if active_lines is not None:
                    for act_line, color in zip(active_lines, line_colors):
                        c = (color[2] / 255, color[1] / 255, color[0] / 255)
                        act_line = act_line.reshape((1,6))
                        gl.glLineWidth(5)
                        gl.glColor3f(*c)
                        pangolin.DrawLines(act_line, 2)


            if len(replays) > 0:
                n = 300
                gl.glPointSize(4)
                gl.glColor3f(1.0, 0.0, 0.0)
                gl.glBegin(gl.GL_POINTS)
                for point in replays[:n]:
                    gl.glVertex3f(*point)
                gl.glEnd()
                replays = replays[n:]


            # show cameras
            if not self.q_camera.empty():
                cams = self.q_camera.get()
                if len(cams) > 20:
                    cameras.clear()
                cameras.extend(cams)
                
            if m_show_keyframes.Get():
                gl.glLineWidth(1)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawCameras(cameras.array(), camera_width)

            
            # show image
            if not self.q_image.empty():
                image = self.q_image.get()
                if image.ndim == 3:
                    image = image[::-1, :, ::-1]
                else:
                    image = np.repeat(image[::-1, :, np.newaxis], 3, axis=2)
                image = cv2.resize(image, (width, height))
            if m_show_image.Get():         
                texture.Upload(image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                dimg.Activate()
                gl.glColor3f(1.0, 1.0, 1.0)
                texture.RenderToViewport()


            if pangolin.Pushed(m_replay):
                replays = mappoints.array()

            if pangolin.Pushed(m_reset):
                m_show_graph.SetVal(True)
                m_show_keyframes.SetVal(True)
                m_show_points.SetVal(True)
                m_show_image.SetVal(True)
                m_follow_camera.SetVal(True)
                follow_camera = True

            if pangolin.Pushed(m_refresh):
                self.q_refresh.put(True)

            pangolin.FinishFrame()