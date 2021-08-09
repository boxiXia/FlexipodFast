import socket
import msgpack
import numpy as np
import open3d as o3d
from flexipod_env import UDPServer


class MarkerMapTracker:
    def __init__(self,
                 local_address = ("127.0.0.1",32000),
                 camera_transform:np.ndarray = np.eye(4)
                 ):
        self.server = UDPServer(local_address=local_address)
        t_cam = np.asarray(camera_transform).reshape(4,4)
        self.t_cam_inv = np.linalg.inv(t_cam) # inverse camera transform
        self.reset()

    def reset(self):
        self.receive(max_attempts=1000000)
        
    def receive(self,max_attempts:int = 10,verbose=False):
        try:
            data = self.server.receive(max_attempts=max_attempts)
            data_unpacked = msgpack.unpackb(data,use_list=False)
            self.transform = self.t_cam_inv@(np.asarray(data_unpacked,dtype=np.float32).reshape((4,4)))
        except TimeoutError:
            pass
        return self.transform


# while(1):
#     try:
#         data = tracker.receive(verbose=False)
#         data = np.asarray(data).reshape((4,4))
#         print(data)
#     except Exception as e:
#         pass
    
################################################################

if __name__ == '__main__':
    
    tracker = MarkerMapTracker()
    
    from scipy.spatial.transform import Rotation as R
    import open3d as o3d
    import numpy as np
    coord_0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.3, origin=np.array([0., 0., 0.]))
    coord_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.3, origin=np.array([0., 0., 0.]))
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(coord_0)
    vis.add_geometry(coord_1)

    t1 = np.eye(4)

    # t_world = np.array([
    #     [0,0,1,0],
    #     [-1,0,0,0],
    #     [0,-1,0,0],
    #     [0,0,0,1]
    # ],dtype=float)

    t_world = np.eye(4)


    t_world = np.array([[0.073567934, 0.099872582, 0.99227685, -0.38515991],
        [-0.09691719, 0.99097961, -0.092556529, 0.050713513],
        [-0.99256986, -0.089359485, 0.082583688, 0.7048679],
        [0, 0, 0, 1]], dtype='float32')
    t_world = np.linalg.inv(t_world)

    ended=False
    def signalEnd(vis):
        global ended
        ended =True
    vis.register_key_callback(256, signalEnd)# key escape

    while(not ended):
        try:
            t1_ = tracker.receive(verbose=False)
    #         print(t1_)
    #         t1_ = np.linalg.inv(t1_)
    #         coord_1.transform(t_world@t1_@np.linalg.inv(t1)@np.linalg.inv(t_world))
            coord_1.transform(t1_@np.linalg.inv(t1))
            t1 = t1_[:]
    #         coord_1.rotate(r, center=(0, 0, 0))
            vis.update_geometry(coord_1)
            vis.poll_events()
            vis.update_renderer()
        except KeyboardInterrupt:
            break
        except Exception as e:
            pass
    vis.destroy_window()