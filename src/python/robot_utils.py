import open3d as o3d
import trimesh
import point_cloud_utils as pcu  # downsampling
import numpy as np
import matplotlib.pyplot as plt
import numba
import copy
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.spatial.transform import Rotation
import tempfile
import gmsh
import meshio
import itertools
import shutil  # file copying
import networkx as nx # graph representation
from matplotlib.colors import to_hex
from lxml import etree
import os
def normalizeSignedDistance(signed_distance, zero_map_to=0.5):
    """
    Normalize to 0-1
    min-0 map to 0-0.5
    0-max map to 0.5-1
    """
    is_negative = signed_distance <= 0

    normalized_distance = np.zeros_like(signed_distance)
    minimum = signed_distance.min()
    maximum = signed_distance.max()
    normalized_distance[is_negative] = (
        signed_distance[is_negative] - minimum) / np.abs(minimum) * zero_map_to
    normalized_distance[~is_negative] = zero_map_to + \
        signed_distance[~is_negative]/np.abs(maximum)*(1-zero_map_to)
    return normalized_distance


def normalizeMinMax(v):
    """
    normalize a vector v to [0,1]
    """
    v_min = v.min()
    v_max = v.max()
    v_n = (v - v_min)/(v_max-v_min)  # normalized
    return v_n


# # # https://matplotlib.org/tutorials/colors/colormaps.html
# # cmap = plt.cm.get_cmap('hot')
# sd = np.linspace(-2,2,51)
# mapped = normalizeSignedDistance(sd, zero_map_to=0.0)
# plt.plot(sd,mapped)


coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=60, origin=[0, 0, 0])


def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(1.0, 0.0)
    return False

# @numba.jit(["float64[:,:](float64[:,::1], float64,int64)",
#             "float64[:,:](float64[:,:], float64,int64)"],nopython=True)


@numba.jit([
    (numba.types.Array(numba.types.float64, 2, 'C', readonly=True),
     numba.types.float64, numba.types.int64),
    (numba.types.Array(numba.types.float64, 2, 'C', readonly=False), numba.types.float64, numba.types.int64)],
    nopython=True, nogil=True)
def uniformRandomAroundPoints(points, radius, num_per_grid=50):
    """
    sample random points uniformly (along xyz) around an arry of n points (n,3)
    input:
        points: (nx3) numpy array of xyz
        radius: (float scalar) the radius along x,y,z direction to randomly sample the points
        num_per_grid:(float scalar) number of random points to sample per specified point
    """
#     num = points.shape[0]*num_per_grid
#     xyz = np.random.uniform(-radius,radius+np.nextafter(0,1),num*3).reshape((num,3))
#     xyz = xyz + points.repeat(num_per_grid,axis=0)
    num_grid = points.shape[0]
    xyz = np.empty((num_grid*num_per_grid, 3), dtype=np.float64)
#     for i,point in enumerate(points):
    for i in range(num_grid):
        point = points[i]
        start = num_per_grid*i
        end = start+num_per_grid
        xyz[start:end, 0] = np.random.uniform(
            point[0]-radius, point[0]+radius, num_per_grid)
        xyz[start:end, 1] = np.random.uniform(
            point[1]-radius, point[1]+radius, num_per_grid)
        xyz[start:end, 2] = np.random.uniform(
            point[2]-radius, point[2]+radius, num_per_grid)
    return xyz


# # example, compile
# _ = uniformRandomAroundPoints(np.zeros((2, 3)), 1.0, num_per_grid=5)
# _ = uniformRandomAroundPoints(np.ascontiguousarray(
#     np.zeros((2, 3))), 1.0, num_per_grid=5)
########################################################################################
########## geometry #####################################################################


def axisAngleRotation(vec, angle):
    """
    return a rotation matrix of axis angle rotation represented by
    a homogeneous transformation matrix (4x4) 
    input:
        vec: (x,y,z) of the rotation axis
        angle: [rad] angle of rotation about the rotation axis
    return:
        (4x4) rotation matrix
    """
    v = np.array(vec, dtype=np.float64)
    v_norm = np.linalg.norm(v)
    if v_norm != 1:
        v = v/v_norm
    h = np.eye(4)
    h[:3, :3] = Rotation.from_rotvec(v*angle).as_matrix()
    return h


def translation(vec, h = None):
    """
    apply translation to a (4x4) homogeneous transformation matrix h
    input:
        vec: (x,y,z) of the rotation axis
        h: a (4x4) homogeneous transformation matrix
    return:
        the translated homogeneous transformation matrix h
    """
    if h is None:
        h=np.eye(4,dtype=np.float64)
    h[:3, -1] += vec
    return h


def applyTransform(xyz, t):
    """
    apply transformation matrix t to 3d points xyz,
    input:
        xyz: nx3 np array of 3d points
        t: if transform.shape=(3,3): rotation
           if transform.shape=(4,4): homogegenious transformation (rotation+translation)
    output:
        nx3 np array of the transformed xyz points
    """
    xyz = np.asarray(xyz)
    t = np.asarray(t)
    if t.shape == (3, 3):  # rotation matrix
        return np.dot(xyz, t.T)
    elif t.shape == (4, 4):  # homogeneous matrix
        return np.dot(xyz, t[:-1, :-1].T)+t[:-1, -1]
    else:
        raise AssertionError("dimension error")
    

################################################################

def ColorizePcdAndLsd(nsd,pcd,lsd,cmap):
    """
    colorize the pcd (o3d pointcloud) and its corresponding lsd (o3d lineset) 
    given nsd (normalized signed distance) and cmap (plt colormap)
    Input:
        nsd: a normalized signed distance of the pcd, normalized to [0,1]
        pcd: o3d point cloud, representing the mass points
        lsd: o3d lineset, representing the springs
        cmap: plt colormap, ref: https://matplotlib.org/tutorials/colors/colormaps.html
    """
    pcd_color = cmap(nsd)[:,:3] # drop alpha channel
    pcd.colors = o3d.utility.Vector3dVector(pcd_color)
    lines = np.asarray(lsd.lines)
    # the line colors are the average of the two end points colors
    lsd_color =  (pcd_color[lines[:,0]]+pcd_color[lines[:,1]])/2
    lsd.colors = o3d.utility.Vector3dVector(lsd_color)
    return pcd,lsd


def createGrid(bounds=[[-1, -1, -1], [1, 1, 1]], dr=0.1):
    """
    retrun a grid of points shaped (nx,ny,nz,3) given bounds and discritization radius
    where nx,ny,nz are the number of points in x,y,z direction
    the bounds are updated and also returned
    input:
        bounds: [(x_low,y_low,z_low),(x_high,y_high,z_high)]
        dr: discretization radius of the grid
    output:
        xyz_grid: a grid of points numpy array of (nx,ny,nz,3)
        bounds: updated bounds
    """
    # round to integer, type is still float
    bounds = bounds/dr
    bounds = np.stack((np.floor(bounds[0]), np.ceil(bounds[1])))*dr
#     print("bounds=\n", bounds)
    # number of points in x,y,z direction:(nx,ny,nz)
    nx, ny, nz = np.ceil((bounds[1]-bounds[0])/dr).astype(int)
    x = np.linspace(bounds[0, 0], bounds[0, 0]+(nx-1)*dr, num=nx)
    y = np.linspace(bounds[0, 1], bounds[0, 1]+(ny-1)*dr, num=ny)
    z = np.linspace(bounds[0, 2], bounds[0, 2]+(nz-1)*dr, num=nz)
    # a flattened grid of xyzs of the vertices
    xyz_grid = np.stack(np.meshgrid(x, y, z), axis=-1)
    return xyz_grid, bounds


def getUniqueEdges(edges: np.ndarray):
    """
    return unique edges given a possibly non-unique edges
    input:
        edges:  np.ndarray, left and right vertex indices (nx2) 
    output:
        unique_edges: np.ndarray, unique edges
        edge_counts: np.ndarray, count of unique edges
    """
    edges = np.sort(edges, axis=1)  # sorted
    unique_edges, edge_counts = np.unique(
        edges, axis=0, return_index=False, return_counts=True)
    return unique_edges, edge_counts


def getNeighborCounts(unique_edges: np.ndarray):
    """
    return number of neighboring points connected by unique_edges
    input:
        unique_edges:  np.ndarray, unique edges (non-repeating)
    output:
        neighbor_counts:np.ndarray, number of neighboring points sorted by index
    """
    neighbor_counts = np.zeros(unique_edges.max()+1, dtype=int)
    for k in range(2):
        v_id, v_count = np.unique(unique_edges[:, k], return_counts=True)
        neighbor_counts[v_id] += v_count
    return neighbor_counts


def getEdges(neighbor: np.ndarray, self_id: np.ndarray = None, return_edge_counts: np.ndarray = False):
    """
    return the edges given neighbor
    input:
        neighbor: nxm int np.array, assume n points, 
        self_id: if self_id is none, assume: each point has m-1 neighbors,
            neighbor[k,0] is the index of point k iteself, the neighbor points are assorted by distance
            if self_id is specified, then each point has m neighbors
    returns:
        edges: nx2 int np.array, of the edges
    """
    if self_id is None:
        candidate = neighbor[1:]
        self_id = neighbor[0]
    else:
        candidate = neighbor
    edges = np.empty((candidate.size, 2), dtype=np.int32)
    edges[:, 0] = self_id
    edges[:, 1] = candidate

    unique_edges, edge_counts = getUniqueEdges(edges)
    if return_edge_counts:
        return unique_edges, edge_counts
    else:
        return unique_edges


def getMidpoints(points: np.ndarray, epid: np.ndarray):
    """
    return the xyzs of the midpoints given
    points: (nx3 double np.array)
    epid: endpoints indices (nx2 int np.array)
    """
    return 0.5*(points[epid[:, 0]]+points[epid[:, 1]])


# def momentOfInertial(p: np.ndarray, p0, n):
#     """
#     calculate the moment of inertia of points p rotated about axis with normal n,
#     p0 is a point on the axis
#     """
#     if np.abs(np.linalg.norm(n)-1) > 1e-16:
#         n = n/np.linalg.norm(n)  # normallize the axis direction
#     p0 = np.asarray(p0)
#     d = np.cross(p-p0, n)
#     return np.sum(np.linalg.norm(d, ord=2, axis=1)**2)

def momentOfInertia(v:np.ndarray):
    """
    return moment of inertia of a given vertices
    """
    x,y,z = v[:,0],v[:,1],v[:,2]
    xx = x.dot(x)
    xy = x.dot(y)
    xz = x.dot(z)
    yy = y.dot(y)
    yz = y.dot(z)
    zz = z.dot(z)
    moi = np.asarray(((yy+zz,-xy,-xz),(-xy,xx+zz,-yz),(-xz,-yz,xx+yy)))
    return moi # moment of inertia


def trimeshToO3dMesh(mesh):
    """
    convert trimesh mesh object to open3d mesh object
    """
    assert(type(mesh) == trimesh.base.Trimesh)
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces))
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.compute_triangle_normals()
#     mesh_o3d.paint_uniform_color((0.8, 0.8, 0.8))
    return mesh_o3d


def o3dShow(geometry, **kwargs):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    if "background_color" in kwargs:  # (r,g,b)
        opt.background_color = np.asarray(
            kwargs["background_color"], dtype=float)
    if "mesh_show_wireframe" in kwargs:  # bool
        opt.mesh_show_wireframe = kwargs["mesh_show_wireframe"]
    if "point_show_normal" in kwargs:  # bool
        opt.point_show_normal = kwargs["point_show_normal"]
    if "show_coordinate_frame" in kwargs:  # bool
        opt.show_coordinate_frame = kwargs["show_coordinate_frame"]
    try:
        for g in geometry:
            vis.add_geometry(g)
    except TypeError:  # geometry is not iteratble
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()
    
###########################################################################
def vmeshSummary(vmesh):
    """
    summerize the volume mesh generated by the gmsh and read from the meshio
    """
    dim = 3 if "tetra" in vmesh.cells_dict.keys() else 2
    if dim==3:
        tetra = vmesh.cells_dict["tetra"] # (nx4) np array of tetrahedron
        # edges from tetrahedron
        edges_tetra = tetra[:, list(itertools.combinations(range(4), 2))].reshape((-1, 2))
        edges_tetra, edges_tetra_counts = getUniqueEdges(edges_tetra)

    vertices = vmesh.points
    faces = vmesh.get_cells_type("triangle")

    edges_face = faces[:, list(itertools.combinations(range(3), 2))].reshape((-1, 2))
    edges_face, edges_face_counts = getUniqueEdges(edges_face)

    edges = edges_face if dim==2 else edges_tetra
    edges_counts = edges_face_counts if dim==2 else edges_tetra_counts

    neighbor_counts = getNeighborCounts(edges)
    edge_lengths = np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)

    print(f"# vertices          = {vertices.shape[0]}")
    print(f"# surface triangle  =",faces.shape[0])
    if dim==3:
        print(f"# tetra             =",tetra.shape[0])
        print(f"# unique tetra edges= {edges_tetra.shape[0]}")
    print(f"# unique face edges = {edges_face.shape[0]}")
    with np.printoptions(precision=3, suppress=True):
        com = np.mean(vmesh.points,axis=0)
        print("COM                 = ",com)
    print(f"COM norm            = {np.linalg.norm(com):.3f}")
    print(f"mean edge length    = {edge_lengths.mean():.2f}")

    fig,ax = plt.subplots(1,3,figsize=(12,2),dpi=75)
    ax[0].hist(edges_counts,bins='auto', density=True)
    ax[0].set_xlabel("edge counts")
    n,bins,_ = ax[1].hist(edge_lengths, density=True, bins='auto')
    ax[1].set_xlabel("edge length")
    ax[1].text(bins[0],0,f"{bins[0]:.1f}",ha="left",va="bottom",fontsize="large",color='r')
    ax[1].text(bins[-1],0,f"{bins[-1]:.1f}",ha="right",va="bottom",fontsize="large",color='r')
    ax[2].hist(neighbor_counts, density=True, bins='auto')
    ax[2].set_xlabel("neighbor counts")
    plt.show()
    
###################################################################################
def generateGmsh(
    in_file_name: str, # should be a step cad file
    out_file_name: str = None,# should end with .msh or .stl(2D only)
    gmsh_args: list = [],
    gmsh_args_3d:list = [],# 3d specific gmsh argument
    dim:int = 3, #mesh dimension {3:volume, 2: surface mesh}
    gui = False, # display gui at the end
):
    """
    return a volume mesh or suface mesh from CAD model
    Input:
    ------------------
    in_file_name : str
        Location of the file to be imported, file type should be 
        ['.brep', '.stp', '.step', '.igs', '.iges',
        '.bdf', '.msh', '.inp', '.diff', '.mesh']
    out_file_name : str
        Location of the file to be imported, file type should be
        '.msh' for 2D/3D mesh or '.stl' for 2D mesh only
    gmsh_args : (n, 2) list
      List of (parameter, value) pairs to be passed to
      gmsh.option.setNumber, called before generating 2d mesh
    gmsh_args_3d : (n, 2) list
      additional 3d specific list of (parameter, value) pairs to be 
      passed to gmsh.option.setNumber, called before generating 3d mesh
    dim : int mesh dimension: 3 for volume mesh, 2 for surface mesh
    
    output:
    -------------------
    a mesh (meshio.Mesh), and out_file_name
    
    note: 
    modified from: ttps://github.com/mikedh/trimesh/blob/master/trimesh/interfaces/gmsh.py 
    for gmsh_arg, refer to: https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options-list
    mesh algorithm: # https://gmsh.info/doc/texinfo/gmsh.html#Choosing-the-right-unstructured-algorithm
    """
    gmsh.initialize() # !!must be call for initialization!!
    gmsh.model.add('Mesh_Generation')
    

    for arg in gmsh_args:
        gmsh.option.setNumber(*arg)
        
    gmsh.open(in_file_name)
    
    gmsh.model.geo.synchronize()
    gmsh.model.occ.synchronize()
    
    gmsh.model.mesh.generate(2)  # generate 2d mesh
#     gmsh.model.mesh.optimize('Relocate2D', True,niter=10)
    
    if dim==3:
        for arg in gmsh_args_3d:
            gmsh.option.setNumber(*arg)
        gmsh.model.mesh.generate(3)  # generate 3d terahedra mesh
#     gmsh.model.mesh.optimize('Netgen', True,niter=2)
#     gmsh.model.mesh.optimize('', True,niter=10)
    
    if out_file_name is None: # create a temporary file for the results
        with tempfile.NamedTemporaryFile(suffix='.msh', delete=False) as out_data:
            out_file_name = out_data.name            
    gmsh.write(out_file_name)
    
    if gui:
        gmsh.fltk.run() # display gui
    
    gmsh.finalize()  # !!must be call for ending!!
    return meshio.read(out_file_name),out_file_name


####################################################################################
class VolumeMesh(dict):
    """
    contains vertices, lines and triangles of a volume mesh
    the class is inherited form dict, and can be extended with custom key:value pair
    """
    def __init__(self,vertices,lines,triangles,cmap = 'hot'):
        self["vertices"] = np.asarray(vertices,dtype=np.float64)
        self["lines"] = np.asarray(lines,dtype=np.int64)
        self["triangles"] = np.asarray(triangles,dtype=np.int64)
        self.reColor(cmap,update_signed_distance=True)# update colors
        
    def copy(self,cmap=None):
        other = copy.copy(self)
        if cmap is not None:
            other = other.reColor(cmap)
        return other
    
    @property
    def vertices(self):
        """return vertices (nx3 float np.array)"""
        return self["vertices"]
    @vertices.setter
    def vertices(self,value):
        """set vertices (nx3 float np.array)"""
        self["vertices"] = np.asarray(value,dtype=np.float64)

    @property
    def lines(self):
        """return lines/edges (nx2 int np.array)"""
        return self["lines"]
    @lines.setter
    def lines(self,value):
        """set lines/edges (nx2 int np.array)"""
        self["lines"] = np.asarray(value,dtype=np.int64)

    @property
    def triangles(self):
        """return triangle faces (nx3 int np.array)"""
        return self["triangles"]
    @triangles.setter
    def triangles(self,value):
        """return triangle faces (nx3 int np.array)"""
        self["triangles"] = np.asarray(value,dtype=np.int64)

    def pcd(self):
        """ return o3d pointcloud """
        pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self["vertices"]))
        pointcloud.colors = o3d.utility.Vector3dVector(self["vertices_color"])
        return pointcloud

    def lsd(self):
        """return o3d lineset """
        lineset = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(self["vertices"]), 
            o3d.utility.Vector2iVector(self["lines"]))
        lineset.colors = o3d.utility.Vector3dVector(self["lines_color"])
        return lineset
    
    def triMesh(self):
        """return trimesh mesh"""
        return trimesh.Trimesh(self["vertices"],self["triangles"],process = False, maintain_order=True)
    
    def o3dMesh(self):
        """return open3d triangle mesh"""
        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self["vertices"]),
            o3d.utility.Vector3iVector(self["triangles"]))
    
    def reColor(self,cmap,update_signed_distance=False):
        """update the vertex and lines colors"""
        if type(cmap)==str:
            cmap = plt.get_cmap(cmap)
        if update_signed_distance:
            self["nsd"] = self.signedDistance(normalized=True)
        self["vertices_color"] = cmap(self["nsd"])[:,:3]
        self["lines_color"] = (self["vertices_color"][self["lines"][:,0]]+
                               self["vertices_color"][self["lines"][:,1]])/2
        return self
        
    def signedDistance(self,test_points=None,normalized=True):
        """
        return signed distance from surface of test_points(default vertices),
        normalize if normalized==True
        """
        if test_points is None:
            test_points = self["vertices"] 
        sd = trimesh.proximity.signed_distance(self.triMesh(),test_points)
        if normalized:
            return normalizeMinMax(sd)
        return sd
    
########################################################################################


class Unit(dict):
    def __init__(self,unit_dict=dict()):
        """
        define the working units
        use toSI(name) method to get the multiplier that convert to SI unit
        ref:
            https://en.wikipedia.org/wiki/International_System_of_Units
            https://en.wikipedia.org/wiki/SI_derived_unit
        """
        self.si = { # si multiplier
        # length -> meters
        "mm": 1e-3,"cm": 1e-2,"m":1.0,
        # time -> seconds
        "hour": 3600.,"minute":60.,"s":1.0,
        # mass -> kilogram
        "kg":1.0,"g":1e-3,
        # angle -> radian
        "deg": np.pi/180.,"rad":1.0,
        # density -> kg/m^3
        "kg/m^3": 1,"g/mm^3":1e6,
        # force - > N
        "N":1,
        }
        for name,unit in unit_dict.items():
            if unit not in self.si:
                raise KeyError(f"invalid unit:{unit}, supported units are {self.si.keys()}")
        super().__init__(unit_dict)
        self.default_unit = dict(time="s",length="m",mass="kg",angle = "rad",density="kg/m^3")
        for key,value in self.default_unit.items():
            if key not in self:
                self[key] = value

        # derived unit
        self.du = {
            "force":("mass","length",("time",-2)),
            "torque":("mass",("length",2),("time",-2)),
#             "inertia":("mass",("length",2)),
            "area":(("length",2),),
            "volume":(("length",3),),
            "density":(("length",-3),"mass"),
        }
    def toSI(self,name):
        try: # can overwrite unit with key:value
            return self.si[self[name]]
        except KeyError:
            derived = 1.0
            for n in self.du[name]:
                if type(n) is tuple:
                    derived*=self.si[self[n[0]]]**n[1]
                else:
                    derived*=self.si[self[n]]
            return derived

# # example
# unit = Unit({"length":"mm"})
# print(unit.toSI("torque"))
# print(unit.toSI("inertia"))
# print(unit.toSI("area"))
# print(unit.toSI("volume"))
# print(unit.toSI("density"))
##########################################################################
# number of points in a coordinate (o,x,y,z,-x,-y,-z)
NUM_POINTS_PER_COORDINATE = 7

def getCoordinateOXYZ(transform):
    coordinate_radius = 16
    o = transform[:3, -1]  # origin
    ox = coordinate_radius * transform[:3, 0]
    oy = coordinate_radius * transform[:3, 1]
    oz = coordinate_radius * transform[:3, 2]
    x = o + ox
    y = o + oy
    z = o + oz
    nx = o - ox  # negative x
    ny = o - oy  # negative y
    nz = o - oz  # netative z
    oxyz = np.vstack([o, x, y, z, nx, ny, nz])
    return oxyz

def getCoordinateOXYZSelfSprings():
    return np.asarray(list(itertools.combinations(range(NUM_POINTS_PER_COORDINATE), 2)), dtype=int)
# print(getCoordinateOXYZSelfSprings())

class RobotDescription(nx.classes.digraph.DiGraph):
    def __init__(self, incoming_graph_data=None,unit_dict=dict(), **attr):
        """
        a directed graph of the robot description
        """
        super().__init__(incoming_graph_data, **attr)
        self.unit = Unit(unit_dict)

    def reverseTraversal(self, node, return_edge: bool = False):
        """
        return a list traversing from root to node
        input:
            node: the name of a node (robot link)
            return_edge: return the eges if true, else return the nodes
        returns:
            a list of nodes or edges traversing from root to node
        """
        nodes = [node]
        while(1):
            try:
                parent = next(self.predecessors(node))
                nodes.append(parent)
                node = parent
            except StopIteration:  # at root node
                nodes.reverse()
                if return_edge:  # return the edges leading from root to node
                    return list(zip(nodes[:-1], nodes[1:]))
                else:  # return a list of nodes from root leading node
                    return nodes

    def worldTransform(self, node, t=np.eye(4), update: bool = True):
        """
        return the the transform from world space
        input:
            node: the name of a node (robot link)
            t: initial (world) 4x4 homogeneous transformation
            update: [bool] if true update the composed transorm of this node
        return:
            t: the composed 4x4 homogeneous transformation in world space
        """
        edges = self.reverseTraversal(node, return_edge=True)
        for e in edges:
            ge = self.edges[e]
            if ge["joint_type"] in {'revolute', 'continous'}:
                t = t@ge["transform"]@axisAngleRotation(
                    ge["axis"], ge["joint_pos"])
            else:
                raise NotImplementedError
        if update:
            self.nodes[node]["world_transform"] = t
        return t

    @property
    def rootNode(self):
        """
        return the name of root node (base link) of the graph
        ref:https://stackoverflow.com/questions/4122390/getting-the-root-head-of-a-digraph-in-networkx-python
        """
        return [n for n, d in self.in_degree() if d == 0][0]

    def updateWorldTransform(self, t=None):
        """
        update the world_transform property for nodes and edges
        input:
            t: initial (world) 4x4 homogeneous transformation
        return: 
            the updated graph
        """
        if t is None:
            t = np.eye(4)
        rn = self.rootNode
        self.nodes[rn]["world_transform"] = t
        for e in nx.edge_bfs(self, source=rn):
            parent_node = self.nodes[e[0]]
            child_node = self.nodes[e[1]]
            edge = self.edges[e]
            edge["world_transform"] = parent_node["world_transform"]@edge["transform"]
            child_node["world_transform"] = edge["world_transform"]@axisAngleRotation(
                edge["axis"], edge["joint_pos"])
        return self

    def nodesProperty(self, property_name, value):
        return [self.nodes[n][property_name] for n in self.nodes]

    @property
    def joint_pos(self):
        """return the joint_pos for all edges"""
        return np.fromiter((self.edges[e]["joint_pos"] for e in self.edges), dtype=np.float64)

    @joint_pos.setter
    def joint_pos(self, values):
        """
        set the joint_pos for all edges given values
        support dict input or iterable
        """
        if values is dict:
            for e, v in values:
                self.edges[e]["joint_pos"] = v
        else:
            for e, v in zip(self.edges, values):
                self.edges[e]["joint_pos"] = v

    def updateCoordinateOXYZ(self):
        for n in self.nodes:
            self.nodes[n]["coord_oxyz"] = getCoordinateOXYZ(
                self.nodes[n]["world_transform"])
        for e in self.edges:
            self.edges[e]["coord_oxyz"] = getCoordinateOXYZ(
                self.edges[e]["world_transform"])
            
    def exportURDF(self,path):
        """
        generate the URDF at path, path should be *./urdf
        """
        tree = URDF(self).export(path)
        return tree

class URDF:
    def __init__(self, graph):
        self.graph = graph

    def _urdfJoint(self, e, root):
        toSI = self.graph.unit.toSI
        name = f"{e[0]},{e[1]}"
        edge = self.graph.edges[e]

        joint_tag = etree.SubElement(root, "joint", name=name)
        joint_tag.attrib["type"] = edge["joint_type"]

        rpy = Rotation.from_matrix(edge["transform"][:3, :3]).as_euler(
            'xyz')*toSI("angle")  # roll pitch yaw
        xyz = edge["transform"][:3, 3]*toSI("length")
        origin_tag = etree.SubElement(joint_tag, "origin",
                                      xyz=" ".join(map(str, xyz)),
                                      rpy=" ".join(map(str, rpy)))
        parent_tag = etree.SubElement(joint_tag, "parent", link=e[0])
        child_tag = etree.SubElement(joint_tag, "child", link=e[1])

        axis_tag = etree.SubElement(joint_tag, "axis",
                                    xyz=' '.join(map(str, edge["axis"])))

        limit_tag = etree.SubElement(joint_tag, "limit")
        for key, value in edge["limit"].items():
            limit_tag.attrib[key] = str(value)  # to do convert to SI
        return joint_tag

    def _urdfLink(self, n, root, export_dir):
        toSI = self.graph.unit.toSI
        node = self.graph.nodes[n]
        name = f"{n}"
        link_tag = etree.SubElement(root, "link", name=name)
        inertial_tag = etree.SubElement(link_tag, "inertial")

        rpy = Rotation.from_matrix(node["transform"][:3, :3]).as_euler(
            'xyz')*toSI("angle")  # roll pitch yaw
        xyz = node["transform"][:3, 3]*toSI("length")

        inertial_origin_tag = etree.SubElement(inertial_tag, "origin",
                                               xyz=" ".join(map(str, xyz)),
                                               rpy=" ".join(map(str, rpy)))

        # deepcopy to avoid changing orginal vertices
        mesh = copy.deepcopy(node["vmd"].triMesh())  # trimesh mesh
        density = node["density"]*toSI("density")
        mesh.density = density
        mesh.vertices *= toSI("length")
        mass = mesh.mass
        moment_inertia = mesh.moment_inertia

        mass_tag = etree.SubElement(inertial_tag, "mass", value=str(mass))

        inertia_tag = etree.SubElement(
            inertial_tag, "inertia",
            ixx=str(moment_inertia[0, 0]), ixy=str(moment_inertia[0, 1]),
            ixz=str(moment_inertia[0, 2]), iyy=str(moment_inertia[1, 1]),
            iyz=str(moment_inertia[1, 2]), izz=str(moment_inertia[2, 2]))

        file_name = f"mesh/{name}.obj"

        _ = mesh.export(file_obj=f"{export_dir}/{file_name}")

        visual_tag = etree.SubElement(link_tag, "visual")
        visual_origin_tag = etree.SubElement(
            visual_tag, "origin",
            xyz=" ".join(map(str, xyz)),
            rpy=" ".join(map(str, rpy)),
        )
        visual_geometry_tag = etree.SubElement(visual_tag, "geometry")
        visual_geometry_mesh_tage = etree.SubElement(
            visual_geometry_tag, "mesh",
            filename=file_name
        )
        
        visual_material_tag = etree.SubElement(
            visual_tag, "material", name=to_hex(node["color"],keep_alpha=True))
        visual_material_color_tag = etree.SubElement(
            visual_material_tag, "color", rgba=" ".join(map(str, node["color"])))

        collision_tag = etree.SubElement(link_tag, "collision")

        collision_origin_tag = etree.SubElement(
            collision_tag, "origin",
            xyz=" ".join(map(str, xyz)),
            rpy=" ".join(map(str, rpy)),
        )
        collision_geometry_tag = etree.SubElement(collision_tag, "geometry")

        collision_geometry_mesh_tage = etree.SubElement(
            collision_geometry_tag, "mesh",
            filename=file_name
        )
        return link_tag

    def export(self, path):
        """
        generate the URDF at path, path should be *./urdf
        """
        # example:
        # URDF(graph).export(path= "../../data/urdf/test/robot.urdf")
        # ref: https://github.com/MPEGGroup/trimesh/blob/master/trimesh/exchange/urdf.py
        full_path = os.path.abspath(path)
        dir_path = os.path.dirname(full_path)
        name, ext = os.path.splitext(os.path.basename(full_path))
        mesh_dir_path = os.path.join(dir_path, "mesh")
        if not os.path.exists(mesh_dir_path):
            os.makedirs(mesh_dir_path)
        root = etree.Element("robot", name=f"{name}")
        links = [self._urdfLink(n, root, export_dir=dir_path)
                 for n in self.graph.nodes]
        joints = [self._urdfJoint(e, root) for e in self.graph.edges]
        tree = etree.ElementTree(root)
        tree.write(f"{full_path}", pretty_print=True,
                   xml_declaration=True, encoding="utf-8")
        print(f"URDF path:{full_path}\n")
        print(etree.tostring(root, pretty_print=True,
                             xml_declaration=True).decode('utf-8'))
        return tree