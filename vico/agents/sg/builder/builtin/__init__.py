import os
import ctypes

lib_builder = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libbuilder.so'))

lib_builder.init_volume_grid.argtypes = [ctypes.c_float, ctypes.c_int] # voxel_size, num_threads
lib_builder.init_volume_grid.restype = ctypes.c_void_p # backend object

lib_builder.volume_grid_add_frame.argtypes = [ctypes.c_void_p, # backend object
                                              ctypes.POINTER(ctypes.c_ubyte), # rgb
                                              ctypes.POINTER(ctypes.c_float), # depth
                                              ctypes.POINTER(ctypes.c_int), # label
                                              ctypes.c_int, ctypes.c_int, ctypes.c_float, # width, height, fov
                                              ctypes.POINTER(ctypes.c_float)] # camera_ext
lib_builder.volume_grid_add_frame.restype = None

lib_builder.volume_grid_insert_from_numpy.argtypes = [ctypes.c_void_p, # backend object
                                                      ctypes.POINTER(ctypes.c_float), # points
                                                      ctypes.POINTER(ctypes.c_ubyte), # colors
                                                      ctypes.POINTER(ctypes.c_int), # label
                                                      ctypes.c_int] # size
lib_builder.volume_grid_insert_from_numpy.restype = None

lib_builder.volume_grid_to_numpy.argtypes = [ctypes.c_void_p, # backend object
                                             ctypes.POINTER(ctypes.c_float), # points
                                             ctypes.POINTER(ctypes.c_ubyte),
                                             ctypes.POINTER(ctypes.c_int)] # colors
lib_builder.volume_grid_to_numpy.restype = None

lib_builder.free_volume_grid.argtypes = [ctypes.c_void_p]
lib_builder.free_volume_grid.restype = None

lib_builder.volume_grid_size.argtypes = [ctypes.c_void_p]
lib_builder.volume_grid_size.restype = ctypes.c_size_t

lib_builder.volume_grid_memory_size.argtypes = [ctypes.c_void_p]
lib_builder.volume_grid_memory_size.restype = ctypes.c_size_t

lib_builder.volume_grid_get_z.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float] # backend object, x, y, radius
lib_builder.volume_grid_get_z.restype = ctypes.c_float

lib_builder.volume_grid_get_label.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float] # backend object, x, y, z, radius
lib_builder.volume_grid_get_label.restype = ctypes.c_int

lib_builder.volume_grid_get_bound.argtypes = [ctypes.c_void_p, # backend object
                                              ctypes.POINTER(ctypes.c_float), # min (x, y, z)
                                              ctypes.POINTER(ctypes.c_float)] # max (x, y, z)
lib_builder.volume_grid_get_bound.restype = None

lib_builder.volume_grid_get_overlap.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float] # backend object1, backend object2, radius
lib_builder.volume_grid_get_overlap.restype = ctypes.c_float

lib_builder.volume_grid_radius_denoise.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float] # backend object, min_points, radius
lib_builder.volume_grid_radius_denoise.restype = None

lib_builder.image_to_pcd.argtypes = [ctypes.POINTER(ctypes.c_ubyte), # rgb
                                         ctypes.POINTER(ctypes.c_float), # depth
                                         ctypes.POINTER(ctypes.c_int), # label(input)
                                         ctypes.POINTER(ctypes.c_float), # point
                                         ctypes.POINTER(ctypes.c_ubyte), # colors
                                         ctypes.POINTER(ctypes.c_int), # label(output)
                                         ctypes.c_int, ctypes.c_int, ctypes.c_float, # width, height, fov
                                         ctypes.POINTER(ctypes.c_float)] # camera_ext
lib_builder.image_to_pcd.restype = None

lib_builder.convex_hull.argtypes = [ctypes.POINTER(ctypes.c_int), # points
                                    ctypes.c_int, # size
                                    ctypes.POINTER(ctypes.c_int)] # size
lib_builder.convex_hull.restype = ctypes.POINTER(ctypes.c_int) # hull

lib_builder.dist_to_hull.argtypes = [ctypes.POINTER(ctypes.c_int), # point
                                    ctypes.POINTER(ctypes.c_int), # hull
                                    ctypes.c_int] # size
lib_builder.dist_to_hull.restype = ctypes.c_float # distance

lib_builder.navigate.argtypes = [ctypes.c_void_p, # backend object
                                 ctypes.POINTER(ctypes.c_float), # start
                                 ctypes.POINTER(ctypes.c_float), # goal (convex hull)
                                 ctypes.c_int, # goal size
                                 ctypes.c_int, # radius(number of voxels combine to a grid)
                                 ctypes.POINTER(ctypes.c_int),
                                 ctypes.POINTER(ctypes.c_float), # ref_path
                                 ctypes.c_int] # ref_path size
lib_builder.navigate.restype = ctypes.POINTER(ctypes.c_float) # path

lib_builder.get_occurancy_map.argtypes = [ctypes.c_void_p, # backend object
                                          ctypes.c_int, # radius
                                          ctypes.POINTER(ctypes.c_int), # x_min
                                          ctypes.POINTER(ctypes.c_int), # y_min
                                          ctypes.POINTER(ctypes.c_int), # x_max
                                          ctypes.POINTER(ctypes.c_int)] # y_max
lib_builder.get_occurancy_map.restype = ctypes.POINTER(ctypes.c_ubyte) # map: 1 unknown, 2 obstacle, 3 road

lib_builder.has_obstacle.argtypes = [ctypes.c_void_p, # backend object
                                     ctypes.c_int, # radius
                                     ctypes.POINTER(ctypes.c_float)] # bounding box
lib_builder.has_obstacle.restype = ctypes.c_ubyte

lib_region = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libregion.so'))
lib_region.smooth.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)]
lib_region.smooth.restype = None

lib_region.bfs.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
lib_region.bfs.restype = None

lib_region.adj_matrix.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
lib_region.adj_matrix.restype = None