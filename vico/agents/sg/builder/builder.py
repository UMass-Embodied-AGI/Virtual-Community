import numpy as np
from dataclasses import dataclass, field
from .volume_grid import VolumeGridBuilder, VolumeGridBuilderConfig

@dataclass
class BuilderConfig:
    volume_grid_conf: VolumeGridBuilderConfig = field(default_factory=VolumeGridBuilderConfig)
    fov: float = 90.0
    debug: bool = False
    output_path: str = "output"
    logger: any = None

class Builder:
    def __init__(self, conf: BuilderConfig) -> None:
        self.volume_grid_builder = VolumeGridBuilder(conf.volume_grid_conf)
        self.objects = {}
        self.num_frames = 0
        self.fov = conf.fov
        self.debug = conf.debug
        self.output_path = conf.output_path
        self.logger = conf.logger
    
    def add_frame(self, rgb: np.ndarray, depth: np.ndarray, labels: np.ndarray, camera_ext: np.ndarray):
        self.num_frames += 1
        if rgb is None:
            return
        if labels is None:
            labels = - np.ones_like(depth, dtype=np.int32)
        self.volume_grid_builder.add_frame(rgb, depth, labels, self.fov, camera_ext)

    def add_object(self, obj):
        self.objects[obj.name + str(obj.idx)] = obj