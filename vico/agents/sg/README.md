# Scene Graph Building Procedure

## Set up

You need to build some C++ dynamic link libraries for low-level procedure.

```bash
cd sg
./setup.sh
```

## Introduction

### Volume Grid

We use a data structure to maintain the volume grid representation of the environment. It supports:

* Insert a point with color with average time complexity $O(1)$.
* Query whether a voxel exists in $O(1)$ and its color in $O(k)$, where $k$ is the number of voxels which has the same $(x,y)$.
* Query all voxels in a given box.
* Query the highest and lowest voxel at any $(x,y)$ with time complexity $O(1)$.