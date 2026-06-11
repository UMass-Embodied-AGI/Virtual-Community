import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid", font_scale=1.1)

root_path = "ViCo/assets/scenes"
stats = {}

for scene in tqdm(os.listdir(root_path)):
    try:
        with open(os.path.join(root_path, scene, "road_data/road_data.xodr")) as f:
            stats[scene] = {}
            road_xodr = f.read()
            stats[scene]["road_counts"] = road_xodr.count('<road name')
            stats[scene]["junc_counts"] = road_xodr.count('<junction name')
    except:
        continue

scene_names = sorted(stats.keys())[:35]

# Plot 1: Buildings per Scene
Roads = [stats[scene]["road_counts"] for scene in scene_names]
print(f"avg roads: {sum(Roads)/len(Roads)}")
df_Roads = pd.DataFrame({'Scene': scene_names, 'Roads': Roads})

plt.figure(figsize=(18, 6))
sns.barplot(data=df_Roads, x='Scene', y='Roads', color='#4C72B0')
plt.xticks(rotation=45, ha='right')
plt.title("Number of Annotated Roads per Scene", weight='bold')
plt.tight_layout()
plt.show()

# Plot 2: Junctions per Scene
junctions = [stats[scene]["junc_counts"] for scene in scene_names]
print(f"avg junctions: {sum(junctions)/len(junctions)}")
df_junctions = pd.DataFrame({'Scene': scene_names, 'Junctions': junctions})

plt.figure(figsize=(18, 6))
sns.barplot(data=df_junctions, x='Scene', y='Junctions', color='#55A868')
plt.xticks(rotation=45, ha='right')
plt.title("Number of Annotated Junctions per Scene", weight='bold')
plt.tight_layout()
plt.show()