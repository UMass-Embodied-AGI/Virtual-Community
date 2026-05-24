# This script contains all commands for annotating the scene. Some steps have dependencies on previous steps, so we recommend running them sequentially as shown below.

SCENES=(LONDON)

SOURCE_PATH="vico/assets/ViCo/scene/v1"
TARGET_PATH="vico/assets/scenes"

for SCENE in "${SCENES[@]}"
do
	# Step 1: Prepare necessary files
	echo ">>> Preparing necessary files for $SCENE >>>"
	DEST="$TARGET_PATH/$SCENE"
    DEST_raw="$TARGET_PATH/$SCENE/raw/"
    mkdir -p "$DEST_raw"
	cp -r "$SOURCE_PATH/$SCENE/road_data" "$DEST"
    cp "$SOURCE_PATH/$SCENE/building_to_osm_tags.json" "$DEST_raw"
	cp "$SOURCE_PATH/$SCENE/center.txt" "$DEST_raw"
	
	# Step 2: Annotate scene places
	echo ">>> Annotating places for $SCENE >>>"
	python3 vico/tools/annotate_scene.py --scene $SCENE --search_original_places --filter_places --filter_distance_square 300 --search_resolution 45.0 --save_metadata --generate_metadata --remove_temp --overwrite
	echo ">>> Annotating global for $SCENE >>>"
	python3 vico/tools/annotate_global.py --scene $SCENE # This is optional, but useful for debugging anything wrong in this step
	
	# Step 3: Annotate transit and annotate the global image for places and transit
	echo ">>> Annotating transit for $SCENE >>>"
	python3 vico/tools/annotate_transit.py --scene $SCENE --num_bus_clusters 3 --num_bicycle_clusters 7 --init_bus_road_min_length 100.0 --reduce_bicycle_road_constraint
	# --remove_bus_road_constraint
	echo ">>> Annotating global for $SCENE >>>"
	python3 vico/tools/annotate_global.py --scene $SCENE
done
