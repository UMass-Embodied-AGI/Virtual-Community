# This script contains all commands for generating characters the scene. Some steps have dependencies on previous steps, so we recommend running them sequentially as shown below.

SCENES=(NY)

for SCENE in "${SCENES[@]}"
do
	# Step 1: Generate characters and annotate images of known places
    echo ">>> Generating characters for $SCENE >>>"
    python3 character-generation/generate_characters.py --scene $SCENE --num_characters 15 --num_groups 4 --overwrite
    python3 vico/tools/annotate_known_place.py --scene $SCENE --num_characters 15 --num_groups 4
    
    # Step 2: Offset the xy positions for indoor places
    echo ">>> Offsetting xy positions for $SCENE >>>"
    python3 vico/tools/misc/offset_indoor_place_xy.py --scene $SCENE --num_agents 15
done
