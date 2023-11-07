export DATASET_PATH="/app/hzq/data/NeRF/husky_joint/test"
# 
# colmap automatic_reconstructor --workspace_path $DATASET_PATH --image_path $DATASET_PATH/images
# 
colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images

colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

mkdir $DATASET_PATH/dense

colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000

colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.cache_size 64 \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --StereoFusion.cache_size 64 \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

# This stage commented out
#colmap poisson_mesher \
#    --input_path $DATASET_PATH/dense/fused.ply \
#    --output_path $DATASET_PATH/dense/meshed-poisson.ply

colmap delaunay_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-delaunay.ply