from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class DenseFeatureRecord:
    image_id:int; rel_path:str=''; width:int|None=None; height:int|None=None; keypoints:object|None=None; descriptors:object|None=None; scores:object|None=None; status:str='not_run'; num_keypoints:int=0
@dataclass
class FramePairRecord:
    id:int|None=None; image_id_a:int=0; image_id_b:int=0; status:str='candidate'; baseline_m:float|None=None; optical_axis_angle_deg:float|None=None; common_marker_count:int=0; estimated_overlap_score:float=0.0; num_raw_matches:int=0; num_epipolar_inliers:int=0
@dataclass
class PairMatchRecord:
    pair_id:int; feature_idx_a:int; feature_idx_b:int; x_a:float; y_a:float; x_b:float; y_b:float; match_score:float|None=None; epipolar_error:float|None=None; is_epipolar_inlier:int=0
@dataclass
class TrackObservationRecord:
    track_id:int; image_id:int; feature_idx:int; x:float; y:float
@dataclass
class TrackRecord:
    id:int|None=None; status:str='candidate'; num_observations:int=0; num_images:int=0; x:float|None=None; y:float|None=None; z:float|None=None; mean_reprojection_error_px:float|None=None; max_reprojection_error_px:float|None=None; min_triangulation_angle_deg:float|None=None
@dataclass
class DensePointRecord:
    id:int|None=None; track_id:int|None=None; x:float=0.0; y:float=0.0; z:float=0.0; is_active:int=1; source:str='triangulated'
@dataclass
class DenseStageSummary:
    stage:str; total:int=0; success:int=0; failed:int=0; details:str=''
@dataclass
class XFeatExtractionConfig:
    max_keypoints:int=20000; detection_threshold:float=0.01; resize_max_side:int=1600; device:str='auto'; use_cuda:bool=True
@dataclass
class PairSelectionConfig:
    min_baseline_m:float=0.03; max_baseline_m:float=5.0; max_optical_axis_angle_deg:float=60.0; min_common_markers:int=0; max_pairs_per_image:int=10; use_common_markers_bonus:bool=True; use_projected_marker_overlap:bool=True
@dataclass
class MatchingConfig:
    max_pairs_to_match:int|None=None; min_match_score:float=0.0; device:str='auto'; force_recompute:bool=False
@dataclass
class EpipolarFilterConfig:
    error_type:str='ray_angular'; max_ray_angular_error:float=0.003; max_sampson_error:float=1e-4; min_baseline_m:float=0.03; force_recompute:bool=False
@dataclass
class TriangulationConfig:
    min_triangulation_angle_deg:float=1.0; max_reprojection_error_px:float=5.0; max_ray_gap_m:float=0.25; max_depth_m:float=100.0
@dataclass
class DuplicateMergeConfig:
    duplicate_merge_radius_m:float=0.02; max_merged_mean_reprojection_error_px:float=3.0; max_merged_reprojection_error_px:float=6.0
@dataclass
class DenseBAConfig:
    mode:str='points_only'; huber_scale_px:float=3.0
@dataclass
class DenseReconstructionConfig:
    extraction:XFeatExtractionConfig=field(default_factory=XFeatExtractionConfig)
