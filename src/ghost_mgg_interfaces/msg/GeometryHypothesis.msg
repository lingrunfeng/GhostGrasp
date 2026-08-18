uint8 SHAPE_UNKNOWN=0
uint8 SHAPE_BOX=1
uint8 SHAPE_CYLINDER=2
uint8 SHAPE_CUP_LIKE=3

uint8 VALIDATION_UNKNOWN=0
uint8 VALIDATION_VALID=1
uint8 VALIDATION_REJECTED=2

string hypothesis_id
uint8 shape_type
geometry_msgs/PoseStamped pose_camera
geometry_msgs/PoseStamped pose_base
geometry_msgs/Vector3 dimensions_m
ScoreBreakdown score
float64 confidence
float64 uncertainty
GraspCandidate[] grasp_candidates
string provenance
uint8 validation_state
string failure_reason
