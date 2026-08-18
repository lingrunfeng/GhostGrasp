uint8 GRASP_TYPE_UNKNOWN=0
uint8 GRASP_TYPE_TOP=1
uint8 GRASP_TYPE_SIDE=2

uint8 VALIDATION_UNKNOWN=0
uint8 VALIDATION_VALID=1
uint8 VALIDATION_REJECTED=2

string grasp_id
geometry_msgs/PoseStamped grasp_pose
geometry_msgs/PoseStamped pregrasp_pose
geometry_msgs/Vector3 approach_vector
float64 gripper_width_m
uint8 grasp_type
float64 score
uint8 validation_state
string failure_reason
