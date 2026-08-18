from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


SCENARIOS = {
    'S0': {
        'failure_mode': 'disabled',
        'pattern_seed': '0',
    },
    'S1': {
        'failure_mode': 'disabled',
        'pattern_seed': '1',
    },
    'S2': {
        'failure_mode': 'mixed',
        'pattern_seed': '2',
    },
    'S3': {
        'failure_mode': 'hole',
        'pattern_seed': '3',
    },
    'S4': {
        'failure_mode': 'table_leakage',
        'pattern_seed': '4',
    },
    'S5': {
        'failure_mode': 'edge_flying',
        'edge_band_pixels': '3',
        'flying_point_offset_m': '0.16',
        'flying_point_stride': '4',
        'pattern_seed': '5',
    },
    'S6': {
        'failure_mode': 'mixed',
        'pattern_seed': '6',
    },
    'S7': {
        'failure_mode': 'reflective',
        'biased_depth_offset_m': '-0.05',
        'flying_point_offset_m': '0.14',
        'pattern_seed': '7',
    },
}

DEFAULT_ARGS = {
    'roi_center_u_ratio': '0.50',
    'roi_center_v_ratio': '0.58',
    'roi_width_ratio': '0.22',
    'roi_height_ratio': '0.22',
    'table_leak_depth_m': '1.20',
    'flying_point_offset_m': '0.12',
    'biased_depth_offset_m': '-0.04',
    'edge_band_pixels': '2',
    'flying_point_stride': '5',
}


def launch_scenario(context):
    scenario_id = LaunchConfiguration('scenario_id').perform(context).upper()
    if scenario_id not in SCENARIOS:
        raise RuntimeError(f'Unknown M3 scenario_id: {scenario_id}')

    args = dict(DEFAULT_ARGS)
    args.update(SCENARIOS[scenario_id])
    args['scenario_id'] = scenario_id
    args['headless'] = LaunchConfiguration('headless').perform(context)
    args['show_rviz'] = LaunchConfiguration('show_rviz').perform(context)
    args['world_file'] = LaunchConfiguration('world_file').perform(context)
    args['world_name'] = LaunchConfiguration('world_name').perform(context)
    args['enable_target_mask_emulator'] = LaunchConfiguration(
        'enable_target_mask_emulator').perform(context)
    args['point_cloud_stride'] = LaunchConfiguration('point_cloud_stride').perform(context)

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare('ghost_mgg_bringup'),
                        'launch',
                        'm3_failure_camera_inspect.launch.py',
                    ]
                )
            ),
            launch_arguments=args.items(),
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument('scenario_id', default_value='S6'),
            DeclareLaunchArgument('headless', default_value='true'),
            DeclareLaunchArgument('show_rviz', default_value='false'),
            DeclareLaunchArgument('world_file', default_value='m2_tabletop_visual.sdf'),
            DeclareLaunchArgument('world_name', default_value='ghost_mgg_m2_visual'),
            DeclareLaunchArgument('enable_target_mask_emulator', default_value='true'),
            DeclareLaunchArgument('point_cloud_stride', default_value='1'),
            OpaqueFunction(function=launch_scenario),
        ]
    )
