from gym.envs.registration import register

register(
    id='Pusher2-v0',
    entry_point='gym_throwandpush.envs:Pusher2Env',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Pusher2Pixel-v0',
    entry_point='gym_throwandpush.envs:Pusher2PixelEnv',
    kwargs={'base_env_id': 'Pusher2-v0'}
)

register(
    id='Pusher2Plus-v0',
    entry_point='gym_throwandpush.envs:Pusher2PlusEnv',
    kwargs={'base_env_id': 'Pusher2-v0'}
)

register(
    id='HalfCheetah2-v0',
    entry_point='gym_throwandpush.envs:Cheetah2Env',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HalfCheetah2Pixel-v0',
    entry_point='gym_throwandpush.envs:Cheetah2PixelEnv',
    kwargs={'base_env_id': 'HalfCheetah2-v0'}
)

register(
    id='HalfCheetah2Plus-v0',
    entry_point='gym_throwandpush.envs:Cheetah2PlusEnv',
    kwargs={'base_env_id': 'HalfCheetah2-v0'}
)

# new from Berkeley
register(
    id='Pusher3Dof-v0',
    entry_point='gym_throwandpush.envs:PusherEnv3Dof',
    max_episode_steps=100,
    reward_threshold=0.0,
)
register(
    id='Pusher3Dof2-v0',
    entry_point='gym_throwandpush.envs:Pusher3Dof2Env',
    max_episode_steps=100,
    reward_threshold=0.0,
)
register(
    id='Pusher3Dof2Pixel-v0',
    entry_point='gym_throwandpush.envs:Pusher3Dof2PixelEnv',
    kwargs={'base_env_id': 'Pusher3Dof2-v0'}
)
register(
    id='Pusher3Dof2Plus-v0',
    entry_point='gym_throwandpush.envs:Pusher3Dof2Plus',
    kwargs={'base_env_id': 'Pusher3Dof2-v0'},
)
register(
    id='Pusher3Dof2Inverse-v0',
    entry_point='gym_throwandpush.envs:Pusher3Dof2Inverse',
    kwargs={'base_env_id': 'Pusher3Dof2-v0'},
)
register(
    id='Pusher3Dof2Lstm-v0',
    entry_point='gym_throwandpush.envs:Pusher3Dof2Lstm',
    kwargs={'base_env_id': 'Pusher3Dof2-v0'},
)
register(
    id='Pusher3Dof2Plus2-v0',
    entry_point='gym_throwandpush.envs:Pusher3Dof2Plus2',
    kwargs={'base_env_id': 'Pusher3Dof2-v0'},
)
register(
    id='Striker3Dof-v0',
    entry_point='gym_throwandpush.envs:StrikerEnv3Dof',
    max_episode_steps=100,
)
register(
    id='Striker3Dof-v1',
    entry_point='gym_throwandpush.envs:StrikerEnv3Dof2',
    max_episode_steps=100,
)
register(
    id='Striker-v1',
    entry_point='gym_throwandpush.envs:StrikerEnv',
    max_episode_steps=100,
)
register(
    id='Striker-v2',
    entry_point='gym_throwandpush.envs:StrikerEnv2',
    max_episode_steps=100,
)
register(
    id='StrikerPlus-v0',
    entry_point='gym_throwandpush.envs:StrikerPlus',
    kwargs={'base_env_id': 'Striker-v1'}
)
register(
    id='StrikerLstm-v0',
    entry_point='gym_throwandpush.envs:StrikerLstm',
    kwargs={'base_env_id': 'Striker-v1'},
)

register(
    id='Thrower-v1',
    entry_point='gym_throwandpush.envs:ThrowerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='ThrowerPlus-v0',
    entry_point='gym_throwandpush.envs:ThrowerPlus',
    kwargs={'base_env_id': 'Thrower-v1'}
)
