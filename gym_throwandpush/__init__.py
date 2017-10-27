from gym.envs.registration import register

register(
    id='Pusher2-v0',
    entry_point='gym_throwandpush.envs:Pusher2Env',
    max_episode_steps=500,
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
    max_episode_steps=500,
    reward_threshold=4800.0,
)

