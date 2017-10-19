from gym.envs.registration import register

register(
    id='Pusher2-v0',
    entry_point='gym_throwandpush.envs:Pusher2Env',
    max_episode_steps=500,
    reward_threshold=-3.75,
)

register(
    id='Pusher2Pixel-v0',
    entry_point='gym_throwandpush.envs:Pusher2PixelEnv',
    kwargs={'base_env_id': 'Pusher2-v0'}
)
