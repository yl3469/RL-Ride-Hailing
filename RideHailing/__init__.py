from gym.envs.registration import register

register(
    id= 'RideHailing-v0',
    entry_point='RideHailing.envs:RideHailingEnv'
)