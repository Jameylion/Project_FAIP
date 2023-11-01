import cv2
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from highway_env.vehicle.kinematics import Performance, Logger
import highway_env
situation = "merge_in-v2" 
frameSize = (1280,560)
# out = cv2.VideoWriter('video'+situation+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16, frameSize)
out = cv2.VideoWriter('video'+situation+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)

print(f'cuda availability:{torch.cuda.is_available()}')
# Create the environment
n_cpu = 6
batch_size = 64
env = make_vec_env(situation, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
model = PPO("MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.9,
            verbose=2,
            tensorboard_log="racetrack_ppo/")

# Train the model
print('learning....')
model.learn(int(100000),progress_bar=True)
print('done!')
name = '_DQN/first_learn'
model.save(situation+name)

perfm = Performance()
lolly = Logger()

number_of_runs = 100
for f in range(number_of_runs):
    done = truncated = False
    obs, info = env.reset()
    reward = 0

    ego_car = env.controlled_vehicles[0]

    stepcounter = 0
    
    while (not done) and ego_car.speed > 2 and stepcounter < 800:        
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        stepcounter += 1
        lolly.file(ego_car)
        env.render()
        

    perfm.add_measurement(lolly)
    lolly.clear_log()
    print(f)

perfm.print_performance()
print('DONE')

number_of_collisions = 0
T = 1
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)  # env.step(action.item(0))
    #print(action)
    #print(obs)
    #print(info)
    #print(reward)
    if info.get('crashed'):
        number_of_collisions += 1
    env.render()
    cur_frame = env.render(mode="rgb_array")
    out.write(cur_frame)
  #print('crashrate is '+str(float(number_of_collisions)/T)+' and T is'+str(T))
  T+=1

out.release()
print('number_of_collisions is:', number_of_collisions)
print('DONE')