import cv2
import gymnasium as gym
import sys
sys.modules["gym"] = gym
import highway_env
highway_env.register_highway_envs()
from stable_baselines3 import PPO, DQN
from sb3_contrib import TRPO
import torch
import tensorflow
from highway_env.vehicle.kinematics import Performance, Logger


#situation = "intersection-v1"
# situation = "complex_city-v1"
#situation = "racetrack-v0"
situation = "merge-v1"


frameSize = (1280,560)
# out = cv2.VideoWriter('video'+situation+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16, frameSize)
out = cv2.VideoWriter('video'+situation+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)


env = gym.make(situation)
env.configure({
    "screen_width": 1280,
    "screen_height": 560,
    "renderfps": 16
})
env.configure({
    "action": {
        "type": "ContinuousAction"
    },
    "offroad_terminal": False,
    "other_vehicles": 1,
    "vehicles_count": 6,
    "initial_vehicle_count": 0,
    "spawn_probability": 0.
    
    
})
env.configure({
    "simulation_frequency": 15,
    "policy_frequency":15
})

env.reset()
n_cpu = 6
batch_size = 64

model = DQN('MlpPolicy', env,
              policy_kwargs=None,
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="highway_dqn/",)


# uncomment the lines below if you want to train a new model

# model = TRPO.load(situation+'_trpo/fixed_test')


model.set_env(env)
# model.set_parameters(params)#, exact_match=True)

print('learning....')
model.learn(int(100000),progress_bar=True)
print('done!')
name = '_dqn/test_merge_in'
model.save(situation+name)

print()
print(situation+name+" is saved!!")
print()



########## Load and test saved model##############
#model = TRPO.load('situation'+'_trpo/model14.5')
#while True:


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
