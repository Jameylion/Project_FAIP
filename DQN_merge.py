import cv2
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
import torch
from highway_env.vehicle.kinematics import Performance, Logger
import highway_env
situation = "merge_in-v3" 
frameSize = (1280,560)
# out = cv2.VideoWriter('video'+situation+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16, frameSize)
out = cv2.VideoWriter('video'+situation+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)

TRAIN = True

if __name__ == '__main__':
    print(f'cuda availability:{torch.cuda.is_available()}')
    print(torch.cuda.device_count())
    # Create the environment
    env = gym.make(situation)#, render_mode="rgb_array")
    obs, info = env.reset()

    # Create the model
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=512,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                device='cuda',
                tensorboard_log="highway_dqn/")

    # Train the model
    print('learning....')
    model.learn(int(100000),progress_bar=True)
    print('done!')
    name = '_DQN/baselinev3'
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