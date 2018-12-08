
'''*******************************************************

By Gokul NC <gokulnc@ymail.com> ( http://about.me/GokulNC )

*******************************************************'''
import numpy as np
from pynput import keyboard
from threading import Thread
import time

from carla_rl.carla_environment_wrapper import CarlaEnvironmentWrapper as CarlaEnv
from carla_settings import get_carla_settings

steering_strength = 0.5
gas_strength = 1.0
brake_strength = -0.5

action = [0.0, 0.0]
reset = False
total_reward = 0.0

frame_skip = 5  # No. of frames to skip, i.e., the no. of frames in which to produce consecutive actions. Already CARLA is low FPS, so better be 1

debug_logs = True

if debug_logs:
    frame_id = 0
    total_frames = 100  # No. of frames once to print the FPS rate
    start_time = time.time()


def start_listen():
    # Listen for keypresses to control game via Terminal. Inspired from: https://pypi.python.org/pypi/pynput
    global action, reset, steering_strength, gas_strength, brake_strength

    def on_press(key):
        global action, reset, steering_strength, gas_strength, brake_strength
        if key == keyboard.KeyCode(char="w"):
            action[1] = gas_strength
        elif key == keyboard.KeyCode(char="s"):
            action[1] = brake_strength
        elif key == keyboard.KeyCode(char="a"):
            action[0] = -steering_strength
        elif key == keyboard.KeyCode(char="d"):
            action[0] = steering_strength
        elif key == keyboard.Key.space:
            reset = True

    def on_release(key):
        global action
        if key == keyboard.KeyCode(char="w") or key == keyboard.KeyCode(char="s"):
            action[1] = 0.0
        elif key == keyboard.KeyCode(char="a") or key == keyboard.KeyCode(char="d"):
            action[0] = 0.0

    # Collect events until released
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()


config_file = "/media/ankurrc/new_volume/689_ece_rl/project/code/autodrive/mysettings.ini"

print("Creating Environment..")
settings = get_carla_settings()
env = CarlaEnv(is_render_enabled=True, automatic_render=True, num_speedup_steps=10, run_offscreen=False,
               cameras=['SceneFinal'], save_screens=False, carla_settings=settings, carla_server_settings=config_file)

print("Resetting the environment..")
env.reset()

# Start listening to key presses and update actions
t = Thread(target=start_listen)
t.start()

print("Start playing..... :)")

try:
    while True:

        if debug_logs:
            print("Action: "+str(action))
            frame_id = (frame_id+1) % total_frames
            if frame_id == 0:
                end_time = time.time()
                print("FPS: "+str(total_frames/(end_time-start_time)))
                start_time = end_time

        r = 0.0
        for _ in range(frame_skip):
            # print(action_map[tuple(action)])m
            observation, reward, done, _ = env.step(action)
            print("Action:{}".format(action))
            # env.render()
            # env.save_screenshots()
            r += reward
            if done:
                break

        total_reward += r
        if reset:
            done = True

        if done:
            env.reset(settings=get_carla_settings())
            reset = False
            print("Total reward in episode:"+str(total_reward))
            total_reward = 0.0

except KeyboardInterrupt:
    t.join(timeout=1)
    env.close_client_and_server()
