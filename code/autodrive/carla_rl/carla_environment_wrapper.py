import sys
import time
import subprocess
import signal
from os import path, environ

from carla.client import CarlaClient
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.sensor import Camera
from carla.client import VehicleControl
from carla.image_converter import depth_to_logarithmic_grayscale, depth_to_local_point_cloud, depth_to_array

from carla_rl.environment_wrapper import EnvironmentWrapper
from carla_rl.utils import *
import carla_rl.carla_config as carla_config
from carla_rl.renderer import Renderer

import numpy as np
import random

try:
	if 'CARLA_ROOT' not in environ:
		raise Exception()
except Exception:
	print("CARLA Environment variable CARLA_ROOT not set")
	sys.exit(1)


class CarlaLevel(Enum):
	TOWN1 = "/Game/Maps/Town01"
	TOWN2 = "/Game/Maps/Town02"


class CarlaEnvironmentWrapper(EnvironmentWrapper):
	def __init__(self,
					num_speedup_steps=10,
					require_explicit_reset=True,
					is_render_enabled=False,
					automatic_render = False,
					early_termination_enabled=False,
					run_offscreen=False,
					cameras=['SceneFinal'],
					save_screens=False,
					settings_file=None):

		EnvironmentWrapper.__init__(self, is_render_enabled, save_screens)

		self.automatic_render = automatic_render
		self.episode_max_time = 100000  # miliseconds for each episode
		self.allow_braking = True
		self.log_path = "logs"
		self.verbose = True
		self.observation = None
		self.num_speedup_steps = num_speedup_steps

		self.rgb_camera_name = 'CameraRGB'
		self.rgb_camera = 'SceneFinal' in cameras

		self.is_game_ready_for_input = False
		self.run_offscreen = run_offscreen
		self.kill_when_connection_lost = True
		# server configuration

		self.port = get_open_port()
		self.host = 'localhost'
		# Why town2: https://github.com/carla-simulator/carla/issues/10#issuecomment-342483829
		self.level = 'town2'
		self.map = CarlaLevel().get(self.level)

		# client configuration
		self.config = settings_file

		if self.config:
			# load settings from file
			with open(self.config, 'r') as fp:
				self.settings = fp.read()
		else:
			self.settings = CarlaSettings()

		self.car_speed = 0.
		self.max_speed = 35.
		# Will be true only when setup_client_and_server() is called, either explicitly, or by reset()
		self.is_game_setup = False

		# measurements
		self.autopilot = None
		self.kill_if_unmoved_for_n_steps = 60
		self.unmoved_steps = 0.0

		self.early_termination_enabled = early_termination_enabled
		if self.early_termination_enabled:
			self.max_neg_steps = 70
			self.cur_neg_steps = 0
			self.early_termination_punishment = 20.0

		# env initialization
		if not require_explicit_reset:
			self.reset(True)

		# camera-view renders
		if self.automatic_render:
			self.init_renderer()
		if self.save_screens:
			create_dir(self.images_path)
			self.rgb_img_path = self.images_path+"/rgb/"

	def setup_client_and_server(self, reconnect_client_only=False):
		# open the server
		if not reconnect_client_only:
			self.server = self._open_server()
			self.server_pid = self.server.pid  # To kill incase child process gets lost

		# open the client
		self.game = CarlaClient(self.host, self.port, timeout=99999999)
		# It's taking a very long time for the server process to spawn, so the client needs to wait or try sufficient no. of times lol
		self.game.connect(connection_attempts=100)
		scene = self.game.load_settings(self.settings)

		# get available start positions
		positions = scene.player_start_spots
		self.num_pos = len(positions)
		self.iterator_start_positions = 0
		self.is_game_setup = self.server and self.game
		return

	def close_client_and_server(self):
		self._close_server()
		print("Disconnecting the client")
		self.game.disconnect()
		self.game = None
		self.server = None
		self.is_game_setup = False
		return

	def _open_server(self):
		# Note: There is no way to disable rendering in CARLA as of now
		# https://github.com/carla-simulator/carla/issues/286
		# decrease the window resolution if you want to see if performance increases
		# Command: $CARLA_ROOT/CarlaUE4.sh /Game/Maps/Town02 -benchmark -carla-server -fps=15 -world-port=9876 -windowed -ResX=480 -ResY=360 -carla-no-hud
		# To run off_screen: SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 <command> #https://github.com/carla-simulator/carla/issues/225
		my_env = None
		if self.run_offscreen:
			# my_env = {**os.environ, 'SDL_VIDEODRIVER': 'offscreen', 'SDL_HINT_CUDA_DEVICE':'0'}
			pass
		with open(self.log_path, "wb") as out:
			cmd = [path.join(environ.get('CARLA_ROOT'), 'CarlaUE4.sh'),
					self.map,
					"-benchmark", "-carla-server", "-fps=10", "-world-port={}".format(
						self.port),
					"-windowed -ResX={} -ResY={}".format(
						carla_config.server_width, carla_config.server_height),
					"-carla-no-hud"]
			if self.config:
				cmd.append("-carla-settings={}".format(self.config))
			p = subprocess.Popen(cmd, stdout=out, stderr=out, env=my_env)
		return p

	def _close_server(self):
		if self.kill_when_connection_lost:
			os.killpg(os.getpgid(self.server.pid), signal.SIGKILL)
			return
		no_of_attempts = 0
		while is_process_alive(self.server_pid):
			print("Trying to close Carla server with pid %d" % self.server_pid)
			if no_of_attempts < 5:
				self.server.terminate()
			elif no_of_attempts < 10:
				self.server.kill()
			elif no_of_attempts < 15:
				os.kill(self.server_pid, signal.SIGTERM)
			else:
				os.kill(self.server_pid, signal.SIGKILL)
			time.sleep(10)
			no_of_attempts += 1

	def check_early_stop(self, player_measurements, immediate_reward):

		if player_measurements.intersection_offroad > 0.95 or \
		   immediate_reward < -1 or \
		   (self.control.throttle == 0.0 and player_measurements.forward_speed < 0.1 and self.control.brake != 0.0):

			self.cur_neg_steps += 1
			early_done = (self.cur_neg_steps > self.max_neg_steps)
			if early_done:
				print("Early kill the mad car")
				return early_done, self.early_termination_punishment
		else:
			self.cur_neg_steps /= 2  # Exponentially decay
		return False, 0.0

	def _update_state(self):

		# get measurements and observations
		try:
			measurements, sensor_data = self.game.read_data()
		except:
			# Connection between cli and server lost; reconnect
			if self.kill_when_connection_lost:
				raise
			print("Connection to server lost while reading state. Reconnecting...........")
			self.close_client_and_server()
			self.setup_client_and_server(reconnect_client_only=False)
			self.done = True

		self.location = (measurements.player_measurements.transform.location.x,
						measurements.player_measurements.transform.location.y,
						measurements.player_measurements.transform.location.z)

		is_collision = measurements.player_measurements.collision_vehicles != 0 \
						or measurements.player_measurements.collision_pedestrians != 0 \
						or measurements.player_measurements.collision_other != 0

		# CARLA doesn't recognize if collision occured and colliding speed is less than 5 km/h (Around 0.7 m/s)
		# Ref: https://github.com/carla-simulator/carla/issues/13
		# Recognize that as a collision
		self.car_speed = measurements.player_measurements.forward_speed

		if self.control.throttle > 0. and self.car_speed < 0.75 and self.control.brake >= 0. and self.is_game_ready_for_input:
			self.unmoved_steps += 1
			if self.unmoved_steps > self.kill_if_unmoved_for_n_steps:
				is_collision = True
				print("Car stuck somewhere.")
		elif self.unmoved_steps > 0:
			# decay slowly, since it may be stuck and not accelerate few times
			self.unmoved_steps -= 0.50

		if is_collision:
			print("Collision occured")

		# Reward Shaping:
		speed_reward = self.car_speed - 1
		if speed_reward > 30.:
			speed_reward = 30.

		self.reward = speed_reward \
					- (measurements.player_measurements.intersection_otherlane * 5) \
					- (measurements.player_measurements.intersection_offroad * 5) \
					- is_collision * 100 \
					- np.abs(self.control.steer) * 10
		# Scale down the reward by a factor
		# self.reward /= 10

		if self.early_termination_enabled:
			early_done, punishment = self.check_early_stop(
				measurements.player_measurements, self.reward)
			if early_done:
				self.done = True
			self.reward -= punishment

		# update measurements
		self.observation = {
							'acceleration': measurements.player_measurements.acceleration,
							'forward_speed': measurements.player_measurements.forward_speed
		}

		if self.rgb_camera:
			self.observation['rgb_image'] = sensor_data[self.rgb_camera_name].data

		self.autopilot = measurements.player_measurements.autopilot_control

		if (measurements.game_timestamp >= self.episode_max_time) or is_collision:
			# screen.success('EPISODE IS DONE. GameTime: {}, Collision: {}'.format(str(measurements.game_timestamp),
			#																	  str(is_collision)))
			self.done = True

	def _take_action(self, action):

		if not self.is_game_setup:
			print("Reset the environment by reset() before calling step()")
			sys.exit(1)

	  	# assert len(actions) == 2, "Send actions in the format [steer, accelaration]"

		self.control = VehicleControl()
		self.control.steer = np.clip(action[0], -1, 1)
		self.control.throttle = np.clip(action[1], 0, 1)
		self.control.brake = np.abs(np.clip(action[1], -1, 0))

		if not self.allow_braking:
			self.control.brake = 0

		self.control.hand_brake = False
		self.control.reverse = False

		# prevent braking
		if not self.allow_braking or self.control.brake < 0.1 or self.control.throttle > self.control.brake:
			self.control.brake = 0

		# prevent over speeding (first convert from m/s to km/h)
		if self.car_speed * 3.6 > self.max_speed and self.control.brake == 0:
			self.control.throttle = 0.0

		controls_sent = False
		while not controls_sent:
			try:
				self.game.send_control(self.control)
				controls_sent = True
				# print("controls sent!")
			except:
				if self.kill_when_connection_lost:
					raise ConnectionAbortedError(
						"Connection to server lost while sending controls. Reconnecting...")

				self.close_client_and_server()
				self.setup_client_and_server(reconnect_client_only=False)
				self.done = True

	def _restart_environment_episode(self, force_environment_reset=True):

		if not force_environment_reset and not self.done and self.is_game_setup:
			print("Can't reset dude, episode ain't over yet")
			return None  # User should handle this

		self.is_game_ready_for_input = False
		if not self.is_game_setup:
			self.setup_client_and_server()
			if self.is_render_enabled:
				self.init_renderer()
		else:
			self.iterator_start_positions += 1
			if self.iterator_start_positions >= self.num_pos:
				self.iterator_start_positions = 0

		try:
			self.game.start_episode(self.iterator_start_positions)
		except:
			self.game.connect()
			self.game.start_episode(self.iterator_start_positions)

		self.unmoved_steps = 0.0
		if self.early_termination_enabled:
			self.cur_neg_steps = 0

		self.car_speed = 0
		# start the game with some initial speed
		observation = None
		for i in range(self.num_speedup_steps):
			observation, reward, done, _ = self.step([0, 0.15])
		self.observation = observation
		self.is_game_ready_for_input = True

		return observation

	def init_renderer(self):

		self.num_cameras = 0
		if self.rgb_camera:
			self.num_cameras += 1
		self.renderer.create_screen(
			carla_config.render_width, carla_config.render_height*self.num_cameras)

	def get_rendered_image(self):

		temp = []
		if self.rgb_camera:
			temp.append(self.observation['rgb_image'])

		return np.vstack((img for img in temp))

	def save_screenshots(self):
		if not self.save_screens:
			print("save_screens is set False")
			return
		filename = str(int(time.time()*100))
		if self.rgb_camera:
			save_image(self.rgb_img_path+filename+".png",
						self.observation['rgb_image'])
