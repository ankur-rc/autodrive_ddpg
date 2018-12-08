'''
Created Date: Saturday December 1st 2018
Last Modified: Saturday December 1st 2018 3:10:27 pm
Author: ankurrc
'''

import random

from carla.settings import CarlaSettings
from carla.sensor import Camera

from carla_rl import carla_config


def get_carla_settings(settings_file=None):

    if settings_file is None:

        # Create a CarlaSettings object. This object is a wrapper around
        # the CarlaSettings.ini file. Here we set the configuration we
        # want for the new episode.
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=False,
            NumberOfVehicles=0,
            NumberOfPedestrians=0,
            # 8-14 are sunset; we want easy first
            WeatherId=random.choice(range(0, 2)),
            QualityLevel='Low'
        )
        settings.randomize_seeds()

        # Now we want to add a couple of cameras to the player vehicle.
        # We will collect the images produced by these cameras every
        # frame.

        # The default camera captures RGB images of the scene.
        camera0 = Camera('CameraRGB')
        # Set image resolution in pixels.
        camera0.set_image_size(carla_config.render_width,
                               carla_config.render_height)
        # Set its position relative to the car in meters.
        camera0.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera0)

    else:

        # Alternatively, we can load these settings from a file.
        with open(settings_file, 'r') as fp:
            settings = fp.read()

    return settings
