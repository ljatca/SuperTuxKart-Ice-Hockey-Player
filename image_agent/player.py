from os import path 
import numpy as np
import torch
from .planner import load_model
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from .utils import save_image
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def norm(vector):
    return np.linalg.norm(vector)

class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.model = load_model()
        self.num_players = None
        self.transform = TF.to_tensor
        self.actions = [{'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0},
                        {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}]

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        GOALS = np.float32([[0, 64.5], [0, -64.5]])

        for i, img in enumerate(player_image):
            action = self.actions[i]
            image = self.transform(img)
            pred = self.model(image).cpu().detach().numpy()

            puck_location_x = pred[0,0]
            puck_location_y = pred[0,1]

            
            player = player_state[i]

            kart_front = np.array(player['kart']['front'])[[0, 2]]
            kart_location = np.array(player['kart']['location'])[[0, 2]]

            dir = kart_front - kart_location
            dir = dir / norm(dir)

            # print("dir: ", dir)


            goal_dir = GOALS[self.team - 1] - kart_location
            # print("goal_dir: ", goal_dir)

            goal_dir = goal_dir / norm(goal_dir)

            goal_angle = np.arccos(np.clip(np.dot(dir, goal_dir), -1, 1))

            goal_dir = GOALS[self.team - 1] - kart_location
            goal_dist = norm(goal_dir)
            goal_dir = goal_dir / norm(goal_dir)

            goal_angle = np.arccos(np.clip(np.dot(dir, goal_dir), -1, 1))
            signed_goal_angle = np.degrees(
                -np.sign(np.cross(dir, goal_dir)) * goal_angle)

            goal_dist = (
                (np.clip(goal_dist, 10, 100) - 10) / 90) + 1
            
            MIN_ANGLE = 20
            MAX_ANGLE = 120

            puck_loc = pred[0][0]

            # print("puck_loc: ", puck_loc)
            if MIN_ANGLE < np.abs(signed_goal_angle) < MAX_ANGLE:
                distW = 1 / goal_dist ** 3
                aim_point = puck_loc + \
                    np.sign(puck_loc - signed_goal_angle /
                            100) * 0.3 * distW                
            else:
                aim_point = puck_loc
              
            
            kart_velocity = np.array(player['kart']['velocity'])

            kart_velocity = np.linalg.norm(kart_velocity)
            # print("kart_velocity: ", kart_velocity)
            
            target_velocity = 15

            # Use a proportional controller for steering
            desired_steer = aim_point * 2.0

            # Limit the steering to avoid extreme values
            desired_steer = max(-1, min(1, desired_steer))

            # Reduce speed if a sharp turn is detected
            if abs(desired_steer) > 0.8:
                target_velocity = 8

            velocity_difference = target_velocity - kart_velocity

            max_acceleration = 0.8 
            brake_threshold = 0.5
            drift_threshold = 0.3 

            # Use a proportional controller for acceleration
            if velocity_difference > 0:
                action["acceleration"] = min(1, max(velocity_difference / 10, max_acceleration))
                action["brake"] = False
            else:
                # Use brake if slowing down too fast
                if kart_velocity < brake_threshold * target_velocity:
                    action["acceleration"] = 0.0
                    action["brake"] = False
                else:
                    action["acceleration"] = 0.0
                    action["brake"] = False
            action["steer"] = desired_steer

            # Activate drift only when turning sharply
            if abs(desired_steer) > drift_threshold:
                action["drift"] = True
            else:
                action["drift"] = False
        
        # TODO: Change me. I'm just cruising straight
        return self.actions
        #return [dict(acceleration=0.5, steer=puck_locaton)] * self.num_players
