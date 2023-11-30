from os import path 
import numpy as np
import torch
from .planner import load_model
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from .utils import save_image
from collections import deque

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
        
        self.prev_puck_location = []
        self.past_states = [None, None]
        self.past_puck_locations = deque(maxlen=5)
        self.past_kart_locations = [deque(maxlen=5), deque(maxlen=5)]
        self.past_actions = [deque(maxlen=5), deque(maxlen=5)]
        self.stuck_count = 0
        
        self.goal = [(-10.4, -64.5), (-10.4, 64.5)]
        
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
    
    def is_stuck(self, kart_location, kart_velocity, past_kart_locations, past_actions):
        
        
        if len(past_kart_locations) < 5:
            return False
        
        MOVEMENT_VELOCITY_THRESHOLD = 0.02
        VELOCITY_THRESHOLD = 2.0
        
        no_movement = (abs(kart_location - past_kart_locations[-1]) < MOVEMENT_VELOCITY_THRESHOLD).all()
        no_velocity = kart_velocity < VELOCITY_THRESHOLD
        danger_zone = abs(kart_location[0]) >= 45 or abs(kart_location[1]) >= 63.5
        
        if no_movement and no_velocity and danger_zone:
            if self.stuck_count < 5:
                self.stuck_count +=1
                return False
            else:
                self.stuck_count = 0
                return True

    def stuck_action(self, kart_front, kart_location, action):
        
                
        if (abs(kart_location[0]) >= 45):
            if (action['acceleration'] > 0 ):
                action['steer'] = np.sign(kart_location[0]) * -1
            else:
                action['steer'] = np.sign(kart_location[0]) * 1
        else:
            if (self.prev_puck_location[-1][1] > kart_location[1]):

                if(kart_location[0] < 0):
                    action['steer'] = 1
                else:
                    action['steer'] = -1

            elif(self.prev_puck_location[-1][1] < kart_location[1]):
                if(kart_location[0] < 0):
                    action['steer'] = -1
                else:
                    action['steer'] = 1

        action['nitro'] = False
        return action
        

    def act(self, player_states, player_images):
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
        import time
        start_time = time.time()
        
        self.player_act(0, player_states[0], player_images[0])
        self.player_act(1, player_states[1], player_images[1])
        
        end_time = time.time()
        if end_time - start_time >= 0.05:
            print('Warning, the act function took more than 50 milliseconds')
        
        return self.actions
    
    def player_act(self, player_id, player_state, player_image):
        
        action = self.actions[player_id]
        past_kart_locations = self.past_kart_locations[player_id]
        past_actions = self.past_actions[player_id]
        past_state = self.past_states[player_id]

        t_image = self.transform(player_image)
        t_image = t_image.unsqueeze(0)
        puck_location = self.model(t_image).cpu().detach().numpy()

        kart_front = np.array(player_state['kart']['front'])[[0, 2]]
        kart_location = np.array(player_state['kart']['location'])[[0, 2]]
        past_kart_locations.append(kart_location)
        kart_velocity = np.array(player_state['kart']['velocity'])
        kart_velocity = np.linalg.norm(kart_velocity)
        
        target_velocity = 15

        # Use a proportional controller for steering
        desired_steer = puck_location[0][0] * 2.0

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
                action["brake"] = True
            else:
                action["acceleration"] = 0.0
                action["brake"] = False
        action["steer"] = desired_steer

        # Activate drift only when turning sharply
        if abs(desired_steer) > drift_threshold:
            action["drift"] = True
        else:
            action["drift"] = False
            
        if self.is_stuck(kart_location, kart_velocity, past_kart_locations, past_actions):
            action = self.stuck_action(kart_front, kart_location, action)
            
        self.prev_puck_location = puck_location
