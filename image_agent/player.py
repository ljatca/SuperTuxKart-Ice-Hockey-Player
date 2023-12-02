from os import path 
import numpy as np
import torch
from .planner import load_model
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from collections import deque

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# This fucntion is adapted from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

def get_direction_vector(p1, p2):
    direction_vector = p2 - p1
    return direction_vector / np.linalg.norm(direction_vector)

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
        self.actions = [{'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0},
                        {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}]
        
        self.prev_puck_location = []
        self.past_states = [None, None]
        self.past_puck_locations = deque(maxlen=5)
        self.past_kart_locations = [deque(maxlen=5), deque(maxlen=5)]
        self.past_actions = [deque(maxlen=5), deque(maxlen=5)]

        self.goal_stuck_count = [0, 0]
        self.wall_stuck_count = [0, 0]
        self.their_goal_left = (-10, -64)
        self.their_goal_center = (0, -64)
        self.their_goal_right = (10, -64)
        self.search_count = 0
        
        self.RED_goal_line = (0, 64.5)
        self.BLUE_goal_line = (0, -64.5)
        
        
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
    
    def stuck_in_goal(self, kart_loc):
        # print("stuck_in_goal")
        return abs(kart_loc[0]) < 10 and abs(kart_loc[1]) > 64
        
    def stuck_in_goal_action(self, action, kart_front, kart_location):
        # print("stuck_in_goal_action")
        
        blue_goal = kart_location[1] > 0

        if (blue_goal and kart_front[1] - kart_location[1] > -0.3) or (not blue_goal and kart_front[1] - kart_location[1] < 0.3):
            action['acceleration'] = 0
            action['brake'] = True
            action['steer'] = 1 if self.prev_puck_location[0] < kart_location[0] else -1

        else:
            action['acceleration'] = 1
            action['brake'] = False
            action['steer'] = -1 if (blue_goal and self.prev_puck_location[0] > kart_location[0]) or \
                                (not blue_goal and self.prev_puck_location[0] < kart_location[0]) else 1

        if abs(kart_location[1]) > 64:
            action['steer'] *= ((10 - abs(kart_location[0])) / 10)

        action['nitro'] = False
        return action

    
    def stuck_on_wall(self, kart_location, kart_velocity, past_kart_locations, past_actions):
        # print("stuck_on_wall")
        if len(past_kart_locations) < 5:
            return False
        
        MOVEMENT_VELOCITY_THRESHOLD = 0.02
        VELOCITY_THRESHOLD = 1.0
        
        no_movement = (abs(kart_location - past_kart_locations[-1]) < MOVEMENT_VELOCITY_THRESHOLD).all()
        no_velocity = kart_velocity < VELOCITY_THRESHOLD
        danger_zone = abs(kart_location[0]) >= 45 or abs(kart_location[1]) >= 63.5
        
        return no_movement and no_velocity and danger_zone
            
    def stuck_on_wall_action(self, kart_front, kart_location, action):
        
        # Got in the blue goal
        if (kart_location[1] < 0):
            
            # if kart is facing blue goal, backup; otherwise keep moving forward
            if (kart_front[1] - kart_location[1] < 0):
                action['acceleration'] = 0
                action['brake'] = True
            else:
                action['acceleration'] = 1
        # Got in the red goal
        else:
            # if kart is facing red goal, backup; otherwise keep moving forward
            if (kart_front[1] - kart_location[1] > 0):
                action['acceleration'] = 0
                action['brake'] = True
            else:
                action['acceleration'] = 1
                
        if (abs(kart_location[0]) >= 45):
            if (action['acceleration'] > 0 ):
                action['steer'] = np.sign(kart_location[0]) * -1
            else:
                action['steer'] = np.sign(kart_location[0]) * 1
        else:
            if (self.prev_puck_location[1] > kart_location[1]):

                if(kart_location[0] < 0):
                    action['steer'] = 1
                else:
                    action['steer'] = -1

            elif(self.prev_puck_location[1] < kart_location[1]):
                if(kart_location[0] < 0):
                    action['steer'] = -1
                else:
                    action['steer'] = 1
        # action['brake'] = True
        # action['acceleration'] = 0
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
        
        goal = self.RED_goal_line if self.team == 0 else self.BLUE_goal_line
        
        # Get game data
        action = self.actions[player_id]
        past_kart_locations = self.past_kart_locations[player_id]
        past_actions = self.past_actions[player_id]
        past_state = self.past_states[player_id]

        t_image = self.transform(player_image)
        t_image = t_image.unsqueeze(0)
        puck_location = (self.model(t_image).cpu().detach().numpy())[0]

        kart_front = np.array(player_state['kart']['front'])[[0, 2]]
        kart_location = np.array(player_state['kart']['location'])[[0, 2]]
        kart_velocity = np.array(player_state['kart']['velocity'])
        kart_velocity = np.linalg.norm(kart_velocity)

        
        puck_dirction = get_direction_vector(kart_location, puck_location)
        heading_dirction = get_direction_vector(kart_location, kart_front)
        goal_dirction = get_direction_vector(kart_location, goal)
        facing_angle = np.degrees(angle_between(heading_dirction, goal_dirction))
        
        vector_right = get_direction_vector(kart_location, self.their_goal_right)
        vector_center = get_direction_vector(kart_location, self.their_goal_center)
        vector_left = get_direction_vector(kart_location, self.their_goal_left)
        attack_cone = np.degrees(angle_between(vector_left, vector_right))
        
        x = puck_location[0]
        y = puck_location[1]
        
        # print(f"player: {player_id}, kart_front: {kart_front}, kart_location: {kart_location}")
        
        if self.goal_stuck_count[player_id] > 0:
            self.goal_stuck_count[player_id] -= 1
            self.stuck_in_goal_action(action, kart_front, kart_location)
            return

        if self.wall_stuck_count[player_id] > 0:
            self.wall_stuck_count[player_id] -= 1
            self.stuck_on_wall_action(kart_front, kart_location, action)
            return

        target_velocity = 15

        # Use a proportional controller for steering
        desired_steer = puck_location[0] * 2.0

        # Limit the steering to avoid extreme values
        desired_steer = max(-1, min(1, desired_steer))

        # Reduce speed if a sharp turn is detected
        if abs(desired_steer) > 0.8:
            target_velocity = 8

        velocity_difference = target_velocity - kart_velocity

        max_acceleration = 0.8 
        brake_threshold = 0.5
        drift_threshold = 0.3 

        if self.stuck_in_goal(kart_location):
            # print("inGoal")
            self.goal_stuck_count[player_id] = 5
            action = self.stuck_in_goal_action(action, kart_front, kart_location)
            
        elif self.stuck_on_wall(kart_location, kart_velocity, past_kart_locations, past_actions):
            # print("get stuck")
            self.wall_stuck_count[player_id] = 7
            action = self.stuck_on_wall_action(kart_front, kart_location, action)
        else:
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
              
          self.prev_puck_location = puck_location
          past_kart_locations.append(kart_location)
