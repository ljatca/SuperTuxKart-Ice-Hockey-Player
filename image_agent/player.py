from os import path 
import numpy as np
import torch
from .planner import load_model
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from .utils import save_image
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

        ret = []
        acceleration = 1

        for i, img in enumerate(player_image):
            plt.imsave(f'player_image_{i}.png', img)
            image = self.transform(player_image[i])
            pred = self.model(image).cpu().detach().numpy()

            puck_locaton = pred[0][0]
            print("puck", puck_locaton)
            save_image(image, pred)

            player= player_state[i]
            front = np.array(player['kart']['front'])[[0]]
            loc = np.array(player['kart']['location'])[[0]]
            velocity = np.array(player['kart']['velocity'])
            velocity = np.linalg.norm(velocity)
            print("velocity: ", velocity)
            if velocity >= 5:
                acceleration = 0
            else:
                acceleration = 1


            ret.append(dict(acceleration=acceleration, steer=puck_locaton))






        direction = (front - loc)
        print(direction)
        direction /= np.linalg.norm(direction)



        # TODO: Change me. I'm just cruising straight
        return ret
        #return [dict(acceleration=0.5, steer=puck_locaton)] * self.num_players
