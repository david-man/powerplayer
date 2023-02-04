import functools

from gymnasium.spaces import Dict
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.examples.policy.rock_paper_scissors_dummies import AlwaysSameHeuristic, BeatLastHeuristic
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
import sys
import gymnasium


sys.modules["gym"] = gymnasium
from stable_baselines3 import PPO
from torch import nn

from pettingzoo.utils.env import ParallelEnv
import math
import random
import numpy as np
import networkx as nx


from copy import copy
from itertools import islice

from gymnasium.spaces import Discrete, MultiDiscrete


def calculatePath(path, cur_orientation):
    last = path.pop(0)
    timeUsed = 0
    while (len(path) != 0):
        nextElement = path.pop(0)
        if (cur_orientation == 0 and nextElement == last + 1 or
                cur_orientation == 90 and nextElement == last - 6 or
                cur_orientation == 180 and nextElement == last - 1 or
                cur_orientation == 270 and nextElement == last + 6):
            timeUsed += 0.5
        else:
            timeUsed += 1  # .5 second for movement, one second for turning
            if (nextElement == last + 1):
                cur_orientation = 0
            elif (nextElement == last - 6):
                cur_orientation = 90
            elif (nextElement == last - 1):
                cur_orientation = 180
            elif (nextElement == last + 6):
                cur_orientation = 270

        last = nextElement

    return timeUsed


def coordsToBox(x, y):
    return math.floor(y) * 6 + math.floor(x)


def boxToCoords(box):
    box_y = math.floor(box / 6)
    box_x = box - box_y * 6
    return box_x, box_y

def junctionToBox(junctionNumber):
    if(junctionNumber == 0):
        return [0]
    elif(junctionNumber == 1):
        return [5]
    elif(junctionNumber == 2):
        return [30]
    elif(junctionNumber == 3):
        return [35]
    else:
        junctionNumber -= 4
        row = math.floor(junctionNumber/5)
        col = junctionNumber - 5*row
        return [coordsToBox(col, row), coordsToBox(col+1, row),
                coordsToBox(col, row+1), coordsToBox(col+1, row+1)]

def euclideanDistance(point_1, point_2):
    return math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1]-point_2[1])**2)



class Powerplayer(ParallelEnv):
    metadata = {"name": "powerplayer", "render_modes": ["time", "debug"]}
    def env(render_mode = None):
        env1 = Powerplayer(render_mode = render_mode)
        env1 = parallel_to_aec(env1)
        env1 = wrappers.AssertOutOfBoundsWrapper(env1)
        env1 = wrappers.OrderEnforcingWrapper(env1)

        return env1
    def __init__(self, render_mode = None):
        self.junctions = np.zeros((29,))# includes terminals, will include a picture of what junction is where later. self.junctions stores the current owner of the junction(0 = nobody, 1 = red, -1 = blue)
        #constnts about the locations of key areas
        self.terminals = [0,1,2,3]
        self.ground_junctions = [4, 6, 8, 14, 16, 18, 24, 26, 28]
        self.low_junctions = [5, 7, 9, 13, 19, 23, 25, 27]
        self.medium_junctions = [10, 12, 20, 22]
        self.high_junctions = [11, 15, 17, 21]
        self.red_cone_areas = [3, 17, 23, 33]
        self.blue_cone_areas = [2, 12, 18, 32]
        self.render_mode = render_mode
        self.order = None



        
        self.blue_points = None
        self.red_points = None
        self.blue_substation = None
        self.red_substation = None
        self.blue_stack_one = None
        self.blue_stack_two = None
        self.red_stack_one = None
        self.red_stack_two = None
        self.red_beacon_one = None
        self.blue_beacon_one = None
        self.red_beacon_two = None
        self.blue_beacon_two = None
        self.red_terminal_one = None
        self.blue_terminal_one = None
        self.red_terminal_two = None
        self.blue_terminal_two = None
        self.adjustmentPeriods = [None,None,None,None]#defines the amount of time the bot needs to adjust BEFORE placing. DOES NOT include placement, which takes 0.5 seconds by definition
        self.actions = {"red_1": None, "red_2": None, "blue_1": None, "blue_2": None}# -1: currently busy, actor isn't choosing anything,
                                            # 0: cone or beacon,
                                            # 1: junction choice
        self.action_spaces = {"red_1": Discrete(29), "red_2": Discrete(29), "blue_1": Discrete(29), "blue_2": Discrete(29)}
        self.action_masks = {"red_1": np.ones(29, dtype=np.int64), "red_2": np.ones(29, dtype=np.int64), "blue_1": np.ones(29, dtype=np.int64), "blue_2": np.ones(29, dtype=np.int64)}


        self.bots = {
            "red_1": {"x": None, "y": None, "heading": None, "holdingCone": False, "holdingBeacon": False, "path": None,
             "team": None, "gotBeacon": False, "junctionTo": None, "busy": None, "released": False,
             "adjustmentPeriod": self.adjustmentPeriods[0], "flagOne": False},
            "red_2": {"x":None, "y":None, "heading": None, "holdingCone": False, "holdingBeacon": False,"path":None,
             "team": None, "gotBeacon": False, "junctionTo": None, "busy": None, "released": False,
             "adjustmentPeriod": self.adjustmentPeriods[1], "flagOne": False},
            "blue_1": {"x": None, "y": None, "heading": None, "holdingCone": False, "holdingBeacon": False,"path": None,
             "team": None, "gotBeacon": False, "junctionTo" : None, "busy":None, "released": False,
             "adjustmentPeriod":self.adjustmentPeriods[2], "flagOne": False},
            "blue_2": {"x": None, "y": None, "heading": None, "holdingCone": False, "holdingBeacon": False,"path": None,
             "team" : None, "gotBeacon":False, "junctionTo": None, "busy":None, "released": False,
             "adjustmentPeriod":self.adjustmentPeriods[3]}, "flagOne": False}
        self.timeElapsed = 0


        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(35))
        for r in range(35):
            col, row = boxToCoords(r)
            #print(str(r) + ": " + str(col) + ", " + str(row))
            if(row<5):
                self.graph.add_edge(r, r+6)
            if(col<5):
                self.graph.add_edge(col, col+1)
        self.possible_agents = ["red_1", "red_2", "blue_1", "blue_2"]
        self.observation_spaces = {a : Dict({"observation": MultiDiscrete(
            [3] * 29 + [242, 11, 2, 30, 7, 6, 6, 4, 2, 2, 6, 6, 4, 2, 2, 6, 6, 4, 2, 2, 6, 6, 4, 2, 2, 40, 40, 25, 25,
                        25, 25, 21, 21, 6, 6, 6, 6, 2, 2, 2, 2], dtype=np.int64), "action_mask": MultiDiscrete([2] * 29, dtype=np.int64)})for a in self.possible_agents}


    def reset(self, seed = 0,  return_info = False, options = None):
        self.agents = copy(self.possible_agents)
        self.blue_points = 0
        self.red_points = 0
        self.blue_substation = 20
        self.red_substation = 20
        self.blue_stack_one = 5
        self.blue_stack_two = 5
        self.red_stack_one = 5
        self.red_stack_two = 5
        self.red_beacon_one = -1
        self.blue_beacon_one = -1
        self.red_beacon_two = -1
        self.blue_beacon_two = -1
        self.red_terminal_one = False
        self.blue_terminal_one = False
        self.red_terminal_two = False
        self.blue_terminal_two = False
        self.adjustmentPeriods = {"red_1": 2, "red_2":2, "blue_1": 2,"blue_2":2}# 2s for all bots to adjust BEFORE placing cone down
        self.actions = {"red_1": -1, "red_2":-1, "blue_1": -1,"blue_2":-1}
        self.order = ["red_1", "red_2", "blue_1", "blue_2"]
        self.rendered = False
        if(seed is not None):
            random.seed(seed)
            random.shuffle(self.order)
            for i in range(4):
                self.adjustmentPeriods[i] = random.randint(1,6)*0.5


        self.bots = {
            "red_1": {"x": 5.5, "y": 1.5, "heading": 180, "holdingCone": False, "holdingBeacon": False, "path": [],
             "team": "red", "gotBeacon": False, "junctionTo": None, "busy": 0, "released": False,
             "adjustmentPeriod": self.adjustmentPeriods["red_1"], "flagOne": True},
            "red_2": {"x": 5.5, "y": 4.5, "heading": 180, "holdingCone": False, "holdingBeacon": False,"path": [],
             "team": "red", "gotBeacon": False, "junctionTo": None, "busy": 0, "released": False,
             "adjustmentPeriod": self.adjustmentPeriods["red_2"], "flagOne": True},
            "blue_1": {"x": 0.5, "y": 1.5, "heading": 180, "holdingCone": False, "holdingBeacon": False,"path": [],
             "team": "blue", "gotBeacon": False, "junctionTo" : None, "busy":0, "released": False,
             "adjustmentPeriod":self.adjustmentPeriods["blue_1"], "flagOne": True},
            "blue_2": {"x": 0.5, "y": 4.5, "heading": 180, "holdingCone": False, "holdingBeacon": False,"path": [],
             "team" : "blue", "gotBeacon":False, "junctionTo": None, "busy":0, "released": False,
             "adjustmentPeriod":self.adjustmentPeriods["blue_2"], "flagOne":True}}

        for a in self.agents:
            self.set_action_space_size(a, 1)

        self.timeElapsed = 0

        observations = {i: self.generate_observation(i) for i in self.agents}

        return observations




    def collisionCheck(self, referal, returnBots  =False):
        ref_name = referal#this function modifies the bot by adding a timer to the "busy" part, checked ONLY by simulateMovement
        referal = self.bots[referal]
        bots = {"red_1": -1, "red_2": -1, "blue_1": -1, "blue_2": -1}
        for bot in self.bots:
            if(referal["released"] or self.bots[bot]["released"]):
                continue
            if(bot == ref_name):
                continue
            elif(len(referal["path"]) == 0):
                if(not returnBots):return False
                else: return {}
            else:
                #three types of collisions are noted here: head-on, hitting stationary, hitting side
                #head-on/side-on
                if((len(referal["path"]) > 0 and len(self.bots[bot]["path"]) > 0) and
                        (referal["path"][0] == coordsToBox(self.bots[bot]["x"], self.bots[bot]["y"]))
                        and (self.bots[bot]["path"][0] == coordsToBox(referal["x"], referal["y"]))):
                    toSet = 0
                    if(abs(referal["heading"] - self.bots[bot]["heading"]) == 180):#designates a head-on:
                        referal["busy"] = 1
                        toSet = 1
                        #print("Head On Collision")
                    else:#side-on
                        referal["busy"] = max(0.5, referal["busy"])
                        toSet = 0.5
                    if(not returnBots): return True
                    else: bots[bot] = toSet
                else:

                    otherBotWillStay = (len(self.bots[bot]["path"]) == 0 or
                                        self.order.index(bot) < self.order.index(ref_name))
                    if(not otherBotWillStay):
                        nextBox = self.bots[bot]["path"][0]
                        curBox = coordsToBox(self.bots[bot]["x"], self.bots[bot]["y"])
                        if(nextBox == curBox+1):
                            if(not (self.bots[bot]["heading"] == 0)):
                                otherBotWillStay = True
                        elif(nextBox == curBox-6):
                            if(not (self.bots[bot]["heading"] == 90)):
                                otherBotWillStay = True
                        elif(nextBox == curBox-1):
                            if(not (self.bots[bot]["heading"] == 180)):
                                otherBotWillStay = True
                        elif(nextBox == curBox+6):
                            if(not (self.bots[bot]["heading"] == 270)):
                                otherBotWillStay = True


                    if(otherBotWillStay and referal["path"][0] == coordsToBox(self.bots[bot]["x"], self.bots[bot]["y"])):
                        referal["busy"] = max(1, referal["busy"])
                        bots[bot] = 0
                        if(not returnBots): return True

        if(not returnBots):return False
        else: return bots


    def shortestPath(self, box_f, box_t, pathNum, cur_orientation):

        t = list(islice(nx.shortest_simple_paths(self.graph, int(box_f), int(box_t)),
                        pathNum))  # djikstra implementation but i was too lazy to actually do it
        # print(t)
        bestPath = []
        bestTime = 1000000
        for path in t:
            p1 = path.copy()
            time = calculatePath(path, cur_orientation)
            if (time < bestTime):
                bestPath = p1
                bestTime = time
        return bestPath[1:]
    
    def step(self, actions):
        collided = {"red_1": False, "red_2": False, "blue_1": False, "blue_2": False}
        for bot in self.bots:
            collided[bot] = self.collisionCheck(bot)

        rewards = {a : 0 for a in self.agents}
        terminations = {a : False for a in self.agents}
        for r in self.order:
            if(collided[r]):
                self.actions[r] = -1
                self.set_action_space_size(r, 1)
                self.bots[r]["released"] = True
                continue
            if(self.actions[r] == 1):# says if it just chose a junction
                if(self.bots[r]["holdingBeacon"]): actions[r]+=4

                toCont = True
                if((self.bots[r]["junctionTo"] is not None) and len(self.bots[r]["path"]) == 0):# if the bot is there; the 0.5s that is spent here acts as a confirmation

                    if (self.bots[r]["junctionTo"] > 0):
                        self.bots[r]["junctionTo"] *= -1
                        self.bots[r]["busy"] = self.bots[r]["adjustmentPeriod"] - 0.5
                        self.actions[r] = -1
                        self.set_action_space_size(r, 1)

                    else:
                        reward = self.simulatePlacement(r, abs(self.bots[r]["junctionTo"]))
                        self.set_action_space_size(r, 1)
                        self.actions[r] = -1
                        rewards[r] += reward
                    continue

                elif(self.bots[r]["junctionTo"] is None or not(self.bots[r]["junctionTo"] == actions[r])):
                    # this means if it just chose a path different to what it originally was going for
                    junction_boxes = junctionToBox(actions[r])
                    junction_box = -1
                    junction_distance = 100000
                    #print(str(actions[r]) + ": " + str(junction_boxes))
                    toCont = True
                    for j in junction_boxes:
                        dist = euclideanDistance([self.bots[r]["x"], self.bots[r]["y"]], boxToCoords(j))
                        if (dist < junction_distance):
                            junction_box = j
                            junction_distance = dist
                    self.bots[r]["junctionTo"] = actions[r]
                    self.bots[r]["released"] = False

                    self.bots[r]["path"] = self.shortestPath(coordsToBox(self.bots[r]["x"], self.bots[r]["y"]),
                                                             junction_box, 5,
                                                             self.bots[r]["heading"])
                    if(len(self.bots[r]["path"]) == 0):# just need to handle beginning of placement, not completion
                        self.bots[r]["junctionTo"] *= -1
                        self.bots[r]["busy"] = self.bots[r]["adjustmentPeriod"] - 0.5
                        self.actions[r] = -1
                        self.set_action_space_size(r, 1)
                        continue
                    else:
                        x = self.collisionCheck(r, True)
                        for i in x:
                            if(not x[i] == -1 and not self.bots[i]["released"]):
                                self.actions[r] = -1
                                self.set_action_space_size(r, 1)
                                self.bots[r]["released"] = True
                                toCont = False
                                if(not x[i] == 0):
                                    self.actions[i] = -1
                                    self.set_action_space_size(r, 1)
                                    self.bots[i]["released"] = True
                                    self.bots[i]["busy"] = x[i]

                if(toCont):
                    self.simulateMovement(r)# continue on the path if it's not there yet
                    self.actions[r] = 1
                    self.set_action_space_size(r, 29) if not self.bots[r]["holdingBeacon"] else self.set_action_space_size(r, 25)
            elif(self.actions[r] == 0):# was it just supposed to choose a beacon/cone
                box = coordsToBox(self.bots[r]["x"], self.bots[r]["y"])  # take one cone away from wherever
                if (box == 3):
                    self.red_stack_one -= 1
                elif (box == 33):
                    self.red_stack_two -= 1
                elif (box == 17 or box == 23):
                    self.red_substation -= 1
                elif (box == 2):
                    self.blue_stack_one -= 1
                elif (box == 32):
                    self.blue_stack_two -= 1
                elif (box == 12 or box == 18):
                    self.blue_substation -= 1

                if(actions[r] == 1):
                    self.bots[r]["holdingBeacon"] = True
                    rewards[r]+=1
                    self.set_action_space_size(r, 25)
                    self.actions[r] = 1
                else:
                    self.bots[r]["holdingCone"] = True
                    self.set_action_space_size(r, 29)
                    self.actions[r] = 1# need to go to a junction next
            else: #was busy doing something else
                if(self.bots[r]["busy"] <= 0.5 and (self.bots[r]["junctionTo"] is not None and self.bots[r]["junctionTo"] < 0)):
                    # if adjustperiod = 0.5s, it will be artificially decreased to 0s due to code structure.
                    # this means that once it gets here, it will spend the 0.5s turn to tell itself to place on the next turn, which is why it needs to be 0s
                    self.set_action_space_size(r, 1)
                    self.actions[r] = 1
                    self.bots[r]["busy"] = 0
                elif(self.bots[r]["busy"] <= 0.5 and (self.bots[r]["holdingCone"] or self.bots[r]["holdingBeacon"])):
                    self.set_action_space_size(r, 29) if self.bots[r]["holdingCone"] else self.set_action_space_size(r, 25)
                    self.actions[r] = 1
                    self.bots[r]["busy"] = 0
                elif(self.bots[r]["busy"] > 0):
                    self.bots[r]["busy"] -= 0.5
                    self.set_action_space_size(r, 1)
                    self.actions[r] = -1
                else:
                    # if the length of the path is 0, must be either have just finished putting down a cone or have just arrived to pick up a cone
                    if(len(self.bots[r]["path"]) == 0):
                        if(self.bots[r]["flagOne"]):# flag indicates that they just placed down a cone
                            box_to_go_to = -1
                            box_to_go_to_dist = 100000
                            shutdown = False
                            self.bots[r]["flagOne"] = False
                            if (self.bots[r]["team"] == "red"):
                                for area in self.red_cone_areas:
                                    if (area == 3 and self.red_stack_one == 0):
                                        continue
                                    elif (area == 33 and self.red_stack_two == 0):
                                        continue
                                    elif ((area == 17 or area == 23) and self.red_substation == 0):

                                        shutdown = True
                                        continue
                                    else:
                                        area_row, area_col = boxToCoords(area)
                                        dist = euclideanDistance([area_row, area_col],
                                                                      [self.bots[r]["x"], self.bots[r]["y"]])
                                        if (dist < box_to_go_to_dist):
                                            box_to_go_to = area
                                            box_to_go_to_dist = dist
                            else:
                                for area in self.blue_cone_areas:
                                    if (area == 2 and self.blue_stack_one == 0):
                                        continue
                                    elif (area == 32 and self.blue_stack_two == 0):
                                        continue
                                    elif ((area == 12 or area == 18) and self.blue_substation == 0):

                                        shutdown = True
                                        continue
                                    else:
                                        area_row, area_col = boxToCoords(area)
                                        dist = euclideanDistance([area_row, area_col],
                                                                      [self.bots[r]["x"], self.bots[r]["y"]])
                                        if (dist < box_to_go_to_dist):
                                            box_to_go_to = area
                                            box_to_go_to_dist = dist
                            if (not shutdown):
                                self.bots[r]["path"] = self.shortestPath(
                                    coordsToBox(self.bots[r]["x"], self.bots[r]["y"]), box_to_go_to, 5,
                                    self.bots[r]["heading"])

                            else:
                                self.bots[r]["busy"] = 10000# kills it at this point - aint no way it getting here but just a precaution
                            self.actions[r] = -1
                            self.set_action_space_size(r, 1)
                        else:
                            #deal with picking up cone
                            if(self.timeElapsed >= 90 and (not self.bots[r]["gotBeacon"])):
                                self.actions[r] = 0
                                self.set_action_space_size(r, 2)
                            else:
                                self.bots[r]["holdingCone"] = True
                                self.set_action_space_size(r, 29)
                                self.actions[r] = 1  # need to go to a junction next

                    else:
                        if(self.bots[r]["holdingCone"] == True):
                            self.actions[r] = 1
                            self.set_action_space_size(r, 29)
                        elif(self.bots[r]["holdingBeacon"] == True):
                            self.actions[r] = 1
                            self.set_action_space_size(r, 25)
                        else:
                            self.actions[r] = -1
                            self.set_action_space_size(r, 1)

                            self.simulateMovement(r)


        truncations = {a : False for a in self.agents}

        infos = {a : {} for a in self.agents}#no idea what this does tbh
        observations = {a : self.generate_observation(a) for a in self.agents}
        if(self.timeElapsed >= 120):
            red_p, blue_p, red_c, blue_c = self.calcFinalPoints()
            rewards["red_1"] += red_p*3/5
            rewards["red_2"] += red_p*3/5
            rewards["blue_1"] += blue_p*3/5
            rewards["blue_2"] += blue_p*3/5

            if(self.bots["red_1"]["adjustmentPeriod"] + self.bots["red_2"]["adjustmentPeriod"]
                    < self.bots["blue_1"]["adjustmentPeriod"] + self.bots["blue_2"]["adjustmentPeriod"]):#deals with cycle times and circuiting
                if(red_p < blue_p):#lower cycle time = expected to win. if they don't, they get heavily penalized
                    rewards["red_1"] -= blue_p/5
                    rewards["red_2"] -= blue_p/5

                else:
                    rewards["red_1"] -= blue_p/20#if they get a higher score with higher cycle time, not as big of penalties
                    rewards["red_2"] -= blue_p/20
                rewards["blue_1"] -= red_p/20 # blue still has to get better tho, even if they got unlucky w/ random cycle times
                rewards["blue_2"] -= red_p/20
                if(not red_c):# lower cycle time = expectation to circuit
                    rewards["red_1"] -= 13
                    rewards["red_2"] -= 13
                    rewards["blue_1"] += 10
                    rewards["blue_2"] += 10 # reward for defense
                else:
                    rewards["red_1"] += 5
                    rewards["red_2"] += 5
                    rewards["blue_1"] -= 8
                    rewards["blue_2"] -= 8 # penalty for not defending, higher than 5 to ensure that the bot defends
                if(blue_c):# lower cycle time = expectation to defend
                    rewards["red_1"] -= 7
                    rewards["red_2"] -= 7
                    rewards["blue_1"] += 15
                    rewards["blue_2"] += 15
                else:
                    rewards["red_1"] += 8 # reward for defense is still high, even w/ low cycle to ensure bot defends
                    rewards["red_2"] += 8
                    rewards["blue_1"] -= 7
                    rewards["blue_2"] -= 7
            else:
                if(blue_p < red_p):
                    rewards["blue_1"] -= red_p/5
                    rewards["blue_2"] -= red_p / 5
                else:
                    rewards["blue_1"] -= red_p/20
                    rewards["blue_2"] -= red_p/20
                rewards["red_1"] -= blue_p/20
                rewards["red_2"] -= blue_p/20
                if (not blue_c):  # lower cycle time = expectation to circuit
                    rewards["blue_1"] -= 15
                    rewards["blue_2"] -= 15
                    rewards["red_1"] += 10
                    rewards["red_2"] += 10
                else:
                    rewards["blue_1"] += 5
                    rewards["blue_2"] += 5
                    rewards["red_1"] -= 5
                    rewards["red_2"] -= 5
                if (red_c):  # lower cycle time = expectation to defend
                    rewards["blue_1"] -= 7
                    rewards["blue_2"] -= 7
                    rewards["red_1"] += 15 # high reward b/c they somehow circuited w/ higher cycle time
                    rewards["red_2"] += 15
                else:
                    rewards["blue_1"] += 7# reward for defense
                    rewards["blue_2"] += 7
                    rewards["red_1"] -= 5
                    rewards["red_2"] -= 5
            truncations = {a : True for a in self.agents}
            terminations = {a : True for a in self.agents}
            self.agents = []


        self.timeElapsed += 0.5

        #print(self.agents)
        return observations, rewards, terminations, truncations, infos





    def circuit(self, team):
        if(team == "red"):
            if((not (self.red_terminal_one and self.red_terminal_two))):
                return False
            else:# simple iterative BFS
                stack = [4,5,9]
                searched = []
                while(len(stack)>0):
                    val = stack.pop(0)
                    if(val in searched):
                        continue
                    else:
                        searched.append(val)
                    if(val < 4 or val >= 29):
                        continue
                    if(self.junctions[val] == 1):
                        if(val in [23,27,28]):
                            return True
                        else:
                            val_abs = val-4#makes it easier to work with since val will be 4-29
                            if(not(val_abs % 5 == 0)):#means its not on left edge
                                stack.append(val-1)#one left
                            if(not(val_abs % 5 == 4)): # not on right edge
                                stack.append(val+1)#one right
                            if(not(val_abs + 5 >= 25)): #not on bottom edge
                                stack.append(val+5)# one down
                            if(not(val_abs - 5 < 0)): # not on top edge
                                stack.append(val-5)# one up
                            if(not(val_abs %5 == 0) and not(val_abs - 5 < 0)):
                                stack.append(val-6)#diag upper left
                            if(not(val_abs % 5 == 4) and not(val_abs - 5 < 0)):
                                stack.append(val-4)#diag upper right
                            if(not(val_abs % 5 ==0) and not (val_abs+5 >= 25)):
                                stack.append(val+4)#diag lower left
                            if(not(val_abs%5 == 4) and not(val_abs+5>=25)):
                                stack.append(val+6)#diag lower right
                return False
        else:
            if ((not (self.blue_terminal_one and self.blue_terminal_two))):
                return False
            else:
                stack = [7, 8, 13]
                searched = []
                while (len(stack) > 0):
                    val = stack.pop(0)
                    if (val in searched):
                        continue
                    else:
                        searched.append(val)

                    if (val < 4 or val >= 29):
                        continue
                    if (self.junctions[val] == -1):
                        if (val in [19, 24, 25]):
                            return True
                        else:
                            val_abs = val - 4  # makes it easier to work with since val will be 4-29
                            if (not (val_abs % 5 == 0)):  # means its not on left edge
                                stack.append(val - 1)  # one left
                            if (not (val_abs % 5 == 4)):  # not on right edge
                                stack.append(val + 1)  # one right
                            if (not (val_abs + 5 >= 25)):  # not on bottom edge
                                stack.append(val + 5)  # one down
                            if (not (val_abs - 5 < 0)):  # not on top edge
                                stack.append(val - 5)  # one up
                            if (not (val_abs % 5 == 0) and not (val_abs - 5 < 0)):
                                stack.append(val - 6)  # diag upper left
                            if (not (val_abs % 5 == 4) and not (val_abs - 5 < 0)):
                                stack.append(val - 4)  # diag upper right
                            if (not (val_abs % 5 == 0) and not (val_abs + 5 >= 25)):
                                stack.append(val + 4)  # diag lower left
                            if (not (val_abs % 5 == 4) and not (val_abs + 5 >= 25)):
                                stack.append(val + 6)  # diag lower right
                return False


    def calcFinalPoints(self):
        red_points = self.red_points
        blue_points = self.blue_points
        red_circuit = self.circuit("red")
        blue_circuit = self.circuit("blue")
        for i in range(4, 29):#ownership calcs
            if(self.junctions[i] == 1):
                red_points += 3
            elif(self.junctions[i] == -1):
                blue_points+=3
        return red_points, blue_points, red_circuit, blue_circuit

    def simulateMovement(self, robotNum):#apply a movement on a robot given a path
        referal = self.bots[robotNum]

        box = coordsToBox(referal["x"], referal["y"])
        if(len(referal["path"]) == 0):
            return True
        else:
            nextElement = referal["path"][0]
            if nextElement == box + 1:
                if referal["heading"]==0:
                    referal["x"] += 1
                    del referal["path"][0]
                else: referal["heading"] = 0
            if nextElement == box - 6:
                if referal["heading"] == 90:
                    referal["y"]-=1
                    del referal["path"][0]
                else: referal["heading"] = 90
            if nextElement == box - 1:
                if referal["heading"] == 180:
                    referal["x"]-=1
                    del referal["path"][0]
                else: referal["heading"] = 180
            if nextElement == box + 6:

                if referal["heading"] == 270:
                    referal["y"]+=1
                    del referal["path"][0]
                else:
                    referal["heading"] = 270


    def simulatePlacement(self, robotNum, junctionAt):#ONLY deals with points and ownership
        referal = self.bots[robotNum]
        rewards = 0
        if(junctionAt > 3 and ((self.junctions[junctionAt] == self.red_beacon_one or self.junctions[junctionAt] == self.red_beacon_two) or
                               (self.junctions[junctionAt] == self.blue_beacon_one or self.junctions[junctionAt] == self.blue_beacon_two))):
            return 0#nothing counts after beacon is placed for either team
        elif(junctionAt > 3):#deals with changing of ownership
            if(referal["team"] == "red" and self.junctions[junctionAt] == -1):
                rewards+=2# reward for taking ownership away
            elif(referal["team"] == "blue" and self.junctions[junctionAt]==1):
                rewards+=2
            self.junctions[junctionAt] = 1 if referal["team"] == "red" else -1
        points = 0 if not referal["holdingBeacon"] else 10#assumption that teams will place cone+beacon at the same time

        if(referal["holdingBeacon"]):
            if(referal["team"] == "red"):
                if(self.red_beacon_one == -1): self.red_beacon_one = junctionAt
                else: self.red_beacon_two = junctionAt
            else:
                if (self.blue_beacon_one == -1):
                    self.blue_beacon_one = junctionAt
                else:
                    self.blue_beacon_two = junctionAt
            referal["gotBeacon"] = True
        if(junctionAt < 4):
            points += 1 if referal["team"] == "red" and (junctionAt == 0 or junctionAt == 35) else 0
            points += 1 if referal["team"] == "blue" and (junctionAt==5 or junctionAt==30) else 0
            if((junctionAt == 5 or junctionAt==30) and referal["team"] == "red"):
                rewards -= 10#make them not go to this junction - this part of the code is a bandage on a really bad bug, but i'm too tired to fix it within the first iteration
            elif((junctionAt == 0 or junctionAt == 35) and referal["team"] == "blue"):
                rewards -= 10
            if((not self.red_terminal_one) and referal["team"] == "red" and junctionAt == 0):
                self.red_terminal_one = True
                rewards+=3
            elif((not self.red_terminal_two) and referal["team"] == "red" and junctionAt == 35):
                self.red_terminal_two = True
                rewards += 3
            elif((not self.blue_terminal_one) and referal["team"] == "blue" and junctionAt == 5):
                self.blue_terminal_one = True
                rewards += 3
            elif ((not self.blue_terminal_two) and referal["team"] == "blue" and junctionAt == 30):
                self.blue_terminal_two = True
                rewards += 3
        elif(junctionAt in self.ground_junctions):
            points += 2
        elif(junctionAt in self.low_junctions):
            points += 3
        elif(junctionAt in self.medium_junctions):
            points += 4
        else:
            points += 5
        if(referal["team"] == "red"):
            self.red_points+=points
        else:
            self.blue_points+=points
        referal["released"] = False
        referal["flagOne"] = True
        referal["junctionTo"] = None
        referal["holdingCone"] = False
        referal["holdingBeacon"] = False
        return points+rewards
    def generate_observation(self, botName):#normalization - a lot of it
        #junction ownerships are normalized to 0,1,2 rather than -1,0,1 for the discrete module
        #lots of casting to ensure discrete module gets the right idea
        current_obs = [a + 1 for a in self.junctions] + [int(self.timeElapsed*2/10),
                len(self.bots[botName]["path"]),
                int(self.bots[botName] == "red"),
                int(abs(self.bots[botName]["junctionTo"]+1) if self.bots[botName]["junctionTo"] is not None else 0),
                int(self.bots["red_1"]["x"]),
                int(self.bots["red_1"]["y"]),
                int(self.bots["red_1"]["heading"]/90.0),
                int(self.bots["red_1"]["holdingCone"]),
                int(self.bots["red_1"]["holdingBeacon"]),
                int(self.bots["red_1"]["adjustmentPeriod"] * 2),
                int(self.bots["red_2"]["x"]),
                int(self.bots["red_2"]["y"]),
                int(self.bots["red_2"]["heading"]/90.0),
                int(self.bots["red_2"]["holdingCone"]),
                int(self.bots["red_2"]["holdingBeacon"]),
                int(self.bots["red_2"]["adjustmentPeriod"] * 2),
                int(self.bots["blue_1"]["x"]),
                int(self.bots["blue_1"]["y"]),
                int(self.bots["blue_1"]["heading"]/90.0),
                int(self.bots["blue_1"]["holdingCone"]),
                int(self.bots["blue_1"]["holdingBeacon"]),
                int(self.bots["blue_1"]["adjustmentPeriod"] * 2),
                int(self.bots["blue_2"]["x"]),
                int(self.bots["blue_2"]["y"]),
                int(self.bots["blue_2"]["heading"]/90.0),
                int(self.bots["blue_2"]["holdingCone"]),
                int(self.bots["blue_2"]["holdingBeacon"]),
                int(self.bots["blue_2"]["adjustmentPeriod"] * 2),
                int(self.blue_points/10),#didn't do no math to calc these weights but whatever lol
                int(self.red_points/10),
                int(self.red_beacon_one),
                int(self.red_beacon_two),
                int(self.blue_beacon_one),
                int(self.blue_beacon_two),
                int(self.blue_substation),
                int(self.red_substation),
                int(self.blue_stack_one),
                int(self.blue_stack_two),
                int(self.red_stack_one),
                int(self.red_stack_two),
                int(self.red_terminal_one),
                int(self.blue_terminal_one),
                int(self.red_terminal_two),
                int(self.blue_terminal_two)
                ]
        return {"observation": tuple(current_obs), "action_mask": self.action_masks[botName]}

    def render(self):

        print("CUR TIME:" +  str(self.timeElapsed))
        if (self.render_mode == "debug"):
            print("ACTIONS FOR FOLLOWING STATE: " + str(self.actions))
            print("STATE OF RED_1 FOR GRAPH BELOW: " + str(self.bots["red_1"]))
            print("STATE OF RED_2 FOR GRAPH BELOW: " + str(self.bots["red_2"]))
            spacesNeededLeft = 2 if self.blue_substation >= 10 else 1
            spacesNeededRight = 2 if self.red_substation >= 10 else 1
            initial = (str(int(self.junctions[0])) + spacesNeededLeft * " " + "   " + str(self.blue_stack_one) + " " + str(
                self.red_stack_one) + spacesNeededRight * " " + "   " + str(int(self.junctions[1])) + "\n" +
                       spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                       spacesNeededLeft * " " + " G L G L G " + spacesNeededRight * " " + "\n" +
                       spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                       spacesNeededLeft * " " + " L M H M L " + spacesNeededRight * " " + "\n" +
                       spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                       str(self.blue_substation) + " G H G H G " + str(self.red_substation) + "\n" +
                       spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                       spacesNeededLeft * " " + " L M H M L " + spacesNeededRight * " " + "\n" +
                       spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                       spacesNeededLeft * " " + " G L G L G " + spacesNeededRight * " " + "\n" +
                       spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                       str(int(self.junctions[2])) + spacesNeededLeft * " " + "   " + str(self.blue_stack_two) + " " + str(
                        self.red_stack_two) + spacesNeededRight * " " + "   " + str(int(self.junctions[3])))

            for bot in self.bots:
                box = coordsToBox(self.bots[bot]["x"], self.bots[bot]["y"])
                box_x, box_y = boxToCoords(box)
                x = box_x * 2 + spacesNeededLeft
                y = box_y * 2 + 1
                ind = y * (12 + spacesNeededRight + spacesNeededLeft) + x
                toPut = "R" if self.bots[bot]["team"] == "red" else "B"
                initial = initial[:ind] + toPut + initial[ind + 1:]


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    #@functools.lru_cache(maxsize=None)
    def action_space(self, agent):

        return self.action_spaces[agent]

    def set_action_space_size(self, agent, size):#sets the "size" by masking
        for i in range(size, 29):
            self.action_masks[agent][i] = 0




from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    parallel_api_test(Powerplayer(), num_cycles=1_000_000)







# import gym
# import torch
# from tianshou.data import Collector, VectorReplayBuffer
# from tianshou.env import DummyVectorEnv
# from tianshou.env.pettingzoo_env import PettingZooEnv
# from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy, PPOPolicy
# from tianshou.trainer import offpolicy_trainer
# from tianshou.utils.net.common import Net
# from typing import Optional, Tuple
#
# from copy import deepcopy
#
# from pettingzoo.utils import wrappers
#
# from tianshou.data import Collector
# from tianshou.env import DummyVectorEnv
# from tianshou.policy import RandomPolicy, MultiAgentPolicyManager
# from pettingzoo.utils.conversions import parallel_wrapper_fn, parallel_to_aec
#
# import argparse
#
#
# # agents should be wrapped into one policy,
# # which is responsible for calling the acting agent correctly
# # here we use two random agents
#
# def get_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=1626)
#     parser.add_argument('--eps-test', type=float, default=0.05)
#     parser.add_argument('--eps-train', type=float, default=0.1)
#     parser.add_argument('--buffer-size', type=int, default=20000)
#     parser.add_argument('--lr', type=float, default=1e-4)
#     parser.add_argument(
#         '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
#     )
#     parser.add_argument('--n-step', type=int, default=3)
#     parser.add_argument('--target-update-freq', type=int, default=320)
#     parser.add_argument('--epoch', type=int, default=50)
#     parser.add_argument('--step-per-epoch', type=int, default=1000)
#     parser.add_argument('--step-per-collect', type=int, default=10)
#     parser.add_argument('--update-per-step', type=float, default=0.1)
#     parser.add_argument('--batch-size', type=int, default=64)
#     parser.add_argument(
#         '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
#     )
#     parser.add_argument('--training-num', type=int, default=10)
#     parser.add_argument('--test-num', type=int, default=10)
#     parser.add_argument('--logdir', type=str, default='log')
#     parser.add_argument('--render', type=float, default=0.1)
#     parser.add_argument(
#         '--win-rate',
#         type=float,
#         default=0.6,
#         help='the expected winning rate: Optimal policy can get 0.7'
#     )
#     parser.add_argument(
#         '--watch',
#         default=False,
#         action='store_true',
#         help='no training, '
#              'watch the play of pre-trained models'
#     )
#     parser.add_argument(
#         '--agent-id',
#         type=int,
#         default=2,
#         help='the learned agent plays as the'
#              ' agent_id-th player. Choices are 1 and 2.'
#     )
#     parser.add_argument(
#         '--resume-path',
#         type=str,
#         default='',
#         help='the path of agent pth file '
#              'for resuming from a pre-trained agent'
#     )
#     parser.add_argument(
#         '--opponent-path',
#         type=str,
#         default='',
#         help='the path of opponent agent pth file '
#              'for resuming from a pre-trained agent'
#     )
#     return parser
#
# def get_args() -> argparse.Namespace:
#     parser = get_parser()
#     return parser.parse_known_args()[0]
# def get_env(render_mode=None):
#     return PettingZooEnv(Powerplayer.env(render_mode=render_mode))
# def get_agents(
#     args: argparse.Namespace = get_args(),
#     agent_learn: Optional[BasePolicy] = None,
#     agent_opponent: Optional[BasePolicy] = None,
#     optim: Optional[torch.optim.Optimizer] = None,
# ) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
#     env = get_env()
#     observation_space = env.observation_space['observation'] if isinstance(
#         env.observation_space, gym.spaces.Dict
#     ) else env.observation_space
#     args.state_shape = observation_space.shape
#     args.action_shape = env.action_space.shape
#     args.device = 'cuda'
#     if agent_learn is None:
#         # model
#         net = Net(
#             args.state_shape,
#             args.action_shape,
#             hidden_sizes=args.hidden_sizes,
#             device=args.device
#         ).to(args.device)
#         if optim is None:
#             optim = torch.optim.Adam(net.parameters(), lr=args.lr)
#         agent_learn = DQNPolicy(
#             net,
#             optim,
#             args.gamma,
#             args.n_step,
#             target_update_freq=args.target_update_freq
#         )
#         if args.resume_path:
#             agent_learn.load_state_dict(torch.load(args.resume_path))
#
#     if agent_opponent is None:
#         if args.opponent_path:
#             agent_opponent = deepcopy(agent_learn)
#             agent_opponent.load_state_dict(torch.load(args.opponent_path))
#         else:
#             agent_opponent = RandomPolicy()
#
#     if args.agent_id == 1:
#         agents = [agent_learn, agent_learn, agent_opponent, agent_opponent]
#     else:
#         agents = [agent_opponent, agent_opponent, agent_learn, agent_learn]
#     policy = MultiAgentPolicyManager(agents, env)
#     return policy, optim, env.agents
#
#
#
#
# import os
#
#
# def train_agent(
#     args: argparse.Namespace = get_args(),
#     agent_learn: Optional[BasePolicy] = None,
#     agent_opponent: Optional[BasePolicy] = None,
#     optim: Optional[torch.optim.Optimizer] = None,
# ) -> Tuple[dict, BasePolicy]:
#
#     # ======== environment setup =========
#     train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
#     test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
#     # seed
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     train_envs.seed(args.seed)
#     test_envs.seed(args.seed)
#
#     # ======== agent setup =========
#     policy, optim, agents = get_agents(
#         args, agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
#     )
#
#     # ======== collector setup =========
#     train_collector = Collector(
#         policy,
#         train_envs,
#         VectorReplayBuffer(args.buffer_size, len(train_envs)),
#         exploration_noise=True
#     )
#     test_collector = Collector(policy, test_envs, exploration_noise=True)
#     # policy.set_eps(1)
#     train_collector.collect(n_step=args.batch_size * args.training_num)
#
#     # ======== callback functions used during training =========
#     def save_best_fn(policy):
#         if hasattr(args, 'model_save_path'):
#             model_save_path = args.model_save_path
#         else:
#             model_save_path = os.path.join(
#                 args.logdir, 'tic_tac_toe', 'dqn', 'policy.pth'
#             )
#         torch.save(
#             policy.policies[agents[args.agent_id - 1]].state_dict(), model_save_path
#         )
#
#     def stop_fn(mean_rewards):
#         return mean_rewards >= args.win_rate
#
#     def train_fn(epoch, env_step):
#         policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)
#
#     def test_fn(epoch, env_step):
#         policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
#
#     def reward_metric(rews):
#         return rews[:, args.agent_id - 1]
#
#     # trainer
#     result = offpolicy_trainer(
#         policy,
#         train_collector,
#         test_collector,
#         args.epoch,
#         args.step_per_epoch,
#         args.step_per_collect,
#         args.test_num,
#         args.batch_size,
#         train_fn=train_fn,
#         test_fn=test_fn,
#         stop_fn=stop_fn,
#         save_best_fn=save_best_fn,
#         update_per_step=args.update_per_step,
#         test_in_train=False,
#         reward_metric=reward_metric
#     )
#
#     return result, policy.policies[agents[args.agent_id - 1]]
#
# # ======== a test function that tests a pre-trained agent ======
# def watch(
#     args: argparse.Namespace = get_args(),
#     agent_learn: Optional[BasePolicy] = None,
#     agent_opponent: Optional[BasePolicy] = None,
# ) -> None:
#     env = get_env(render_mode="human")
#     env = DummyVectorEnv([lambda: env])
#     policy, optim, agents = get_agents(
#         args, agent_learn=agent_learn, agent_opponent=agent_opponent
#     )
#     policy.eval()
#     policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
#     collector = Collector(policy, env, exploration_noise=True)
#     result = collector.collect(n_episode=1, render=args.render)
#     rews, lens = result["rews"], result["lens"]
#     print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")
#
# # train the agent and watch its performance in a match!
# args = get_args()
# result, agent = train_agent(args)
# watch(args, agent)



