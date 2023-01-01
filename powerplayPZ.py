import functools

from pettingzoo.utils.env import ParallelEnv
import math
import random
import numpy as np
import networkx as nx


from copy import copy
from itertools import islice

from gymnasium.spaces import Discrete




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
    def __init__(self):
        self.junctions = np.zeros((29,))# includes terminals, will include a picture of what junction is where later. self.junctions stores the current owner of the junction(0 = nobody, 1 = red, -1 = blue)
        #constnts about the locations of key areas
        self.terminals = [0,1,2,3]
        self.ground_junctions = [4, 6, 8, 14, 16, 18, 24, 26, 28]
        self.low_junctions = [5, 7, 9, 13, 19, 23, 25, 27]
        self.medium_junctions = [10, 12, 20, 22]
        self.high_junctions = [11, 15, 17, 21]
        self.red_cone_areas = [3, 17, 23, 33]
        self.blue_cone_areas = [2, 12, 18, 32]

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
        self.red_beaconed = None
        self.blue_beaconed = None
        self.red_terminal_one = None
        self.blue_terminal_one = None
        self.red_terminal_two = None
        self.blue_terminal_two = None
        self.adjustmentPeriods = [None,None,None,None]
        self.actions = {"red_1": None, "red_2": None, "blue_1": None, "blue_2": None}# -1: currently busy, actor isn't choosing anything,
                                            # 0: cone or beacon,
                                            # 1: junction choice
        self.action_spaces = {"red_1": Discrete(0), "red_2": Discrete(0), "blue_1": Discrete(0), "blue_2": Discrete(0)}

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
            if(row<5):
                self.graph.add_edge(r, r+6)
            if(col<5):
                self.graph.add_edge(col, col+1)
        self.possibleAgents = ["red_1", "red_2", "blue_1", "blue_2"]
        
    
    def reset(self, seed = None,  return_info = False, options = None):
        self.agents = copy(self.possibleAgents)
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
        self.adjustmentPeriods = [2,2,2,2]
        self.actions = [-1,-1,-1,-1]
        self.order = ["red_1", "red_2", "blue_1", "blue_2"]
        if(seed is not None):
            random.seed(seed)
            random.shuffle(self.order)
            for i in range(4):
                self.adjustmentPeriods[i] = random.randint(2,7)*0.5


        self.bots = {
            "red_1": {"x": 5.5, "y": 1.5, "heading": 180, "holdingCone": False, "holdingBeacon": False, "path": [],
             "team": "red", "gotBeacon": False, "junctionTo": None, "busy": 0, "released": False,
             "adjustmentPeriod": self.adjustmentPeriods[0], "flagOne": False},
            "red_2": {"x": 5.5, "y": 4.5, "heading": 180, "holdingCone": False, "holdingBeacon": False,"path": [],
             "team": "red", "gotBeacon": False, "junctionTo": None, "busy": 0, "released": False,
             "adjustmentPeriod": self.adjustmentPeriods[1], "flagOne": False},
            "blue_1": {"x": 0.5, "y": 1.5, "heading": 180, "holdingCone": False, "holdingBeacon": False,"path": [],
             "team": "blue", "gotBeacon": False, "junctionTo" : None, "busy":0, "released": False,
             "adjustmentPeriod":self.adjustmentPeriods[2], "flagOne": False},
            "blue_2": {"x": 0.5, "y": 4.5, "heading": 180, "holdingCone": False, "holdingBeacon": False,"path": [],
             "team" : "blue", "gotBeacon":False, "junctionTo": None, "busy":0, "released": False,
             "adjustmentPeriod":self.adjustmentPeriods[3], "flagOne":False}}

        self.action_spaces = {"red_1": Discrete(1), "red_2": Discrete(1), "blue_1": Discrete(1), "blue_2": Discrete(1)}

        self.timeElapsed = 0




    def collisionCheck(self, referal):#this function modifies the bot by adding a timer to the "busy" part, checked ONLY by simulateMovement
        toReturn = False
        for bot in self.bots:
            if(referal["released"] or self.bots[bot]["released"]):
                continue
            if(id(referal) == id(bot)):
                continue
            elif(len(referal["path"]) == 0):
                return False
            else:
                #three types of collisions are noted here: head-on, hitting stationary, hitting side
                #head-on/side-on
                if((len(referal["path"]) > 0 and len(self.bots[bot]["path"]) > 0) and
                        (referal["path"][0] == coordsToBox(self.bots[bot]["x"], self.bots[bot]["y"]))
                        and (self.bots[bot]["path"][0] == coordsToBox(referal["x"], referal["y"]))):
                    self.bots[bot]["released"] = True
                    referal["released"] = True
                    toReturn = True
                    if(abs(referal["heading"] - self.bots[bot]["heading"]) == 180):#designates a head-on:
                        self.bots[bot]["busy"] = 1
                        referal["busy"] = 1
                        #print("Head On Collision")
                    else:#side-on
                        referal["busy"] = max(0.5, referal["busy"])
                elif((len(referal["path"]) > 0 and len(self.bots[bot]["path"]) == 0) and
                     (referal["path"][0] == coordsToBox(self.bots[bot]["x"], self.bots[bot]["y"]))):#stationary hit
                    referal["busy"] = 1
                    self.bots[bot]["busy"] = 1
                    toReturn = True
                    referal["released"] = True
        return toReturn

    def shortestPath(self, box_f, box_t, pathNum, cur_orientation):
        t = list(islice(nx.shortest_simple_paths(self.graph, box_f, box_t),
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
        for bot in self.bots:
            self.collisionCheck(bot)

        rewards = {a : 0 for a in self.agents}
        terminations = {a : False for a in self.agents}
        for r in self.order:
            if(self.actions[r] == 1):# says if it just chose a junction
                if(self.bots[r]["holdingBeacon"]): actions[r]+=4
                junction_box = junctionToBox(actions[r])
                if(not(self.bots[r]["path"][-1] == junction_box)):# this means if it just chose a path different to what it originally was going for
                    self.bots[r]["path"] = self.shortestPath(coordsToBox(self.bots[r]["x"], self.bots[r]["y"]), junction_box,
                                                            5, self.bots[r]["heading"])
                reached = self.simulateMovement(r)
                if(reached):# bot is going to score
                    reward = self.simulatePlacement(r, self.bots[r]["junctionTo"])
                    self.action_spaces[r] = Discrete(1)
                    self.actions[r] = -1
                    rewards[r] += reward
                else:
                    self.actions[r] = 1
                    self.action_spaces[r] = Discrete(39) if not self.bots[r]["holdingBeacon"] else Discrete(35)
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
                    self.action_spaces[r] = Discrete(36)
                    self.actions[r] = 1
                else:
                    self.bots[r]["holdingCone"] = True
                    self.action_spaces[r] = Discrete(39)
                    self.actions[r] = 1# need to go to a junction next
            else: #was busy doing something else
                if(self.bots[r]["busy"] > 0):
                    self.bots[r]["busy"] -= 0.5
                    self.action_spaces[r] = Discrete(1)
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
                            self.action_spaces[r] = Discrete(1)
                        else:
                            #deal with picking up cone
                            if(self.timeElapsed >= 90 and (not self.bots[r]["gotBeacon"])):
                                self.actions[r] = 0
                                self.action_spaces[r] = Discrete(2)
                            else:
                                self.actions[r] = 1
                                self.bots[r]["holdingCone"] = True
                                self.action_spaces[r] = Discrete(39)
                                self.actions[r] = 1  # need to go to a junction next
                    else:
                        self.actions[r] = -1
                        self.action_spaces[r] = Discrete(1)

        truncations = {a : False for a in self.agents}
        if(self.timeElapsed):
            red_p, blue_p = self.calcFinalPoints()
            rewards["red_1"] += red_p/10
            rewards["red_2"] += red_p/10
            rewards["blue_1"] += blue_p/10
            rewards["blue_2"] += blue_p/10
            truncations = {a : True for a in self.agents}
            self.agents = []
        self.timeElapsed += 0.5

        infos = {a : {} for a in self.agents}#no idea what this does tbh
        observations = {a : {self.generate_observation(a)} for a in self.agents}
        return observations, rewards, terminations, truncations, infos





    def circuit(self, team):
        if(team == "red" and (not (self.red_terminal_one and self.red_terminal_two))):
            return False
        else:
            stack = [4,5,9]
            while(len(stack)>0):
                val = stack.pop(0)
                if(val < 4 or val >= 29):
                    continue
                if(self.junctions[val] == 1):
                    if(val in [23,27,28]):
                        return True
                    else:
                        val_abs = val-4#makes it easier to work with
                        if(not(val_abs % 5 == 0)):#means its not on left edge
                            stack.append(val-1)
                        if(not(val_abs % 5 == 4)): # not on right edge
                            stack.append(val+1)
                        if(not(val_abs + 5 >= 25)): #not on bottom edge
                            stack.append(val+5)
                        if(not(val_abs - 5 < 0)): # not on top edge
                            stack.append(val-5)
            return False


    def calcFinalPoints(self):
        red_points = self.red_points + int(self.circuit("red")) * 20
        blue_points = self.blue_points + int(self.circuit("blue"))*20
        for i in range(4, 29):#ownership calcs
            if(self.junctions[i] == 1):
                red_points += 3
            elif(self.junctions[i] == -1):
                blue_points+=3
        return red_points ,blue_points

    def simulateMovement(self, robotNum):#apply a movement on a robot given a path
        referal = self.bots[robotNum]
        box = coordsToBox(referal["x"], referal["y"])
        if(len(referal["path"]) == 0):
            return True
        else:
            nextElement = referal["path"][0]
            if nextElement == box + 1:
                if referal["heading"]==0:referal["x"] += 1; del referal["path"][0]
                else: referal["heading"] = 0
            if nextElement == box - 6:
                if referal["heading"] == 90: referal["y"]-=1; del referal["path"][0]
                else: referal["heading"] = 90
            if nextElement == box - 1:
                if referal["heading"] == 180: referal["x"]-=1; del referal["path"][0]
                else: referal["heading"] = 180
            if nextElement == box + 6:
                if referal["heading"] == 270: referal["y"]+=1; del referal["path"][0]
                else: referal["heading"] = 270

    def simulatePlacement(self, robotNum, junctionAt):#ONLY deals with points and ownership
        referal = self.bots[robotNum]
        rewards = 0
        if(junctionAt > 3 and ((self.junctions[junctionAt] in self.red_beaconed) or (self.junctions[junctionAt] in self.blue_beaconed))):
            return#nothing counts after beacon is placed for either team
        elif(junctionAt > 3):#deals with changing of ownership
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
        if(junctionAt < 4):
            points += 1
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
        return points+rewards
    def generate_observation(self, botName):
        return (
                len(self.bots[botName]["path"])/10,
                int(self.bots[botName] == "red"),
                self.bots[botName]["junctionTo"]/29,
                self.bots[botName]["adjustmentPeriod"],
                self.bots["red_1"]["x"]/6.0,
                self.bots["red_1"]["y"]/6.0,
                self.bots["red_1"]["heading"]/360.0,
                int(self.bots["red_1"]["holdingCone"]),
                int(self.bots["red_1"]["holdingBeacon"]),
                self.bots["red_2"]["x"]/6.0,
                self.bots["red_2"]["y"]/6.0,
                self.bots["red_2"]["heading"]/360.0,
                int(self.bots["red_2"]["holdingCone"]),
                int(self.bots["red_2"]["holdingBeacon"]),
                self.bots["blue_1"]["x"]/6.0,
                self.bots["blue_1"]["y"]/6.0,
                self.bots["blue_1"]["heading"]/360.0,
                int(self.bots["blue_1"]["holdingCone"]),
                int(self.bots["blue_1"]["holdingBeacon"]),
                self.bots["blue_2"]["x"]/6.0,
                self.bots["blue_2"]["y"]/6.0,
                self.bots["blue_2"]["heading"]/360.0,
                int(self.bots["blue_2"]["holdingCone"]),
                int(self.bots["blue_2"]["holdingBeacon"]),
                self.blue_points/256.0,
                self.red_points/256.0,
                self.red_beacon_one/29,
                self.red_beacon_two/29,
                self.blue_beacon_one/20,
                self.blue_beacon_two/36,
                self.blue_substation/20,
                self.red_substation/20,
                self.blue_stack_one/5,
                self.blue_stack_two/5,
                self.red_stack_one/5,
                self.red_stack_two/5,
                int(self.red_terminal_one),
                int(self.blue_terminal_one),
                int(self.red_terminal_two),
                int(self.blue_terminal_two)
                )
    def render(self):
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
        return initial

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]



