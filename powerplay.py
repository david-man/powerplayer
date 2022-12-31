import math

import pyspiel
import numpy as np
import networkx as nx

from itertools import islice
'''
class PowerplayState(pyspiel.state):
    def __init__(self, game, auton_results):
        super().__init__(game)
        self.cur_player = 0#0 = blue, 1 = red
        self.blue_score = 0
        self.red_score = 0
        self.isDone = 0
        self.timeLeft = 120#120 seconds

    def current_player(self):
        return pyspiel.PlayerId.TERMINAL if self.timeLeft==0 else self.cur_player
'''

'''
Assumptions laid out here:
- Bots move at 2 tiles/s
- Bots take 1s to turn to any given position while moving to junctions
- Bots take an adjustably long time to adjust to a junction, this will be the only thing affecting cycle time between bots
- Each "move" in this game is 0.5 seconds long

'''
class PowerplayGame:
    def __init__(self):
        self.junctions = np.zeros((29,))# includes terminals, will include a picture of what junction is where later
        #constants about the locations of key areas
        self.terminals = [0,1,2,3]
        self.ground_junctions = [4, 6, 8, 14, 16, 18, 24, 26, 28]
        self.low_junctions = [5, 7, 9, 13, 19, 23, 25, 27]
        self.medium_junctions = [10, 12, 20, 22]
        self.high_junctions = [11, 15, 17, 21]
        self.red_cone_areas = [3, 17, 23, 33]
        self.blue_cone_areas = [2, 12, 18, 32]


        #variables about game
        self.blue_points = 0
        self.red_points = 0
        self.blue_substation = 20
        self.red_substation = 20
        self.blue_stack_one = 5
        self.blue_stack_two = 5
        self.red_stack_one = 5
        self.red_stack_two = 5
        self.red_beaconed = []
        self.blue_beaconed = []
        self.red_terminal_one = False
        self.blue_terminal_one = False
        self.red_terminal_two = False
        self.blue_terminal_two = False

        self.bots = [
            {"x": 5.5, "y": 1.5, "heading": 180, "holdingCone": False, "holdingBeacon": False, "path": [],
             "team": "red", "gotBeacon": False, "junctionTo": None, "busy": 0, "released": False,
             "adjustmentPeriod": 2},
            {"x": 5.5, "y": 4.5, "heading": 180, "holdingCone": False, "holdingBeacon": False,"path": [],
             "team": "red", "gotBeacon": False, "junctionTo": None, "busy": 0, "released": False,
             "adjustmentPeriod": 2},
            {"x": 0.5, "y": 1.5, "heading": 180, "holdingCone": False, "holdingBeacon": False,"path": [],
             "team": "blue", "gotBeacon": False, "junctionTo" : None, "busy":0, "released": False,
             "adjustmentPeriod":2},
            {"x": 0.5, "y": 4.5, "heading": 180, "holdingCone": False, "holdingBeacon": False,"path": [],
             "team" : "blue", "gotBeacon":False, "junctionTo": None, "busy":0, "released": False,
             "adjustmentPeriod":2}
        ]

        self.timeElapsed = 0

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(35))
        for r in range(35):
            col, row = self.boxToCoords(r)
            if (row < 5):
                self.graph.add_edge(r, r + 6)
            if (col < 5):
                self.graph.add_edge(r, r + 1)

        # right now, no auton implementation
    def availableActions(self, actionNum):
        if(actionNum == 0):
            return range(29)#where to put cone; this will be learned
        elif(actionNum==1):
            return [0,1]#0 for cone, 1 for beacon
        else:
            return 0 #program makes the decision : it will lead you to the closest area for getting cones

    def requestRequest(self, robotNum, requestNumber):
        possibilities = self.availableActions(requestNumber)
        print("Robot #" + str(robotNum) + " at " + str(self.bots[robotNum]["x"]) + " " + str(self.bots[robotNum]["y"]) + " needs an input.")
        return int(input("Possible choices: " + str(possibilities)))
    def junctionToBox(self, junctionNumber):
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
            return [self.coordsToBox(col, row), self.coordsToBox(col+1, row),
                    self.coordsToBox(col, row+1), self.coordsToBox(col+1, row+1)]

    def euclideanDistance(self, point_1, point_2):
        return math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1]-point_2[1])**2)

    def simulatePlacement(self, robotNum, junctionAt):#ONLY deals with points and ownership
        referal = self.bots[robotNum]
        if(junctionAt > 3 and ((self.junctions[junctionAt] in self.red_beaconed) or (self.junctions[junctionAt] in self.blue_beaconed))):
            return#nothing counts after beacon is placed for either team
        elif(junctionAt > 3):#deals with changing of ownership
            self.junctions[junctionAt] = 1 if referal["team"] == "red" else -1
        points = 0 if not referal["holdingBeacon"] else 10#assumption that teams will place cone+beacon at the same time
        if(referal["holdingBeacon"]):
            if(referal["team"] == "red"):
                self.red_beaconed.append(junctionAt)
            else:
                self.blue_beaconed.append(junctionAt)
        if(junctionAt < 4):
            points += 1
            if((not self.red_terminal_one) and referal["team"] == "red" and junctionAt == 0):
                self.red_terminal_one = True
            elif((not self.red_terminal_two) and referal["team"] == "red" and junctionAt == 35):
                self.red_terminal_two = True
            elif((not self.blue_terminal_one) and referal["team"] == "blue" and junctionAt == 5):
                self.blue_terminal_one = True
            elif ((not self.blue_terminal_two) and referal["team"] == "blue" and junctionAt == 30):
                self.blue_terminal_two = True
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
    def totalSimulate(self):
        for bot in self.bots:
            self.collisionCheck(bot)
        for r in range(4):
            if(self.bots[r]["busy"] > 0):
                self.bots[r]["busy"] -= 0.5
                self.requestRequest(r, -1)#request a -1 input from the actor
            else:
                b = self.simulateMovement(r)
                if(b): #means that the path is complete: find a new path to pick up a cone, pick up the cone and find a junction, or put a cone on a junction
                    if(self.bots[r]["junctionTo"] is not None):#this means the bot wants to put something on a junction
                        if(self.bots[r]["junctionTo"] >= 4):
                            self.bots[r]["busy"] = self.bots[r]["adjustmentPeriod"]#gives the adjustment time as busy timer ONLY if junction is not a terminal
                            self.bots[r]["junctionTo"] *= -1#flips the "junctionTo" so that it can pass thru to the next part
                        else:
                            self.simulatePlacement(r, abs(self.bots[r]["junctionTo"]))
                            self.bots[r]["holdingCone"] = False
                            self.bots[r]["holdingBeacon"] = False
                            self.bots[r]["released"] = False  # means that the robot can get itself stuck again, limit is one "stuck" per cycle
                            self.bots[r]["junctionTo"] = None
                        self.requestRequest(r, -1)

                    elif((self.bots[r]["team"] == "red" and (self.coordsToBox(self.bots[r]["x"], self.bots[r]["y"]) not in self.red_cone_areas)) or
                         (self.bots[r]["team"] == "blue" and (self.coordsToBox(self.bots[r]["x"], self.bots[r]["y"]) not in self.blue_cone_areas))):
                        #means bot needs to head to a cone station since it's not heading to a junction
                        box_to_go_to = -1
                        box_to_go_to_dist = 100000
                        shutdown = False
                        if(self.bots[r]["team"] == "red"):
                            for area in self.red_cone_areas:
                                if(area == 3 and self.red_stack_one == 0):
                                    continue
                                elif(area == 33 and self.red_stack_two == 0):
                                    continue
                                elif((area == 17 or area == 23) and self.red_substation == 0):

                                        shutdown = True
                                        continue
                                else:
                                    area_row, area_col = self.boxToCoords(area)
                                    dist = self.euclideanDistance([area_row, area_col], [self.bots[r]["x"], self.bots[r]["y"]])
                                    if(dist < box_to_go_to_dist):
                                        box_to_go_to = area
                                        box_to_go_to_dist = dist
                        else:
                            for area in self.blue_cone_areas:
                                if (area == 2 and self.blue_stack_one == 0):
                                    continue
                                elif (area == 32 and self.blue_stack_two == 0):
                                    continue
                                elif ((area == 12 or area == 18) and self.blue_substation==0):

                                        shutdown = True
                                        continue
                                else:
                                    area_row, area_col = self.boxToCoords(area)
                                    dist = self.euclideanDistance([area_row, area_col],
                                                                  [self.bots[r]["x"], self.bots[r]["y"]])
                                    if (dist < box_to_go_to_dist):
                                        box_to_go_to = area
                                        box_to_go_to_dist = dist
                        if(not shutdown):
                            self.bots[r]["path"] = self.shortestPath(self.coordsToBox(self.bots[r]["x"], self.bots[r]["y"]), box_to_go_to, 5, self.bots[r]["heading"])
                        self.requestRequest(r, -1)#actor has no choice where the bot goes: it's going to the closest one possible or nowhere at all
                    elif ((self.bots[r]["team"] == "red" and (self.coordsToBox(self.bots[r]["x"], self.bots[r]["y"]) in self.red_cone_areas)) or
                          (self.bots[r]["team"] == "blue" and (self.coordsToBox(self.bots[r]["x"], self.bots[r]["y"]) in self.blue_cone_areas))):
                        #means bot is looking to pick up a cone
                        if((not self.bots[r]["holdingCone"]) and (not self.bots[r]["holdingBeacon"])):
                            #pickup simulation
                            self.bots[r]["holdingCone"] = True
                            if(self.timeElapsed>=90 and
                                    (not self.bots[r]["gotBeacon"]) and
                                    ((self.bots[r]["team"] == "red" and (self.coordsToBox(self.bots[r]["x"], self.bots[r]["y"]) in [17,23])) or
                                     (self.bots[r]["team"] == "blue" and (self.coordsToBox(self.bots[r]["x"], self.bots[r]["y"]) in [12,18])))):
                                f = self.requestRequest(r, 1)
                                if(f == 1):#if the actor chooses to get a beacon
                                    self.bots[r]["holdingBeacon"] = True
                                    self.bots[r]["holdingCone"] = False
                                    self.bots[r]["gotBeacon"] = True
                            else:
                                self.requestRequest(r, -1)#actor only given choice to choose a cone


                            box = self.coordsToBox(self.bots[r]["x"], self.bots[r]["y"])#take one cone away from wherever
                            if(box == 3):
                                self.red_stack_one-=1
                            elif(box==33):
                                self.red_stack_two -= 1
                            elif(box == 17 or box == 23):
                                self.red_substation -= 1
                            elif(box == 2):
                                self.blue_stack_one -= 1
                            elif(box == 32):
                                self.blue_stack_two -= 1
                            elif(box == 12 or box == 18):
                                self.blue_substation -= 1
                        else:
                            #choose path simulation
                            junction = self.requestRequest(r, 0)
                            junction_boxes = self.junctionToBox(junction)
                            junction_box = -1
                            junction_distance = 100000
                            print(junction_boxes)

                            for j in junction_boxes:
                                dist = self.euclideanDistance([self.bots[r]["x"], self.bots[r]["y"]], self.boxToCoords(j))
                                if(dist < junction_distance):
                                    junction_box = j
                                    junction_distance = dist
                            print(junction_box)
                            self.bots[r]["junctionTo"] = junction
                            self.bots[r]["released"] = False

                            self.bots[r]["path"] = self.shortestPath(self.coordsToBox(self.bots[r]["x"], self.bots[r]["y"]),
                                                                     junction_box, 3,
                                                                     self.bots[r]["heading"])

                else:
                    self.requestRequest(r, -1)
        self.timeElapsed+=0.5

    def simulateMovement(self, robotNum):#apply a movement on a robot given a path
        referal = self.bots[robotNum]
        box = self.coordsToBox(referal["x"], referal["y"])
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

    def collisionCheck(self, referal):#this function modifies the bot by adding a timer to the "busy" part, checked ONLY by simulateMovement
        toReturn = False
        for bot in self.bots:
            if(referal["released"] or bot["released"]):
                continue
            if(id(referal) == id(bot)):
                continue
            elif(len(referal["path"]) == 0):
                return False
            else:
                #three types of collisions are noted here: head-on, hitting stationary, hitting side
                #head-on/side-on
                if((len(referal["path"]) > 0 and len(bot["path"]) > 0) and
                        (referal["path"][0] == self.coordsToBox(bot["x"], bot["y"]))
                        and (bot["path"][0] == self.coordsToBox(referal["x"], referal["y"]))):
                    bot["released"] = True
                    referal["released"] = True
                    toReturn = True
                    if(abs(referal["heading"] - bot["heading"]) == 180):#designates a head-on:
                        bot["busy"] = 1
                        referal["busy"] = 1
                        #print("Head On Collision")
                    else:#side-on
                        referal["busy"] = max(0.5, referal["busy"])
                elif((len(referal["path"]) > 0 and len(bot["path"]) == 0) and
                     (referal["path"][0] == self.coordsToBox(bot["x"], bot["y"]))):#stationary hit
                    referal["busy"] = 1
                    bot["busy"] = 1
                    toReturn = True
                    referal["released"] = True
        return toReturn






    #none of the functions below modify the bots' states in any way, shape, or form
    def shortestPath(self, box_f, box_t, pathNum, cur_orientation):
        t = list(islice(nx.shortest_simple_paths(self.graph, box_f, box_t), pathNum))  # djikstra implementation but i was too lazy to actually do it
        #print(t)
        bestPath = []
        bestTime = 1000000
        for path in t:
            p1 = path.copy()
            time = self.calculatePath(path, cur_orientation)
            if(time < bestTime):
                bestPath = p1
                bestTime = time
        return bestPath[1:]

    def calculatePath(self, path, cur_orientation):
        last = path.pop(0)
        timeUsed = 0
        while(len(path)!=0):
            nextElement = path.pop(0)
            if(cur_orientation == 0 and nextElement == last+1 or
            cur_orientation == 90 and nextElement == last - 6 or
            cur_orientation == 180 and nextElement == last-1 or
            cur_orientation == 270 and nextElement == last+6):
                timeUsed += 0.5
            else:
                timeUsed += 1 #.5 second for movement, one second for turning
                if(nextElement == last+1):
                    cur_orientation = 0
                elif(nextElement == last-6):
                    cur_orientation = 90
                elif(nextElement == last-1):
                    cur_orientation = 180
                elif(nextElement == last+6):
                    cur_orientation = 270

            last = nextElement

        return timeUsed
    def coordsToBox(self, x, y):
        return math.floor(y)*6+math.floor(x)

    def boxToCoords(self, box):
        box_y = math.floor(box/6)
        box_x = box - box_y*6
        return box_x, box_y

    def junctionToStr(self):
        print("CORNERS: " + str(int(self.junctions[0])) + " " + str(int(self.junctions[1])) + " " + str(int(self.junctions[2])) + " " + str(int(self.junctions[3])))
        print("JUNCTION DATA: ")
        for row in range(5):
            row_ = row*5+4
            print(str(int(self.junctions[row_])) + " " + str(int(self.junctions[row_+1])) + " " +
                  str(int(self.junctions[row_+2])) + " " + str(int(self.junctions[row_ + 3])) + " " + str(int(self.junctions[row_+4])))

    def toString(self):
        spacesNeededLeft = 2 if self.blue_substation >= 10 else 1
        spacesNeededRight = 2 if self.red_substation >= 10 else 1
        initial = (str(int(self.junctions[0])) + spacesNeededLeft * " " + "   " + str(self.blue_stack_one) + " " + str(self.red_stack_one) + spacesNeededRight * " " + "   " + str(int(self.junctions[1])) + "\n" +
                   spacesNeededLeft*" " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                   spacesNeededLeft*" " +" G L G L G " + spacesNeededRight * " " +"\n" +
                   spacesNeededLeft*" " +"_ _ _ _ _ _" + spacesNeededRight * " " +"\n" +
                   spacesNeededLeft*" " +" L M H M L " + spacesNeededRight * " " +"\n" +
                   spacesNeededLeft*" " +"_ _ _ _ _ _" + spacesNeededRight * " " +"\n" +
                   str(self.blue_substation) + " G H G H G " + str(self.red_substation) + "\n" +
                   spacesNeededLeft*" " +"_ _ _ _ _ _" + spacesNeededRight * " " +"\n" +
                   spacesNeededLeft*" " +" L M H M L " + spacesNeededRight * " " +"\n" +
                   spacesNeededLeft*" " +"_ _ _ _ _ _" + spacesNeededRight * " " +"\n" +
                   spacesNeededLeft*" " +" G L G L G " + spacesNeededRight * " " +"\n" +
                   spacesNeededLeft*" " +"_ _ _ _ _ _" + spacesNeededRight * " " +"\n" +
                   str(int(self.junctions[2])) + spacesNeededLeft*" " +"   " + str(self.blue_stack_two) + " " + str(self.red_stack_two) + spacesNeededRight * " " +"   " + str(int(self.junctions[3])))

        for bot in self.bots:
            box = self.coordsToBox(bot["x"], bot["y"])
            box_x, box_y = self.boxToCoords(box)
            x = box_x*2+spacesNeededLeft
            y = box_y*2 + 1
            ind = y*(12 + spacesNeededRight + spacesNeededLeft) + x
            toPut = "R" if bot["team"] == "red" else "B"
            initial = initial[:ind] + toPut + initial[ind+1:]
        return initial



p = PowerplayGame()
#f = p.shortestPath(5,6, 3, 180)

for i in range(20):
    print(p.toString() + "\n\n")
    print(str(p.bots) + "\n\n")
    print("TIME: " + str(p.timeElapsed))
    p.junctionToStr()
    print("RED: " + str(p.red_points))
    print("BLUE: " + str(p.blue_points))
    p.totalSimulate()
