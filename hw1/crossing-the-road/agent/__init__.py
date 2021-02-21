SUBMISSION = False #Set this to true for submission. Set it to False if testing on your machine.

import gym
import gym_grid_driving
from gym_grid_driving.envs.grid_driving import LaneSpec, Point
import os 

################### DO NOT CHANGE THIS ###################
if not SUBMISSION :
    FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH = "/fast_downward/"
else :
    FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH = ""
PDDL_FILE_ABSOLUTE_PATH = ""
##########################################################

### Sample test cases for Parking task. 
# test_config = [{'lanes' : [LaneSpec(2, [0, 0])] *3,'width' :5, 'seed' : 13},
#                {'lanes' : [LaneSpec(2, [0, 0])] *3,'width' :5, 'seed' : 10},
#                {'lanes' : [LaneSpec(3, [0, 0])] *4,'width' :10, 'seed' : 25},
#                {'lanes' : [LaneSpec(4, [0, 0])] *4,'width' :10, 'seed' : 25},
#                {'lanes' : [LaneSpec(8, [0, 0])] *7,'width' :20, 'seed' : 25},
#                {'lanes' : [LaneSpec(7, [0, 0])] *10,'width' :20, 'seed' : 125}]

### Sample test cases for Crossing Task
"""
[
    LaneSpec(cars=6, speed_range=[-2, -2]), # 6 cars in lane 1 with speed -2
    LaneSpec(cars=6, speed_range=[-2, -2]), # 6 cars in lane 2 with speed -2
    LaneSpec(cars=6, speed_range=[-5, -5]), # 6 cars in lane 3 with speed -5
    LaneSpec(cars=6, speed_range=[-5, -5]), # 6 cars in lane 4 with speed -5
    LaneSpec(cars=5, speed_range=[-4, -4]), # 5 cars in lane 5 with speed -4
    LaneSpec(cars=5, speed_range=[-4, -4]), # 5 cars in lane 6 with speed -4
    LaneSpec(cars=5, speed_range=[-2, -2])  # 5 cars in lane 7 with speed -2
]
"""

test_config = [{'lanes': [LaneSpec(6, [-2, -2])] *2 + [LaneSpec(6, [-5, -5])] *2 + [LaneSpec(5, [-4, -4])] *2 + [LaneSpec(5, [-2, -2])] *1, 
                'width': 30, 
                'seed': 101},
               {'lanes': [LaneSpec(3, [-1, -1])] *3,
                'width': 5, 
                'seed': 12},
               {'lanes': [LaneSpec(2, [-1, -1])] *3,
                'width': 5, 
                'seed': 25},
               {'lanes': [LaneSpec(3, [-1, -1])] *4,
                'width': 10, 
                'seed': 125},
               {'lanes': [LaneSpec(6, [-2, -2])] *2 + [LaneSpec(6, [-5, -5])] *2 + [LaneSpec(5, [-4, -4])] *2 + [LaneSpec(5, [-2, -2])] *1, 
                'width': 30, 
                'seed': 101},
               {'lanes': [LaneSpec(7, [-3, -3])] *2 + [LaneSpec(6, [-4, -4])] *2 + [LaneSpec(6, [-4, -4])] *2 + [LaneSpec(6, [-3, -3])] *2 + [LaneSpec(5, [-2, -2])] *2 + [LaneSpec(5, [-3, -3])] *2, 
                'width': 35, 
                'seed': 40}
              ]

test_case_number = 0 # Change the index for a different test case
LANES = test_config[test_case_number]['lanes']
WIDTH = test_config[test_case_number]['width']
RANDOM_SEED = test_config[test_case_number]['seed']

TIME_FACTOR = 1 # Only cars can wrap around, agent can't wrap around.


class GeneratePDDL_Stationary :
    '''
    Class to generate the PDDL files given the environment description.
    '''
    def __init__ (self, env, num_lanes, width, file_name) :
        self.state = env.reset()
        self.lanes = env.lanes
        self.num_lanes = num_lanes
        self.width = width
        self.file_name = file_name
        self.problem_file_name = self.file_name + 'problem.pddl' 
        self.domain_file_name = self.file_name + 'domain.pddl' 
        self.domain_string = ""
        self.type_string = ""
        self.predicate_strings = self.addHeader("predicates")
        self.action_strings = ""
        self.problem_string = ""
        self.object_strings = self.addHeader("objects")

        self.time_limit = TIME_FACTOR * self.width


    def addDomainHeader(self, name='default_header') :
        '''
        Adds the header in the domain file.

        Parameters : 
        name (string): domain name.
        '''
        self.domain_header = "(define (domain " + name +" ) \n" +"(:requirements :strips :typing) \n"


    def addTypes(self, types={}) :
        '''
        Adds the object types to the PDDL domain file.

        Parameters : 
        types (dict): contains a dictionary of (k,v) pairs, where k is the object type, and v is the supertype. If k has no supertype, v is None.
        '''
        type_string = "(:types "

        for _type, _supertype in types.items() :
            if _supertype is None :
                type_string += _type +  "\n"
            else : 
                type_string += _type + " - " + _supertype + "\n"
        type_string += ") \n"
        self.type_string = type_string


    def addPredicate(self, name='default_predicate', parameters = (), isLastPredicate=False) :
        '''
        Adds predicates to the PDDL domain file

        Parameters : 
        name (string) : name of the predicate.
        parameters (tuple or list): contains a list of (var_name, var_type) pairs, where var_name is an instance of object type var_type.
        isLastPredicate (bool) : True for the last predicate added.
        '''
        predicate_string = "(" + name
        for var_name, var_type in parameters :
            predicate_string += " ?" + var_name + " - " + var_type
        predicate_string += ") \n"
        self.predicate_strings+= predicate_string

        if isLastPredicate :
            self.predicate_strings += self.addFooter()


    def addAction(self, name='default_action', parameters=(), precondition_string= "", effect_string= "") :
        '''
        Adds actions to the PDDL domain file

        Parameters : 
        name (string) : name of the action.
        parameters (tuple or list): contains a list of (var_name, var_type) pairs, where var_name is an instance of object type var_type.
        precondition_string (string) : The precondition for the action.
        effect_string (string) : The effect of the action.
        '''
        action_string = name + "\n"
        parameter_string = ":parameters ("
        for var_name, var_type in parameters :
            parameter_string += " ?" + var_name + " - " + var_type
        parameter_string += ") \n"
        
        precondition_string = ":precondition " + precondition_string + "\n"
        effect_string = ":effect " + effect_string + "\n"
        action_string += parameter_string + precondition_string + effect_string
        action_string = self.addHeader("action") + action_string + self.addFooter()
        self.action_strings+= action_string

    def generateDomainPDDL(self) :
        '''
        Generates the PDDL domain file after all actions, predicates and types are added
        '''
        domain_file = open(PDDL_FILE_ABSOLUTE_PATH + self.domain_file_name, "w")
        PDDL_String = self.domain_header + self.type_string + self.predicate_strings + self.action_strings + self.addFooter()
        domain_file.write(PDDL_String)
        domain_file.close()

   
    def addProblemHeader(self, problem_name='default_problem_name', domain_name='default_domain_name') :
        '''
        Adds the header in the problem file.

        Parameters : 
        problem_name (string): problem name.
        domain_name (string): domain name.
        '''
        self.problem_header = "(define (problem " + problem_name + ") \n (:domain " + domain_name + ") \n"
    

    def addObjects(self, obj_type, obj_list=[], isLastObject=False) :
        '''
        Adds object instances of the same type to the problem file

        Parameters :
        obj_type (string) : the object type of the instances that are being added
        obj_list (list(str)) : a list of object instances to be added
        isLastObject (bool) : True for the last set of objects added.
        '''
        obj_string = ""
        for obj in obj_list :
            obj_string += obj + " "
        obj_string += " - " + obj_type
        self.object_strings += obj_string + "\n "
        if isLastObject :
            self.object_strings += self.addFooter()


    def addInitState(self) :
        '''
        Generates the complete init state
        '''
        initString = self.generateInitString()
        self.initString = self.addHeader("init") + initString + self.addFooter()


    def addGoalState(self) :
        '''
        Generates the complete goal state
        '''
        goalString = self.generateGoalString()
        self.goalString = self.addHeader("goal") + goalString + self.addFooter()


    def generateGridCells(self) :
        '''
        Generates the grid cell objects. 
        
        For a |X+1| x |Y+1| sized grid, |X+1| x |Y+1| objects to represent each grid cell are created. 
        pt0pt0, pt1pt0, .... ptxpt0
        pt0pt1, pt1pt1, .... ptxpt1
        ..       ..            ..
        ..       ..            ..
        pt0pty, pt1pty, .... ptxpty


        '''
        self.grid_cell_list = []
        for w in range(self.width) :
            for lane in range(self.num_lanes) :
                self.grid_cell_list.append("pt{}pt{}".format(w, lane))
    

    def generateTimespan(self):
        self.time_span = []
        for t in range(self.time_limit + 1):
            self.time_span.append("t{}".format(t))


    def set_to_str(self, set_):
        list_of_str = [str(s) for s in set_]
        joined_str = " ".join(list_of_str)
        return joined_str

    def generateInitString(self) :
        '''
        FILL ME : Should return the init string in the problem PDDL file. 
        Hint : Use the defined grid cell objects from genearateGridCells and predicates to construct the init string.

        Information that might be useful here :

        1. Initial State of the environment : self.state
        2. Agent's x position : self.state.agent.position.x
        3. Agent's y position : self.state.agent.position.y
        4. The object of type agent is called "agent1" (see generateProblemPDDLFile() ).
        5. Set of cars in the grid: self.state.cars
        6. For a car in self.state.cars, it's x position: car.position.x
        7. For a car in self.state.cars, it's y position: car.position.y
        8. List of grid cell objects : self.grid_cell_list
        9. Width of the grid: self.width
        10. Number of lanes in the grid : self.num_lanes
        
        Play with environment (https://github.com/cs4246/gym-grid-driving) to see the type of values above objects return

        Example: The following statement adds the initial condition string from https://github.com/pellierd/pddl4j/blob/master/pddl/logistics/p01.pddl  

        return "(at apn1 apt2) (at tru1 pos1) (at obj11 pos1) (at obj12 pos1) (at obj13 pos1) (at tru2 pos2) (at obj21 pos2) (at obj22 pos2)
                (at obj23 pos2) (in-city pos1 cit1) (in-city apt1 cit1) (in-city pos2 cit2) (in-city apt2 cit2)" 
        '''  
        start_x = self.state.agent.position.x
        start_y = self.state.agent.position.y
        start_t = 0
        agent = 'agent1'

        at_pos = f'(at pt{start_x}pt{start_y} {agent})'
        at_time = f'(at_time t{start_t})'

        next_time = ''
        for time in range(start_t, self.time_limit):
            next_time += f'(next_time t{time} t{time + 1}) '

        next_time = next_time.rstrip()
        agent_str = f'{at_pos} \n{at_time} \n{next_time}'

        trail_set = set()
        blocked_set = set()
        # car status at t0 is of no interest, so we start with t1
        for t in range(1, self.time_limit + 1):
            for car in self.state.cars:
                car_lane = car.position.y
                lane_speed = self.lanes[car_lane][1][0]
                # lane_speed is defined as negative
                # wrapped_x = self.width - ((car.position.x - lane_speed * t) % self.width)
                wrapped_x = car.position.x + lane_speed * t
                if wrapped_x < 0:
                    wrapped_x = self.width - (abs(wrapped_x) % self.width)

                # block occupancy trails
                for s in range(1, abs(lane_speed)):
                    trail_x = (wrapped_x + s) % self.width
                    trail_set.add(f'(blocked pt{trail_x}pt{car_lane} t{t}) ')

                # block current car position at time t
                car_pos = f'pt{wrapped_x}pt{car_lane}'
                blocked_set.add(f'(blocked {car_pos} t{t}) ')
        
        car_set = trail_set.union(blocked_set)
        car_str = self.set_to_str(car_set)        

        up_str = ''
        down_str = ''
        forward_str = ''
        for w in range(self.width):
            for lane in range(self.num_lanes):
                # consider wrapping in the conditions
                if lane < self.num_lanes - 1:
                    pos_x = (w + 1) % self.width
                    up_str += f'(up_next pt{pos_x}pt{lane+1} pt{w}pt{lane}) '
                if lane > 0:
                    pos_x = (w + 1) % self.width
                    down_str += f'(down_next pt{pos_x}pt{lane-1} pt{w}pt{lane}) '

                # include agent's speed range
                upper_speed = self.state.agent.speed_range[0]
                lower_speed = self.state.agent.speed_range[1]
                for s in range(upper_speed, lower_speed + 1):
                    pos_x = (w - s) % self.width
                    forward_str += f'(forward_next pt{pos_x}pt{lane} pt{w}pt{lane}) '
        
        move_str = f'{up_str.rstrip()} \n{down_str.rstrip()} \n{forward_str}'

        agent_str = agent_str.rstrip()
        car_str = car_str.rstrip()
        move_str = move_str.rstrip()

        init_str = f'{agent_str} \n\n\n{car_str} \n\n\n{move_str}'
        return init_str


    def generateGoalString(self) :
        '''
        FILL ME : Should return the goal string in the problem PDDL file
        Hint : Use the defined grid cell objects from genearateGridCells and predicates to construct the goal string.

        Information that might be useful here :
        1. Goal x Position : self.state.finish_position.x
        2. Goal y Position : self.state.finish_position.y
        3. The object of type agent is called "agent1" (see generateProblemPDDLFile() ).
        Play with environment (https://github.com/cs4246/gym-grid-driving) to see the type of values above objects return

        Example: The following statement adds goal string from https://github.com/pellierd/pddl4j/blob/master/pddl/logistics/p01.pddl  

        return "(and (at obj11 apt1) (at obj23 pos1) (at obj13 apt1) (at obj21 pos1)))"
        '''
        goal_x = self.state.finish_position.x
        goal_y = self.state.finish_position.y
        agent = 'agent1'

        return f'(at pt{goal_x}pt{goal_y} {agent})'



    def generateProblemPDDL(self) :
        '''
        Generates the PDDL problem file after the object instances, init state and goal state are added
        '''
        problem_file = open(PDDL_FILE_ABSOLUTE_PATH + self.problem_file_name, "w")
        PDDL_String = self.problem_header + self.object_strings + self.initString + self.goalString + self.addFooter()
        problem_file.write(PDDL_String)
        problem_file.close()


    '''
    Helper Functions 
    '''
    def addHeader(self, name) :
        return "(:" + name + " "


    def addFooter(self) :
        return ") \n"


      

def initializeSystem(env):
    gen = GeneratePDDL_Stationary(env, len(env.lanes), width=env.width, file_name='HW1')
    return gen


def generateDomainPDDLFile(gen):
    '''
    Function that specifies the domain and generates the PDDL Domain File. 
    As a part of the assignemnt, you will need to add the actions here.
    '''
    gen.addDomainHeader("grid_world")
    gen.addTypes(types = {"car" : None, "agent" : "car", "gridcell" : None, "time" : None})

    '''
    Predicate Definitions :
    (at ?pt ?car) : car is at gridcell pt.
    (up_next ?pt1 ?pt2) : pt2 is the next location of the car when it takes the UP action from pt1
    (down_next ?pt1 ?pt2) : pt2 is the next location of the car when it takes the DOWN action from pt1
    (forward_next ?pt1 ?pt2) : pt2 is the next location of the car when it takes the FORWARD action from pt1
    (blocked ?pt) : The gridcell pt is occupied by a car and is "blocked".
    '''
    gen.addPredicate(name="at", parameters=(("pt1" , "gridcell"), ("car", "car")))
    gen.addPredicate(name="at_time", parameters=[("t1", "time")])
    gen.addPredicate(name="next_time", parameters=(("t1", "time"), ("t2", "time")))
    gen.addPredicate(name="up_next", parameters=(("pt1" , "gridcell"), ("pt2", "gridcell")))
    gen.addPredicate(name="down_next", parameters=(("pt1" , "gridcell"), ("pt2", "gridcell")))
    gen.addPredicate(name="forward_next", parameters=(("pt1" , "gridcell"), ("pt2", "gridcell")))
    gen.addPredicate(name="blocked", parameters=[("pt1" , "gridcell"), ("t2", "time")] , isLastPredicate=True)

    '''
    FILL ME : Add the actions UP, DOWN, FORWARD with the help of gen.addAction() as follows :

        gen.addAction(name="UP", parameters = (...), precondition_string = "...", effect_string="...")
        gen.addAction(name="DOWN", parameters = (...), precondition_string = "...", effect_string="...")
        gen.addAction(name="FORWARD", parameters = (...), precondition_string = "...", effect_string="...")
        
        You have to fill up the ... in each of gen.addAction() above.
        
    Example :

    The following statement adds the LOAD-TRUCK action from https://tinyurl.com/y3jocxdu [The domain file referenced in the assignment] to the domain file 
    gen.addAction(name="LOAD-TRUCK", 
                  parameters=(("pkg", "package"), ("truck" , "truck"), ("loc", "place")), 
                  precondition_string="(and (at ?truck ?loc) (at ?pkg ?loc))", 
                  effect_string= "(and (not (at ?pkg ?loc)) (in ?pkg ?truck))")
    '''
    gen.addAction(name="UP", 
                  parameters=(("pt1" , "gridcell"), ("pt2" , "gridcell"), ("agt", "agent"), ("t1", "time"), ("t2", "time")), 
                  precondition_string="(and (at ?pt1 ?agt) (at_time ?t1) (next_time ?t1 ?t2) (up_next ?pt1 ?pt2) (not (blocked ?pt2 ?t2)))", 
                  effect_string= "(and (not (at ?pt1 ?agt)) (not (at_time ?t1)) (at ?pt2 ?agt) (at_time ?t2))")

    gen.addAction(name="DOWN", 
                  parameters=(("pt1" , "gridcell"), ("pt2" , "gridcell"), ("agt", "agent"), ("t1", "time"), ("t2", "time")), 
                  precondition_string="(and (at ?pt1 ?agt) (at_time ?t1) (next_time ?t1 ?t2) (down_next ?pt1 ?pt2) (not (blocked ?pt2 ?t2)))", 
                  effect_string= "(and (not (at ?pt1 ?agt)) (not (at_time ?t1)) (at ?pt2 ?agt) (at_time ?t2))")

    gen.addAction(name="FORWARD", 
                  parameters=(("pt1" , "gridcell"), ("pt2" , "gridcell"), ("agt", "agent"), ("t1", "time"), ("t2", "time")), 
                  precondition_string="(and (at ?pt1 ?agt) (at_time ?t1) (next_time ?t1 ?t2) (forward_next ?pt1 ?pt2) (not (blocked ?pt2 ?t2)))", 
                  effect_string= "(and (not (at ?pt1 ?agt)) (not (at_time ?t1)) (at ?pt2 ?agt) (at_time ?t2))")

    gen.generateDomainPDDL()

def generateProblemPDDLFile(gen):
    '''
    Function that specifies the domain and generates the PDDL Domain File.
    Objects defined here should be used to construct the init and goal strings
    '''
    gen.addProblemHeader("parking", "grid_world")
    gen.addObjects("agent", ["agent1"])
    gen.generateTimespan()
    gen.addObjects("time", gen.time_span)
    gen.generateGridCells()
    gen.addObjects("gridcell", gen.grid_cell_list, isLastObject=True)
    gen.addInitState()
    gen.addGoalState()
    gen.generateProblemPDDL()
    pass

def runPDDLSolver(gen):
    '''
    Runs the fast downward solver to get the optimal plan
    '''
    os.system(FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH + 'fast-downward.py ' + PDDL_FILE_ABSOLUTE_PATH + gen.domain_file_name + ' ' + PDDL_FILE_ABSOLUTE_PATH + gen.problem_file_name + ' --search  \"lazy_greedy([ff()], preferred=[ff()])\"' + ' > temp ')

def delete_files(gen) :
    '''
    Deletes PDDL and plan files created.
    '''
    os.remove(PDDL_FILE_ABSOLUTE_PATH + gen.domain_file_name)
    os.remove(PDDL_FILE_ABSOLUTE_PATH + gen.problem_file_name)
    os.remove('sas_plan')


def simulateSolution(env):
    '''
    Simulates the plan given by the solver on the environment
    '''
    env.render() # env.actions: up, down, forward[-3], forward[-2], forward[-1]
    plan_file = open('sas_plan', 'r')
    for line in plan_file.readlines() :
        if line[0] == '(' :
            action = line.split()[0][1:]
            if action == 'up' :
                env.step(env.actions[0])
            if action == 'down' :
                env.step(env.actions[1])
            if action == 'forward' :
                start = int(line.split()[1].split("pt")[1])
                end = int(line.split()[2].split("pt")[1])
                res = start - end
                fs_range = len(env.actions) - 2
                action_idx = (2 + (fs_range - res)) % len(env.actions)

                env.step(env.actions[action_idx])
            env.render()

def generatePlan(env):
    '''
    Extracts the plan given by the solver into a list of actions
    '''
    plan_file = open('sas_plan', 'r')
    action_sequence = []
    for line in plan_file.readlines() :
        if line[0] == '(' :
            action = line.split()[0][1:]
            if action == 'up' :
                action_sequence.append(env.actions[0])
            if action == 'down' :
                action_sequence.append(env.actions[1])
            if action == 'forward' :
                start = int(line.split()[1].split("pt")[1])
                end = int(line.split()[2].split("pt")[1])
                res = start - end
                fs_range = len(env.actions) - 2
                action_idx = (2 + (fs_range - res)) % len(env.actions)

                action_sequence.append(env.actions[action_idx])
    return action_sequence

def test() :
    '''
    Generates the PDDL files, solves for the optimal solution and simulates the plan. The PDDL files are deleted at the end.
    '''
    env=gym.make('GridDriving-v0', lanes=LANES, width=WIDTH, random_seed=RANDOM_SEED, agent_speed_range=(-3,-1))
    gen = initializeSystem(env)
    generateDomainPDDLFile(gen)
    generateProblemPDDLFile(gen)
    runPDDLSolver(gen)
    simulateSolution(env)
    print(f"\nAction sequence:\n{generatePlan(env)}")
    delete_files(gen)

if SUBMISSION :
    from runner.abstracts import Agent
    class PDDLAgent(Agent):
        def initialize(self, params):
            global FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH
            FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH = params[0]
            self.env = params[1]
            gen = initializeSystem(self.env)
            generateDomainPDDLFile(gen)
            generateProblemPDDLFile(gen)
            runPDDLSolver(gen)
            self.action_plan = generatePlan(self.env)
            self.time_step = 0
            delete_files(gen)

        def step(self, state, *args, **kwargs):
            action = self.action_plan[self.time_step]
            self.time_step +=1
            return action

    def create_agent(test_case_env, *args, **kwargs):
        return PDDLAgent() 
else :
    test()
