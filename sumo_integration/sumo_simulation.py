#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
""" This module is responsible for the management of the sumo simulation. """

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import collections
import enum
import logging
import os

import carla  # pylint: disable=import-error
import sumolib  # pylint: disable=import-error
import traci  # pylint: disable=import-error

from .constants import INVALID_ACTOR_ID
# from constants import INVALID_ACTOR_ID
from utils.arguements import parse_args

import lxml.etree as ET  # pylint: disable=import-error

# ==================================================================================================
# -- sumo definitions ------------------------------------------------------------------------------
# ==================================================================================================


# https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#signal_state_definitions
class SumoSignalState(object):
    """
    SumoSignalState contains the different traffic light states.
    """
    RED = 'r'
    YELLOW = 'y'
    GREEN = 'G'
    GREEN_WITHOUT_PRIORITY = 'g'
    GREEN_RIGHT_TURN = 's'
    RED_YELLOW = 'u'
    OFF_BLINKING = 'o'
    OFF = 'O'


# https://sumo.dlr.de/docs/TraCI/Vehicle_Signalling.html
class SumoVehSignal(object):
    """
    SumoVehSignal contains the different sumo vehicle signals.
    """
    BLINKER_RIGHT = 1 << 0
    BLINKER_LEFT = 1 << 1
    BLINKER_EMERGENCY = 1 << 2
    BRAKELIGHT = 1 << 3
    FRONTLIGHT = 1 << 4
    FOGLIGHT = 1 << 5
    HIGHBEAM = 1 << 6
    BACKDRIVE = 1 << 7
    WIPER = 1 << 8
    DOOR_OPEN_LEFT = 1 << 9
    DOOR_OPEN_RIGHT = 1 << 10
    EMERGENCY_BLUE = 1 << 11
    EMERGENCY_RED = 1 << 12
    EMERGENCY_YELLOW = 1 << 13


# https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#abstract_vehicle_class
class SumoActorClass(enum.Enum):
    """
    SumoActorClass enumerates the different sumo actor classes.
    """
    IGNORING = "ignoring"
    PRIVATE = "private"
    EMERGENCY = "emergency"
    AUTHORITY = "authority"
    ARMY = "army"
    VIP = "vip"
    PEDESTRIAN = "pedestrian"
    PASSENGER = "passenger"
    HOV = "hov"
    TAXI = "taxi"
    BUS = "bus"
    COACH = "coach"
    DELIVERY = "delivery"
    TRUCK = "truck"
    TRAILER = "trailer"
    MOTORCYCLE = "motorcycle"
    MOPED = "moped"
    BICYCLE = "bicycle"
    EVEHICLE = "evehicle"
    TRAM = "tram"
    RAIL_URBAN = "rail_urban"
    RAIL = "rail"
    RAIL_ELECTRIC = "rail_electric"
    RAIL_FAST = "rail_fast"
    SHIP = "ship"
    CUSTOM1 = "custom1"
    CUSTOM2 = "custom2"


SumoActor = collections.namedtuple('SumoActor', 'type_id vclass transform signals extent color')

# ==================================================================================================
# -- sumo traffic lights ---------------------------------------------------------------------------
# ==================================================================================================


class SumoTLLogic(object):
    """
    SumoTLLogic holds the data relative to a traffic light in sumo.
    """
    def __init__(self, tlid, states, parameters):
        self.tlid = tlid
        self.states = states

        self._landmark2link = {}
        self._link2landmark = {}
        for link_index, landmark_id in parameters.items():
            # Link index information is added in the parameter as 'linkSignalID:x'
            link_index = int(link_index.split(':')[1])

            if landmark_id not in self._landmark2link:
                self._landmark2link[landmark_id] = []
            self._landmark2link[landmark_id].append((tlid, link_index))
            self._link2landmark[(tlid, link_index)] = landmark_id

    def get_number_signals(self):
        """
        Returns number of internal signals of the traffic light.
        """
        if len(self.states) > 0:
            return len(self.states[0])
        return 0

    def get_all_signals(self):
        """
        Returns all the signals of the traffic light.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        return [(self.tlid, i) for i in range(self.get_number_signals())]

    def get_all_landmarks(self):
        """
        Returns all the landmarks associated with this traffic light.
        """
        return self._landmark2link.keys()

    def get_associated_signals(self, landmark_id):
        """
        Returns all the signals associated with the given landmark.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        return self._landmark2link.get(landmark_id, [])


class SumoTLManager(object):
    """
    SumoTLManager is responsible for the management of the sumo traffic lights (i.e., keeps control
    of the current program, phase, ...)
    """
    def __init__(self):
        self._tls = {}  # {tlid: {program_id: SumoTLLogic}
        self._current_program = {}  # {tlid: program_id}
        self._current_phase = {}  # {tlid: index_phase}

        for tlid in traci.trafficlight.getIDList():  # 获取当前仿真环境中所有交通信号灯的ID列表
            self.subscribe(tlid) # 订阅该交通信号灯的状态更新，以便获取最新的状态信息。

            self._tls[tlid] = {}
            for tllogic in traci.trafficlight.getAllProgramLogics(tlid): # 获取交通信号灯当前所使用的程序逻辑（即当前控制策略）的ID
                states = [phase.state for phase in tllogic.getPhases()]  # 一个包含所有相位状态的列表
                parameters = tllogic.getParameters()  # 这一行获取当前程序逻辑的所有参数。这些参数可能包括相位持续时间、周期时间、绿灯优先级等信号控制策略的设置
                tl = SumoTLLogic(tlid, states, parameters)
                self._tls[tlid][tllogic.programID] = tl  # self._tls是一个字典里面的套字典的结构，{tlid:{programID:tl}}

            # Get current status of the traffic lights.
            self._current_program[tlid] = traci.trafficlight.getProgram(tlid)  # self._current_program[tlid] = 0
            self._current_phase[tlid] = traci.trafficlight.getPhase(tlid)  # 返回的是给定交通灯的当前相位索引。

        self._off = False

    @staticmethod
    def subscribe(tlid):
        """
        Subscribe the given traffic ligth to the following variables:

            * Current program.
            * Current phase.
        """
        traci.trafficlight.subscribe(tlid, [
            traci.constants.TL_CURRENT_PROGRAM,
            traci.constants.TL_CURRENT_PHASE,
        ])

    @staticmethod
    def unsubscribe(tlid):
        """
        Unsubscribe the given traffic ligth from receiving updated information each step.
        """
        traci.trafficlight.unsubscribe(tlid)

    def get_all_signals(self):
        """
        Returns all the traffic light signals.
        """
        signals = set()
        for tlid, program_id in self._current_program.items():
            signals.update(self._tls[tlid][program_id].get_all_signals())
        return signals

    def get_all_landmarks(self):
        """
        Returns all the landmarks associated with a traffic light in the simulation.
        """
        landmarks = set()
        for tlid, program_id in self._current_program.items():
            landmarks.update(self._tls[tlid][program_id].get_all_landmarks())
        return landmarks

    def get_all_associated_signals(self, landmark_id):
        """
        Returns all the signals associated with the given landmark.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        signals = set()
        for tlid, program_id in self._current_program.items():
            signals.update(self._tls[tlid][program_id].get_associated_signals(landmark_id))
        return signals

    def get_state(self, landmark_id):
        """
        Returns the traffic light state of the signals associated with the given landmark.
        """
        states = set()
        for tlid, link_index in self.get_all_associated_signals(landmark_id):
            current_program = self._current_program[tlid]
            current_phase = self._current_phase[tlid]

            tl = self._tls[tlid][current_program]
            states.update(tl.states[current_phase][link_index])

        if len(states) == 1:
            return states.pop()
        elif len(states) > 1:
            logging.warning('Landmark %s is associated with signals with different states',
                            landmark_id)
            return SumoSignalState.RED
        else:
            return None

    def set_state(self, landmark_id, state):
        """
        Updates the state of all the signals associated with the given landmark.
        """
        for tlid, link_index in self.get_all_associated_signals(landmark_id):
            traci.trafficlight.setLinkState(tlid, link_index, state)
        return True

    def switch_off(self):
        """
        Switch off all traffic lights.
        """
        for tlid, link_index in self.get_all_signals():
            traci.trafficlight.setLinkState(tlid, link_index, SumoSignalState.OFF)
        self._off = True

    def tick(self):
        """
        Tick to traffic light manager
        """
        if self._off is False:
            for tl_id in traci.trafficlight.getIDList():  # 获取当前仿真环境中所有交通信号灯的ID列表。
                results = traci.trafficlight.getSubscriptionResults(tl_id)
                current_program = results[traci.constants.TL_CURRENT_PROGRAM]
                current_phase = results[traci.constants.TL_CURRENT_PHASE]

                if current_program != 'online':
                    self._current_program[tl_id] = current_program
                    self._current_phase[tl_id] = current_phase


# ==================================================================================================
# -- sumo simulation -------------------------------------------------------------------------------
# ==================================================================================================

def _get_sumo_net(cfg_file):
    """
    Returns sumo net.

    This method reads the sumo configuration file and retrieve the sumo net filename to create the
    net.
    """
    cfg_file = os.path.join(os.getcwd(), cfg_file)

    tree = ET.parse(cfg_file)
    tag = tree.find('//net-file')
    if tag is None:
        return None

    net_file = os.path.join(os.path.dirname(cfg_file), tag.get('value'))
    logging.debug('Reading net file: %s', net_file)

    sumo_net = sumolib.net.readNet(net_file)
    return sumo_net

class SumoSimulation(object):
    """
    SumoSimulation is responsible for the management of the sumo simulation.
    """

    def __init__(self, cfg_file, step_length, host=None, sumo_gui=False, client_order=1, seed=42, sumo_label=8813, sumo_port=None):
        self.label = str(sumo_label)
        if sumo_gui is True:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')

        print(sumo_port, self.label)
        if host is None or sumo_port is None:
            logging.info('Starting new sumo server...')
            if sumo_gui is True:
                logging.info('Remember to press the play button to start the simulation')

            # traci.start([sumo_binary,
            #     '--configuration-file', cfg_file,
            #     '--step-length', str(step_length),
            #     '--lateral-resolution', '0.25',
            #     '--collision.check-junctions',
            #     '--seed', str(seed)])

            # traci.start([sumo_binary,
            #     '--configuration-file', cfg_file,
            #     '--step-length', str(step_length),
            #     '--seed'], label=self.label, port=8814)

            traci.start([sumo_binary,
                '--configuration-file', cfg_file,
                '--step-length', str(step_length),
                '--seed', str(seed)])

        else:
            logging.info('Connection to sumo server. Host: %s Port: %s', host, sumo_port)
            traci.init(host=host, port=sumo_port)
        traci.setOrder(client_order)

        # self.conn = traci

        # Retrieving net from configuration file.
        self.net = _get_sumo_net(cfg_file)  # 'sumo_integration/examples/Town05.sumocfg'

        # To keep track of the vehicle classes for which a route has been generated in sumo.
        self._routes = set()

        # Variable to asign an id to new added actors.
        self._sequential_id = 0

        # Structures to keep track of the spawned and destroyed vehicles at each time step.
        self.spawned_actors = set()
        self.destroyed_actors = set()

        # Traffic light manager.
        self.traffic_light_manager = SumoTLManager()

        self.ts_id = list(traci.trafficlight.getIDList())[0]

    @property
    def traffic_light_ids(self):
        return self.traffic_light_manager.get_all_landmarks()

    @staticmethod
    def subscribe(actor_id):
        """
        Subscribe the given actor to the following variables:

            * Type.
            * Vehicle class.
            * Color.
            * Length, Width, Height.
            * Position3D (i.e., x, y, z).
            * Angle, Slope.
            * Speed.
            * Lateral speed.
            * Signals.
        """
        traci.vehicle.subscribe(actor_id, [
            traci.constants.VAR_TYPE, traci.constants.VAR_VEHICLECLASS, traci.constants.VAR_COLOR,
            traci.constants.VAR_LENGTH, traci.constants.VAR_WIDTH, traci.constants.VAR_HEIGHT,
            traci.constants.VAR_POSITION3D, traci.constants.VAR_ANGLE, traci.constants.VAR_SLOPE,
            traci.constants.VAR_SPEED, traci.constants.VAR_SPEED_LAT, traci.constants.VAR_SIGNALS
        ])

    @staticmethod
    def unsubscribe(actor_id):
        """
        Unsubscribe the given actor from receiving updated information each step.
        """
        traci.vehicle.unsubscribe(actor_id)

    def get_net_offset(self):
        """
        Accessor for sumo net offset.
        """
        if self.net is None:
            return (0, 0)
        return self.net.getLocationOffset()

    @staticmethod
    def get_actor(actor_id):
        """
        Accessor for sumo actor.
        """
        results = traci.vehicle.getSubscriptionResults(actor_id)

        type_id = results[traci.constants.VAR_TYPE]
        vclass = SumoActorClass(results[traci.constants.VAR_VEHICLECLASS])
        color = results[traci.constants.VAR_COLOR]

        length = results[traci.constants.VAR_LENGTH]
        width = results[traci.constants.VAR_WIDTH]
        height = results[traci.constants.VAR_HEIGHT]

        location = list(results[traci.constants.VAR_POSITION3D])
        rotation = [results[traci.constants.VAR_SLOPE], results[traci.constants.VAR_ANGLE], 0.0]
        transform = carla.Transform(carla.Location(location[0], location[1], location[2]),
                                    carla.Rotation(rotation[0], rotation[1], rotation[2]))

        signals = results[traci.constants.VAR_SIGNALS]
        extent = carla.Vector3D(length / 2.0, width / 2.0, height / 2.0)

        return SumoActor(type_id, vclass, transform, signals, extent, color)

    def spawn_actor(self, type_id, color=None):
        """
        Spawns a new actor.

            :param type_id: vtype to be spawned.
            :param color: color attribute for this specific actor.
            :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        actor_id = 'carla' + str(self._sequential_id)
        try:
            vclass = traci.vehicletype.getVehicleClass(type_id)
            if vclass not in self._routes:
                logging.debug('Creating route for %s vehicle class', vclass)
                allowed_edges = [e for e in self.net.getEdges() if e.allows(vclass)]
                if allowed_edges:
                    traci.route.add("carla_route_{}".format(vclass), [allowed_edges[0].getID()])
                    self._routes.add(vclass)
                else:
                    logging.error(
                        'Could not found a route for %s. No vehicle will be spawned in sumo',
                        type_id)
                    return INVALID_ACTOR_ID

            traci.vehicle.add(actor_id, 'carla_route_{}'.format(vclass), typeID=type_id)
        except traci.exceptions.TraCIException as error:
            logging.error('Spawn sumo actor failed: %s', error)
            return INVALID_ACTOR_ID

        if color is not None:
            color = color.split(',')
            traci.vehicle.setColor(actor_id, color)

        self._sequential_id += 1

        return actor_id

    @staticmethod
    def destroy_actor(actor_id):
        """
        Destroys the given actor.
        """
        traci.vehicle.remove(actor_id)

    def get_traffic_light_state(self, landmark_id):
        """
        Accessor for traffic light state.

        If the traffic ligth does not exist, returns None.
        """
        return self.traffic_light_manager.get_state(landmark_id)

    def switch_off_traffic_lights(self):
        """
        Switch off all traffic lights.
        """
        self.traffic_light_manager.switch_off()

    def synchronize_vehicle(self, vehicle_id, transform, signals=None):
        """
        Updates vehicle state.

            :param vehicle_id: id of the actor to be updated.
            :param transform: new vehicle transform (i.e., position and rotation).
            :param signals: new vehicle signals.
            :return: True if successfully updated. Otherwise, False.
        """
        loc_x, loc_y = transform.location.x, transform.location.y
        yaw = transform.rotation.yaw

        traci.vehicle.moveToXY(vehicle_id, "", 0, loc_x, loc_y, angle=yaw, keepRoute=2)
        if signals is not None:
            traci.vehicle.setSignals(vehicle_id, signals)
        return True

    def synchronize_traffic_light(self, landmark_id, state):
        """
        Updates traffic light state.

            :param tl_id: id of the traffic light to be updated (logic id, link index).
            :param state: new traffic light state.
            :return: True if successfully updated. Otherwise, False.
        """
        self.traffic_light_manager.set_state(landmark_id, state)

    def tick(self):
        """
        Tick to sumo simulation.
        """
        traci.simulationStep()
        self.traffic_light_manager.tick()

        # Update data structures for the current frame.
        self.spawned_actors = set(traci.simulation.getDepartedIDList())  # 在此时间步中插入道路网络的车辆 ID 列表
        self.destroyed_actors = set(traci.simulation.getArrivedIDList())  # 在此时间步内到达（已到达目的地并从道路网络中删除）的车辆 ID 列表。

    @staticmethod
    def close():
        """
        Closes traci client.
        """
        traci.close()

if __name__ == '__main__':
    args = parse_args()
    # sumo_gui = True
    if args.sumo_gui is True:
        sumo_binary = sumolib.checkBinary('sumo-gui')
    else:
        sumo_binary = sumolib.checkBinary('sumo')
    try:
        # traci.start([sumo_binary,
        #              '-c', "examples/gong_ten_cross_one.sumocfg",
        #              '--step-length', str(0.5)])

        # traci.start([sumo_binary,
        #              '--configuration-file', "examples/gong_ten_cross_one.sumocfg",
        #              '--step-length', str(args.step_length),
        #              '--seed', str(args.sumo_seed)
        #              ])

        sumo = SumoSimulation(args.sumo_cfg_file, args.step_length, args.sumo_host, args.sumo_gui, args.client_order, args.sumo_seed, 8813, args.sumo_port)

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            print('current step:', traci.simulation.getCurrentTime())
    finally:
        traci.close()