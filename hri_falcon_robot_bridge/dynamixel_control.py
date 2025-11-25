import sys
import tty
import termios
from dynamixel_sdk import *  # Uses Dynamixel SDK library
import numpy as np  

class DynamixelControl:
    def __init__(self, config, max_current_pos=500, max_current_neg=500):
        self.cfg = config
        self.max_current_pos = max_current_pos
        self.max_current_neg = max_current_neg
        self.portHandler = PortHandler(self.cfg.device_name)
        self.packetHandler = PacketHandler(self.cfg.protocol_version)
        # Only save terminal settings if stdin is a TTY
        try:
            if sys.stdin.isatty():
                self.old_settings = termios.tcgetattr(sys.stdin.fileno())
            else:
                self.old_settings = None
        except:
            self.old_settings = None

    def getch(self):
        if self.old_settings is None:
            # Not a TTY, can't use getch
            return None
        fd = sys.stdin.fileno()
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self.old_settings)
        return ch

    def connect(self):
        
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            raise Exception("Failed to open the port")

        if self.portHandler.setBaudRate(self.cfg.baudrate):
            print("Succeeded to change the baudrate")
        else:
            raise Exception("Failed to change the baudrate")
        self.disable_torque()
        for id in self.cfg.ids:
            cur_Operating_mode = self.get_operating_mode(id)
            if cur_Operating_mode != self.cfg.control_modes.default_mode:
                self.set_operating_mode_one(id, self.cfg.control_modes.default_mode)
        
        self.enable_torque() # for dynamaixel operation
        # self.disable_torque() # for torque input operation
    def set_delaytime(self, delaytime): # 0~254
        for id in self.cfg.ids:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, id, self.cfg.control_table.addr_delaytime, delaytime)
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception(f"Failed to set delaytime: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"Delaytime set for Dynamixel ID {id}")

    def set_baudrate(self, baudrate): # 0 ~7
        for id in self.cfg.ids:
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                self.portHandler, id, self.cfg.control_table.addr_baudrate, baudrate)
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception(f"Failed to set baudrate: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"Baudrate set for Dynamixel ID {id}")
                            

    def set_operating_mode_all(self, mode):
        for id in self.cfg.ids:
            # mode = self.cfg.control_modes.default_mode
            addr_operating_mode = self.cfg.control_table.addr_operating_mode
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, id, addr_operating_mode, mode)
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception(f"Failed to set operating mode: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"Dynamixel ID {id} set to mode {mode}")
    def set_operating_mode_one(self, id, mode ):
        # mode = self.cfg.control_modes.default_mode
        addr_operating_mode = self.cfg.control_table.addr_operating_mode
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, id, addr_operating_mode, mode)
        if dxl_comm_result != COMM_SUCCESS:
            raise Exception(f"Failed to set operating mode: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
        else:
            print(f"Dynamixel ID {id} set to mode {mode}")

    def get_operating_mode_all(self) -> int:
        addr_operating_mode = self.cfg.control_table.addr_operating_mode
        for id in self.cfg.ids:
            dxl_current_state, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx( # state -> mode depenedent
                self.portHandler, id, addr_operating_mode)
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception(f"Failed to get operating mode: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"Operating mode for Dynamixel ID {id} is {dxl_current_state}")
        return dxl_current_state
    
    def get_operating_mode(self, id) -> int:
        addr_operating_mode = self.cfg.control_table.addr_operating_mode
        dxl_current_state, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx( # state -> mode depenedent
            self.portHandler, id, addr_operating_mode)
        if dxl_comm_result != COMM_SUCCESS:
            raise Exception(f"Failed to get operating mode: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
        else:
            print(f"Operating mode for Dynamixel ID {id} is {dxl_current_state}")
        return dxl_current_state
    
    def enable_torque(self):
        addr_torque_enable = self.cfg.control_table.addr_torque_enable
        for id in self.cfg.ids:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, id, addr_torque_enable, self.cfg.torque_enable)
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception(f"Failed to enable torque: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"Torque enabled for Dynamixel ID {id}")

    def dynamixel_pos_to_deg(self, pos):
        return np.array(pos) * 0.0879120879
    
    def dynamixel_pos_to_rad(self, pos) -> np.array:
        return np.array(pos) * 0.001533981

    
    def get_joint_velocities(self) :
        ADDR_PRESENT_VELOCITY = self.cfg.control_table.ADDR_PRESENT_VELOCITY
        dxl_present_velocities = []
        for id in self.cfg.ids:
            dxl_present_velocity, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                self.portHandler, id, ADDR_PRESENT_VELOCITY)
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception(f"Failed to get velocity: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
            # else:
            #     print(f"Velocity for Dynamixel ID {id} is {dxl_present_velocity}")
            if dxl_present_velocity > 0x7fffffff:
                dxl_present_velocity = dxl_present_velocity - 4294967296
            dxl_present_velocities.append(dxl_present_velocity)
        
        return np.array(dxl_present_velocities)
    
    def get_joint_positions(self,type=None) -> list[int]:
        ADDR_PRESENT_POSITION = self.cfg.control_table.ADDR_PRESENT_POSITION
        dxl_present_positions = []

        for id in self.cfg.ids:
            dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                self.portHandler, id, ADDR_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception(f"Failed to get position: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                # print(f"Position for Dynamixel ID {id} is {dxl_present_position}")
                pass
            dxl_present_positions.append(dxl_present_position*0.1*10)
        if type=="rad": # 1.57 rad = 90 deg
            return self.dynamixel_pos_to_rad(dxl_present_positions)
        elif type=="deg":
            return self.dynamixel_pos_to_deg(dxl_present_positions)
        else :
            return dxl_present_positions    

    def set_joint_positions(self, goal_pos) :
        ADDR_GOAL_POSITION = self.cfg.control_table.ADDR_GOAL_POSITION
        ids = self.cfg.ids
        dxl_present_positions = []
        
        for idx,id in enumerate(ids):
            # self.set_operating_mode_one(id,self.cfg.control_modes.position_control_mode)
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                self.portHandler, id, ADDR_GOAL_POSITION, goal_pos[idx])
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception(f"Failed to set position: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
                print("works?")
            elif dxl_error != 0:
                raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
            # print("works!")

        
    
    def disable_torque(self):
        addr_torque_enable = self.cfg.control_table.addr_torque_enable
        for id in self.cfg.ids:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, id, addr_torque_enable, self.cfg.torque_disable)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"Failed to disable torque: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"Torque disabled for Dynamixel ID {id}")
                
    def test_torqueinput(self, input_torque, log=False):
        ADDR_GOAL_CURRENT = self.cfg.control_table.ADDR_GOAL_CURRENT
        for idx,id in enumerate(self.cfg.ids):
            print(idx)
            if abs(input_torque.any()) >= 10 :
                print(f"Torque input is too high")
                return
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                self.portHandler, id, ADDR_GOAL_CURRENT, input_torque[idx])
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception(f"Failed to write torque: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                if log:
                    print(f"Torque written for Dynamixel ID {id}")
    
    def test_torqueinput_one(self, id, input_torque, log = False):
        ADDR_GOAL_CURRENT = self.cfg.control_table.ADDR_GOAL_CURRENT
        print(f"\n\n{id}\n\n")
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, id, ADDR_GOAL_CURRENT, input_torque)
        if dxl_comm_result != COMM_SUCCESS:
            raise Exception(f"Failed to write torque: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
        else:
            if log:
                print(f"Torque written for Dynamixel ID {id}")
    def close_port(self):
        self.disable_torque()
        self.portHandler.closePort()
        print("Port closed")

    def test_torqueinputs(self, ids, input_torque, log=False):
        ADDR_GOAL_CURRENT = self.cfg.control_table.ADDR_GOAL_CURRENT
        for idx,id in enumerate(ids):
            # print(idx)
            # Check against asymmetric limits
            if input_torque[idx] > self.max_current_pos or input_torque[idx] < -self.max_current_neg:
                print(f"Torque input out of range: {input_torque[idx]} (limits: -{self.max_current_neg} to +{self.max_current_pos})")
                return
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                self.portHandler, id, ADDR_GOAL_CURRENT, input_torque[idx])
            if dxl_comm_result != COMM_SUCCESS:
                raise Exception(f"Failed to write torque: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                raise Exception(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                if log:
                    print(f"Torque written for Dynamixel ID {id}")