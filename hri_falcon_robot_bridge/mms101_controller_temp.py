import socket
import socket
import time
import sys
import array
import numpy as np

# Sensor Constants
PROTOCOL_SPI = 0x01
SENSOR_MAP = {1: 0x01, 2: 0x02, 3: 0x04, 4: 0x08, 5: 0x10}


class MMS101Controller:
    """
    Temporary controller variant with EMA-based baseline (zero-drift suppression),
    contact hysteresis, leak recovery, plausibility masking, and deadbands.
    """

    def __init__(self, config):
        # Network/config (defensive defaults if keys missing)
        cfg = getattr(config, 'mms101', config)
        self.dest_ip = getattr(cfg, 'dest_ip', '192.168.0.10')
        self.dest_port = int(getattr(cfg, 'dest_port', 2000))
        self.src_port = int(getattr(cfg, 'src_port', 2001))
        self.measure_max = float(getattr(cfg, 'measure_max', 100.0))
        self.debug_mode = bool(getattr(cfg, 'debug', False))
        self.sensors = list(getattr(cfg, 'sensors', [1, 2, 3]))
        self.n_sensors = int(getattr(cfg, 'n_sensors', len(self.sensors)))

        # Baseline/EMA params
        self.warmup_time = float(getattr(cfg, 'warmup_time', 5.0))
        self.tau_warmup = float(getattr(cfg, 'tau_warmup', 2.0))
        self.tau_run = float(getattr(cfg, 'tau_run', 120.0))
        self.contact_enter_th = float(getattr(cfg, 'contact_enter_th', 0.5))  # N
        self.contact_exit_th = float(getattr(cfg, 'contact_exit_th', 0.3))  # N
        self.contact_hold_time = float(getattr(cfg, 'contact_hold_time', 1.0))
        self.outlier_delta_th = float(getattr(cfg, 'outlier_delta_th', 1e6))
        # Recovery/Leak params to prevent "stuck" after contact
        self.tau_contact_leak = float(getattr(cfg, 'tau_contact_leak', 10.0))  # s
        self.drift_recover_max = float(getattr(cfg, 'drift_recover_max', 3.0))  # N norm threshold

        # Plausibility and deadband params
        self.plaus_force_limit = float(getattr(cfg, 'plaus_force_limit', self.measure_max * 20.0))
        self.plaus_torque_limit = float(getattr(cfg, 'plaus_torque_limit', 20.0))
        self.invalid_consecutive = int(getattr(cfg, 'invalid_consecutive', 2))
        self.deadband_force = float(getattr(cfg, 'deadband_force', 0.2))
        self.deadband_torque = float(getattr(cfg, 'deadband_torque', 0.01))

        # State
        self.sockOpenFlag = 0
        self.destAddr = (self.dest_ip, self.dest_port)
        self.srcAddr = ('', self.src_port)
        self.sensorNo = self.select_sensors(self.sensors)
        self.sockOpen()

        self.baseline = np.zeros((self.n_sensors, 6), dtype=float)
        self.prev_raw = np.zeros((self.n_sensors, 6), dtype=float)
        self.have_prev = False
        self.in_contact = False
        self.last_contact_ts = None
        self.t_start = time.time()
        self._ready = False
        self.invalid_mask = np.zeros((self.n_sensors,), dtype=bool)
        self.invalid_count = np.zeros((self.n_sensors,), dtype=int)

        # Fixed factory offset (if any) can be added on top of EMA baseline
        self.factory_offset = np.zeros((self.n_sensors, 6), dtype=float)

        # Boot sequence
        self.offset = self.factory_offset.copy()  # compatibility
        self.cmdReset()
        self.cmdSelect()
        self.cmdBoot()
        # Wait until READY
        while True:
            status = self.cmdStatus()
            if status[4] == 0x03:
                break
            elif status[4] == 0x02:
                pass
            else:
                print('BOOT Error')
                sys.exit(1)

    def __del__(self):
        self.sockClose()

    # --- Socket/Protocol ---
    def sockOpen(self):
        self.sockDsc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sockDsc.bind(self.srcAddr)
        self.sockOpenFlag = 1

    def sockClose(self):
        if self.sockOpenFlag:
            self.cmdStop()
            self.sockDsc.close()
            self.sockOpenFlag = 0

    def recvData(self, rcvLen):
        data = self.sockDsc.recv(rcvLen)
        if self.debug_mode:
            print(data.hex())
        return data

    def send_cmd(self, cmd):
        sendSz = self.sockDsc.sendto(array.array('B', cmd), self.destAddr)
        if sendSz != len(cmd):
            print(f"Error: Command send {cmd}")

    def cmdStart(self):
        self.send_cmd([0xF0])
        return self.recvData(2)

    def cmdData(self):
        self.send_cmd([0xE0])
        return self.recvData(100)

    def cmdRestart(self):
        self.send_cmd([0xC0])
        return self.recvData(2)

    def cmdBoot(self):
        self.send_cmd([0xB0])
        return self.recvData(100)

    def cmdStop(self):
        self.send_cmd([0xB2])
        return self.recvData(2)

    def cmdReset(self):
        self.send_cmd([0xB4])
        return self.recvData(2)

    def cmdStatus(self):
        self.send_cmd([0x80])
        return self.recvData(6)

    def cmdSelect(self):
        self.send_cmd([0xA0, PROTOCOL_SPI, self.sensorNo])
        return self.recvData(2)

    def cmdVersion(self):
        self.send_cmd([0xA2])
        return self.recvData(8)

    def select_sensors(self, sensor_list):
        selected = 0
        for sens in sensor_list:
            if sens in SENSOR_MAP:
                selected |= SENSOR_MAP[sens]
        return selected

    # --- Measurement ---
    def run(self, period):
        # Ensure measuring state
        self.cmdStart()
        time.sleep(0.0001)

        rData = self.cmdData()
        if len(rData) != 100 or rData[0] != 0x00:
            return np.zeros((self.n_sensors, 6), dtype=float)

        # Parse packet
        intervalTime = (rData[6] << 24) + (rData[7] << 16) + (rData[8] << 8) + rData[9]
        dt = max(intervalTime / 1_000_000.0, 1e-4)

        mms101data = np.zeros((self.n_sensors, 6), dtype=float)
        for sens in range(self.n_sensors):
            for axis in range(6):
                base_index = (sens * 18) + (axis * 3) + 10
                val = (rData[base_index] << 16) + (rData[base_index + 1] << 8) + rData[base_index + 2]
                if val >= 0x00800000:
                    val -= 0x1000000
                mms101data[sens, axis] = val / (1000.0 if axis < 3 else 100000.0)

        # Outlier rejection on raw delta
        if self.have_prev:
            delta = np.abs(mms101data - self.prev_raw)
            if np.any(delta > self.outlier_delta_th):
                corrected = mms101data - (self.baseline + self.factory_offset)
                self.prev_raw = mms101data
                return corrected
        self.prev_raw = mms101data
        self.have_prev = True

        # Per-sensor plausibility check (absolute + relative to peers)
        f_abs = np.abs(mms101data[:, :3])
        t_abs = np.abs(mms101data[:, 3:])
        force_over = np.any(f_abs > self.plaus_force_limit, axis=1)
        torque_over = np.any(t_abs > self.plaus_torque_limit, axis=1)
        f_norms = np.linalg.norm(mms101data[:, :3], axis=1)
        # Use median of sane values (> small epsilon) for relative check
        eps_rel = 0.05
        sane = f_norms > eps_rel
        median_norm = np.median(f_norms[sane]) if np.any(sane) else 0.0
        relative_outlier = np.zeros_like(force_over)
        if median_norm > 0.0:
            relative_outlier = f_norms > max(5.0, 50.0 * median_norm)
        current_invalid = force_over | torque_over | relative_outlier
        # Update invalid counters/mask (require consecutive hits)
        self.invalid_count[current_invalid] += 1
        self.invalid_count[~current_invalid] = np.maximum(self.invalid_count[~current_invalid] - 1, 0)
        self.invalid_mask = self.invalid_count >= self.invalid_consecutive

        now = time.time()
        elapsed = now - self.t_start

        # During warmup, always build baseline and output zeros
        if elapsed < self.warmup_time:
            alpha_warm = 1.0 - np.exp(-dt / max(self.tau_warmup, 1e-6))
            # Skip invalid sensors in baseline update
            upd = (1.0 - alpha_warm) * self.baseline + alpha_warm * mms101data
            self.baseline[~self.invalid_mask] = upd[~self.invalid_mask]
            self.in_contact = False
            self.last_contact_ts = None
            self._ready = False
            return np.zeros_like(mms101data)

        # After warmup: detect contact based on force relative to current baseline
        rel_forces = mms101data - self.baseline
        f_rel_norm = np.linalg.norm(rel_forces[:, :3], axis=1)

        # hysteresis on relative forces
        if not self.in_contact and np.any(f_rel_norm > self.contact_enter_th):
            self.in_contact = True
            self.last_contact_ts = now
        elif self.in_contact:
            if np.all(f_rel_norm < self.contact_exit_th):
                if self.last_contact_ts is None:
                    self.last_contact_ts = now
                if (now - self.last_contact_ts) >= self.contact_hold_time:
                    self.in_contact = False
                    self.last_contact_ts = None
            else:
                self.last_contact_ts = now

        # Baseline update
        alpha_run = 1.0 - np.exp(-dt / max(self.tau_run, 1e-6))
        if not self.in_contact:
            # Normal update when not in contact (skip invalid sensors)
            upd = (1.0 - alpha_run) * self.baseline + alpha_run * mms101data
            self.baseline[~self.invalid_mask] = upd[~self.invalid_mask]
        else:
            # Leak update to recover from slow drift when contact seems small/persistent
            max_rel = float(np.max(f_rel_norm)) if f_rel_norm.size > 0 else 0.0
            if max_rel < self.drift_recover_max:
                alpha_leak = 1.0 - np.exp(-dt / max(self.tau_contact_leak, 1e-6))
                upd = (1.0 - alpha_leak) * self.baseline + alpha_leak * mms101data
                self.baseline[~self.invalid_mask] = upd[~self.invalid_mask]

        corrected = mms101data - (self.baseline + self.factory_offset)
        # Zero out invalid sensors entirely
        if np.any(self.invalid_mask):
            corrected[self.invalid_mask, :] = 0.0
        # Apply configurable deadband (forces vs torques)
        eps = np.array([
            self.deadband_force, self.deadband_force, self.deadband_force,
            self.deadband_torque, self.deadband_torque, self.deadband_torque
        ], dtype=float)
        mask = np.abs(corrected) < eps
        corrected = np.where(mask, 0.0, corrected)
        self._ready = True
        return corrected

    def is_ready(self) -> bool:
        return self._ready
