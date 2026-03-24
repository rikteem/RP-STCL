# -*- coding: utf-8 -*-
"""
RP_Lock_os2.py  —  OS 2.x port of RP_Lock.py
=============================================
Original author : epultinevicius (LangenGroup)
Port            : OS 2.x / rp module (SWIG wrapper around _rp_py C extension)

What changed vs the original RP_Lock.py
----------------------------------------
1.  Import: `from redpitaya.overlay.mercury import mercury as overlay`
    replaced by `import rp` + `rp.rp_Init()`.

2.  Class `_GenProxy`  (NEW)
    A thin wrapper that gives the rest of the code the same attribute-assignment
    interface as the mercury gen objects:
        gen.offset    = x  →  rp.rp_GenOffset(ch, x)
        gen.amplitude = x  →  rp.rp_GenAmp(ch, x)
        gen.reset()        →  rp.rp_GenResetChannelSM(ch)
        gen.start_trigger()→  rp.rp_GenTriggerOnly(ch)
    Without this proxy every `gen_ramp.offset = val` write in RP_Lock and
    RP_Server would need editing — this keeps those classes untouched.

3.  Class `_GpioProxy`  (NEW)
    Wraps rp.rp_DpinGetState() so that ext_trig1.read() / ext_trig2.read()
    still return a plain bool, matching the original mercury gpio contract.

4.  Class `RP`  (REWRITTEN)
    Hardware abstraction layer — the only class that touches the rp module.
    Trigger source: RP_TRIG_SRC_AWG_PE (confirmed working on OS 2.x board
    without any external loopback cable — AWG fires ADC internally).
    Burst period: computed in microseconds from sample count and clock rate.
    VALIDATE tag marks the period formula that should be verified on hardware.

5.  mode="lock" and mode="monitor" in RP.__init__
    Raise NotImplementedError until Phase 1b/1c are complete.

6.  All other classes (Receiver, reaction_loop, RP_Server, PID, RP_Lock)
    are VERBATIM copies of the originals — zero logic changes.

Hardware assumptions (confirmed by board_test.py on rp-f0efbb, OS 2.07):
    - ADC_BUFFER_SIZE = DAC_BUFFER_SIZE = 16384
    - RP_TRIG_SRC_AWG_PE fires immediately when generator is triggered
    - RP_DEC_16 enum accepted and reads back as 16
    - GPIO DIO0_P / DIO0_N readable as inputs

VALIDATE items (need hardware confirmation with signals connected):
    - Burst period formula: period_us = int(2 * dec * N_gen * 8e-3)
      At 125 MHz, 1 sample = 8 ns = 8e-3 µs.
      For dec=16, N_gen=16384: period_us = 4194 µs ≈ 4.2 ms per scan cycle.
    - Ramp amplitude 0.5 V peak — verify piezo scan range is appropriate.
    - Trigger delay = N (all post-trigger) — verify waveform alignment.
"""

import rp
import socket, selectors, traceback, libserver
import numpy as np
from time import perf_counter, sleep
from peak_finders import SG_array, peak_finders
from copy import deepcopy


# ---------------------------------------------------------------------------
# Decimation integer → rp enum lookup table
# Covers all values that set_dec() might receive from settings or the client.
# ---------------------------------------------------------------------------
_DEC_MAP = {
    1:     rp.RP_DEC_1,
    2:     rp.RP_DEC_2,
    4:     rp.RP_DEC_4,
    8:     rp.RP_DEC_8,
    16:    rp.RP_DEC_16,
    32:    rp.RP_DEC_32,
    64:    rp.RP_DEC_64,
    128:   rp.RP_DEC_128,
    256:   rp.RP_DEC_256,
    512:   rp.RP_DEC_512,
    1024:  rp.RP_DEC_1024,
    2048:  rp.RP_DEC_2048,
    4096:  rp.RP_DEC_4096,
    8192:  rp.RP_DEC_8192,
    16384: rp.RP_DEC_16384,
    32768: rp.RP_DEC_32768,
    65536: rp.RP_DEC_65536,
}

# Channel constants — named clearly to avoid confusion with osc channel indices
_CH = {
    0: rp.RP_CH_1,   # gen index 0 (gen_trig) → Out1 → RP_CH_1
    1: rp.RP_CH_2,   # gen index 1 (gen_ramp) → Out2 → RP_CH_2
}

# ADC channel constants — osc index 0 → In1, osc index 1 → In2
_ACQ_CH = {
    0: rp.RP_CH_1,   # osc[0] → In1 (cavity transmission)
    1: rp.RP_CH_2,   # osc[1] → In2 (not used for cavity mode)
}


# ---------------------------------------------------------------------------
# _GenProxy
# ---------------------------------------------------------------------------
class _GenProxy:
    """
    Proxy object that gives the rest of the code the same attribute-assignment
    interface as the mercury gen objects, while dispatching to rp_Gen*() calls.

    Attributes supported (matching what RP_Lock / RP_Server write):
        .offset     → rp.rp_GenOffset(ch, value)
        .amplitude  → rp.rp_GenAmp(ch, value)

    Methods supported:
        .reset()          → rp.rp_GenResetChannelSM(ch)
        .start_trigger()  → rp.rp_GenTriggerOnly(ch)
    """

    def __init__(self, gen_index):
        """
        Parameters
        ----------
        gen_index : int
            0 for gen_trig (Out1 / RP_CH_1), 1 for gen_ramp (Out2 / RP_CH_2).
        """
        self._ch = _CH[gen_index]
        self._offset = 0.0
        self._amplitude = 0.0

    # --- attribute writes intercepted via __setattr__ ---

    def __setattr__(self, name, value):
        if name == "offset":
            rp.rp_GenOffset(self._ch, float(value))
            object.__setattr__(self, "_offset", float(value))
        elif name == "amplitude":
            rp.rp_GenAmp(self._ch, float(value))
            object.__setattr__(self, "_amplitude", float(value))
        else:
            object.__setattr__(self, name, value)

    # --- attribute reads ---

    @property
    def offset(self):
        return self._offset

    @property
    def amplitude(self):
        return self._amplitude

    # --- methods ---

    def reset(self):
        rp.rp_GenResetChannelSM(self._ch)

    def start_trigger(self):
        rp.rp_GenTriggerOnly(self._ch)


# ---------------------------------------------------------------------------
# _GpioProxy
# ---------------------------------------------------------------------------
class _GpioProxy:
    """
    Proxy object so that ext_trig1.read() / ext_trig2.read() return a plain
    bool, matching the mercury gpio contract used in check_gpio_ext_trig().

    Original:  fpga.gpio("p", 0, "in").read()  →  bool
    OS 2.x:    rp.rp_DpinGetState(pin)          →  (retcode, RP_HIGH|RP_LOW)
    """

    def __init__(self, pin):
        """
        Parameters
        ----------
        pin : rp pin constant
            e.g. rp.RP_DIO0_P or rp.RP_DIO0_N
        """
        self._pin = pin
        rp.rp_DpinSetDirection(pin, rp.RP_IN)

    def read(self):
        _ret, state = rp.rp_DpinGetState(self._pin)
        return state == rp.RP_HIGH


# ---------------------------------------------------------------------------
# Receiver  (VERBATIM from original)
# ---------------------------------------------------------------------------
class Receiver:
    def __init__(self, addr, action_dict={}):
        self.sel = selectors.DefaultSelector()
        self.addr = addr
        self.action_dict = action_dict
        self.iteration = None

    def accept_wrapper(self, sock, stop=True):
        """
        a small wrapper to call whenever a command is received. The message
        class then handles the rest according to the

        Parameters
        ----------
        sock : socket
            the socket which the communication relies on.
        """

        conn, addr = sock.accept()  # Should be ready to read
        # print("Accepted connection from {}".format(addr))
        conn.setblocking(False)  # non-blocking, to keep the lock running!
        # print(conn)
        message = libserver.Message(
            self.sel, conn, addr, action_dict=self.action_dict, stop=stop
        )  # initialize the message object
        self.sel.register(
            conn, selectors.EVENT_READ, data=message
        )  # register in selector

    def setup_server(self, loop=False):
        """
        Starts the socket connection and initializes the event loop. During that
        loop, commands are avaited. Those are strings which may include actions
        with values that can be queried to the called functions.

        Returns
        -------
        None.

        """
        self.sel = selectors.DefaultSelector()  # reinitialize the selector!
        # setup the socket to listen to external commands!
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # IPv4 TCP socket
        # Avoid bind() exception: OSError: [Errno 48] Address already in use
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.listen()
        print("Listening on {}".format(self.addr))
        self.loop = loop
        # start by accepting a socket connection! Even though we are using selectors,
        # we only deal with one connection at a time! The communication framework
        # in the message class is used to cleanly deal with the messaging.
        if loop:
            print("waiting to accept socket ...")
            self.accept_wrapper(self.sock, stop=False)
            self.sock.setblocking(False)  # avoid blocking while waiting for commands
        else:
            self.sock.setblocking(False)  # avoid blocking while waiting for commands
            self.sel.register(self.sock, selectors.EVENT_READ, data=None)
        self.server_running = True

    def start_server(self):
        # start the actual event loop!
        try:
            while self.server_running:
                # print('hi')
                sleep(1e-4)
                # if defined, run the iteration!
                if self.iteration != None:
                    self.iteration()
                if self.loop:
                    events = self.sel.select(
                        timeout=1e-4
                    )  # check the handled socket connections
                else:
                    events = self.sel.select(timeout=None)
                for key, mask in events:
                    if (not self.loop) and (key.data == None):
                        self.accept_wrapper(key.fileobj)
                    else:
                        message = key.data
                        try:
                            message.process_events(
                                mask
                            )  # initially: event = read --> await command. if command read out, carry out command and write response!
                        except Exception:
                            print(
                                "Main: Error: Exception for {}:\n {}".format(
                                    message.addr, traceback.format_exc()
                                )
                            )
                            message.close()

        except Exception as e:
            print(str(e) + "Caught keyboard interrupt, exiting")
        finally:
            self.sel.close()

    def stop(self, query):
        # Some functions might take arguments, so by definition an argument is expected for all functions.
        # set boolean variable to False to stop server loop
        self.server_running = False
        self.sock.close()
        return "Stopped!"


# ---------------------------------------------------------------------------
# reaction_loop  (VERBATIM from original)
# ---------------------------------------------------------------------------
class reaction_loop(Receiver):  #### USE THIS FOR THE LOCKING LOOP
    def __init__(self, addr):
        action_dict = {
            "stop": self.stop,  # By default, stop function must be provided!
        }
        Receiver.__init__(self, addr, action_dict=action_dict)
        self.iteration = None  # to be defined in children class
        self.var_dict = {}  # a dictionary of variables (?)
        self.r_buff = ""  # read buffer
        self.setup = None  # optional setup function, set to None initially
        # dont start immediatley!

    def start_loop(self):
        print("starting loop")
        self.setup_server(loop=True)
        print("server connection established! Calling loop setup!")
        # call the setup prior to starting the event loop, if defined
        if self.setup != None:
            self.setup()
        print("finished setup, starting server!")
        # start the event loop!
        self.start_server()


# ---------------------------------------------------------------------------
# RP_Server  (VERBATIM from original)
# ---------------------------------------------------------------------------
class RP_Server(Receiver):  # handles socket communication from redpitaya side
    def __init__(self, host, port, port2, RP_mode="scan"):
        Receiver.__init__(
            self, (host, port)
        )  # initialize the receiver which handles the event_loop
        if RP_mode == "monitor":
            self.RP_mode = "lock"
        else:
            self.RP_mode = RP_mode
        self.lock = RP_Lock((host, port2), mode=RP_mode)
        self.action_dict = {
            "acquire": self.action_acquire,
            "acquire_ch": self.action_acquire_ch,
            "echo": self.action_echo,
            "count": self.action_count,
            "close": self.action_close,
            "acquire_ch_n": self.action_acquire_ch_n,
            "monitor": self.action_monitor,
            "acquire_peaks_ch": self.action_acquire_peaks_ch,
            "update_settings": self.action_update_settings,
            "start_lock":      self.action_start_lock,
            "start_lock2":     self.action_start_lock,
            "start_lock3":     self.action_start_lock,
            "start_scan":      self.action_start_scan,
            "stop_scan":       self.action_stop_scan,
            "set_scan_output": self.action_set_scan_output,
            "start_scan":  self.action_start_scan,
            "stop_scan":   self.action_stop_scan,
            "test": self.action_test,
            "set": self.action_set,
            "stop": self.stop,
            "set_dec": self.action_set_dec,
            "acquire_errs": self.action_acquire_errs,
            "set_peakfinder": self.action_set_peakfinder,
        }

    def action_set(self, query):
        query_list = query.split("|")  # query contains to strings split by |
        mod = query_list[0]
        value = query_list[1]

    def action_set_dec(self, query):
        self.lock.set_dec(query)
        print("Set decimation to {}".format(query))
        return "Set decimation to {}".format(query)

    def action_test(self, query):  # has been used for testing timings
        times = []
        range = [0, 8000]
        for i in range(1000):
            dat = self.lock.acquire_ch(0)
            t0 = perf_counter()
            # np.argmax(dat[:8000])
            x, y = dat[range[0] : range[-1]], dat[range[0] : range[-1]]
            t1 = perf_counter() - t0
            times.append(t1)
        return times

    def action_update_settings(self, query):
        self.lock.update_settings(query)
        return "Lock settings updated!"

    def action_start_lock(self, query):
        if self.RP_mode in ["scan", "lock"]:
            print("starting lock!")
            self.lock.start()  # starts the lock --> after that method is done, the lock is finished
            # the below code is executed after the lock is finished!
            print("lock stopped")
            for key, val in self.lock.settings.items():  # reset all PIDs
                val["PID"].reset()
            print("PIDs reset")
            self.lock.gen_ramp.offset = 0.0  # reset ramp offset to 0
            if self.RP_mode == "lock":
                self.lock.gen_trig.offset = 0.0
            if self.RP_mode == "scan":
                self.lock.scan_output_disable()  # silence Out2 when lock stops
            print("outputs reset")
            return "Done!"

    def action_start_scan(self, query):
        """Enable Out2 waveform then run a bare acquisition loop on port 5065.

        Mirrors action_monitor() exactly — a lightweight reaction_loop that
        just calls acquire_ch(0) each iteration. Does NOT call lock.start()
        so it never crashes on empty settings (the chicken-and-egg problem
        where settings cannot be sent until the scan is running).

        Sequence:
          1. scan_output_enable()  — Out2 triangle fires immediately
          2. reaction_loop opens port 5065 — PC sets loop_running = True
          3. Loop iterates: acquire_ch(0) each cycle
          4. "stop" command received → loop exits
          5. scan_output_disable() — Out2 goes silent
        """
        if self.RP_mode != "scan":
            return "action_start_scan: only valid in scan mode"

        # Step 1 — enable Out2 synchronously before loop machinery starts
        # query may carry amplitude/offset/waveform overrides as a dict
        amplitude = None
        offset    = None
        waveform  = None
        if isinstance(query, dict):
            amplitude = query.get("amplitude", None)
            offset    = query.get("offset",    None)
            waveform  = query.get("waveform",  None)
            period_ms = query.get("period_ms", None)
        if period_ms is not None:
            self.lock.set_scan_period(float(period_ms))
        print("action_start_scan: enabling Out2")
        self.lock.scan_output_enable(waveform=waveform,
                                     amplitude=amplitude,
                                     offset=offset)

        # Step 2 — bare reaction_loop (identical structure to action_monitor)
        addr = (self.addr[0], 5065)
        rl = reaction_loop(addr)
        rl.var_dict["i"] = 0

        def iteration():
            self.lock.acquire_ch(0)
            rl.var_dict["i"] += 1

        def rl_update_settings(settings):
            self.lock.update_settings(settings)
            return "settings updated"

        def rl_set_scan_output(query):
            """Handle set_scan_output on port 5065 (the loop socket).
            Without this entry in rl.action_dict the command arrives but gets
            no handler, so libserver never sends a response and the PC-side
            send() blocks forever — causing the >1 min hang.
            """
            if not isinstance(query, dict):
                return "set_scan_output: query must be a dict"
            if "period_ms" in query:
                self.lock.set_scan_period(float(query["period_ms"]))
            self.lock.scan_output_enable(
                waveform=query.get("waveform",   None),
                amplitude=query.get("amplitude", None),
                offset=query.get("offset",       None))
            return "scan output updated"

        rl.iteration = iteration
        rl.action_dict["update_settings"] = rl_update_settings
        rl.action_dict["set_dec"]         = self.action_set_dec
        rl.action_dict["set_scan_output"] = rl_set_scan_output  # fixes hang

        print("action_start_scan: loop starting on port 5065")
        rl.start_loop()   # blocks until "stop" received on port 5065

        # Step 5 — clean up after loop exits
        print("action_start_scan: {} iterations done, disabling Out2".format(
            rl.var_dict["i"]))
        self.lock.gen_ramp.offset = 0.0
        self.lock.scan_output_disable()
        return "Done"

    def action_stop_scan(self, query):
        """Disable Out2 immediately — safe to call even if loop is not running."""
        self.lock.scan_output_disable()
        return "Out2 disabled"

    def action_start_scan(self, query):
        """Enable Out2 waveform then run a bare acquisition loop on port 5065.

        query may be a dict with keys: amplitude, offset, waveform.
        Does NOT call lock.start() — avoids crash on empty settings.
        """
        if self.RP_mode != "scan":
            return "action_start_scan: only valid in scan mode"
        amplitude = None
        offset    = None
        waveform  = None
        if isinstance(query, dict):
            amplitude = query.get("amplitude", None)
            offset    = query.get("offset",    None)
            waveform  = query.get("waveform",  None)
            period_ms = query.get("period_ms", None)
        if period_ms is not None:
            self.lock.set_scan_period(float(period_ms))
        print("action_start_scan: enabling Out2")
        self.lock.scan_output_enable(waveform=waveform,
                                     amplitude=amplitude,
                                     offset=offset)
        addr = (self.addr[0], 5065)
        rl = reaction_loop(addr)
        rl.var_dict["i"] = 0

        def iteration():
            self.lock.acquire_ch(0)
            rl.var_dict["i"] += 1

        def rl_update_settings(settings):
            self.lock.update_settings(settings)
            return "settings updated"

        rl.iteration = iteration
        rl.action_dict["update_settings"] = rl_update_settings
        rl.action_dict["set_dec"]         = self.action_set_dec
        print("action_start_scan: loop starting on port 5065")
        rl.start_loop()
        print("action_start_scan: {} iterations done, disabling Out2".format(
            rl.var_dict["i"]))
        self.lock.gen_ramp.offset = 0.0
        self.lock.scan_output_disable()
        return "Done"

    def action_stop_scan(self, query):
        """Disable Out2 immediately. Safe even if loop is not running."""
        self.lock.scan_output_disable()
        return "Out2 disabled"

    def action_set_scan_output(self, query):
        """Update Out2 amplitude/offset/waveform/period while scan is running.

        query: dict with any of: "amplitude", "offset", "waveform", "period_ms"
        """
        if not isinstance(query, dict):
            return "set_scan_output: query must be a dict"
        if "period_ms" in query:
            self.lock.set_scan_period(float(query["period_ms"]))
        self.lock.scan_output_enable(
            waveform=query.get("waveform",   None),
            amplitude=query.get("amplitude", None),
            offset=query.get("offset",       None))
        return "scan output updated"

    def action_close(self, query):
        self.server_running = False
        self.sock.close()
        print("closed!")
        return "closed!"

    def action_echo(self, query):
        print("{}".format(query))
        return "{}".format(query)

    def action_count(self, query):  # testing purpose
        answer = "\n"
        try:
            for i in range(int(query)):
                answer += str(i + 1)
                if i < int(query) - 1:
                    answer += "\n"
        except:
            answer = "Error: {} is not an integer!".format(query)
        return answer

    def action_acquire(self, query):
        print("Hi!")
        self.lock.acquire()
        data = self.lock.acquisition
        return data.tolist()

    def action_acquire_ch(self, query):  # this is used to monitor the cavity signal
        ch = int(query)
        data = self.lock.acquire_ch(ch)
        duration = self.lock.times[
            -1
        ]  # instead of the full time trace, just give the last time value! the first one is always 0 and the number of data points is always the same
        return [duration, data.tolist()]

    def action_acquire_ch_n(self, query):
        dat_list = []
        ch, n = int(query[0]), int(
            query[2:]
        )  # syntax example: query =  '1|100' for 100 traces on ch1 (in2)
        if n <= 100:
            pass
        else:
            print("max 100 data sets!")
            n = 100
        t0 = perf_counter()
        for i in range(n):
            dat_list.append(self.lock.acquire_ch(ch))
        print(perf_counter() - t0)
        dat_arr = np.stack(dat_list, axis=0)
        # return dat_list
        return dat_arr.tolist()

    def action_acquire_peaks_ch(self, query):
        # acquire the peak on a certain channel --> range must be given in query!
        # split the query in order to obtain ch, ranges(range from R1 to R2)
        query_list = query.split("|")  # seperator: '|'
        ch = int(query_list[0])
        acq = self.lock.acquire_ch(ch)  # obtain data trace
        peaks = []
        # Remaining query may contain a bunch of ranges --> two indices separated by ','
        for R in query_list[1:]:  # iterate through each range string
            R1, R2 = R.split(",")
            try:
                P, FWHM = self.lock.acquire_peaks(
                    [int(R1), int(R2)]
                )  # retrieve the peak
                peaks.append(P[0])
            except:
                peaks.append(None)
        return peaks  # return the list of peaks!

    def action_acquire_errs(self, query):
        if self.lock.FSR_ref == None:
            self.lock.init_FSR_ref()  # first, save the FSR for proper error calculation!
        self.lock.update_pos()
        if self.lock.skipped:
            return "skipped"
        else:
            for key in self.lock.settings:
                self.lock.update_err(key)
            return self.lock.errs

    def action_set_peakfinder(self, query):
        # query is a dictionary
        name = query.pop("name")
        self.peak_finder = name
        if name[:2] == "SG":  # if savitzky golay filter is involved
            self.SG_m = SG_array(**query)  # calculate conv. matrix
        return "updated peakfinder {fname}".format(fname=name)

    def action_monitor(self, query):
        addr = (self.addr[0], 5065)
        rl = reaction_loop(addr)
        rl.var_dict["dat_list"] = []
        rl.var_dict["i"] = 0
        rl.var_dict["j"] = 0
        rl.var_dict["settings"] = {}
        rl.var_dict2 = {}
        rl.var_dict2["j"] = 0
        t0 = perf_counter()
        i = 0

        def give(q):
            print("giving data")
            rl.var_dict["dat_list"].append(self.lock.acquisition)
            return "giving data!"

        def update_settings(settings):
            for key, val_dict in settings.items():  # iterate through the lasers
                if key not in rl.var_dict["settings"].keys():
                    rl.var_dict["settings"][
                        key
                    ] = {}  # if settings dont exist yet, initialize dictionary
                for (
                    val_key,
                    val_val,
                ) in val_dict.items():  # iterate through the laser-settings
                    if val_key == "PID":  # for PIDs, only the gains are sent!
                        pid = PID(
                            **val_val
                        )  # gains stored in yet another dictionary...
                        rl.var_dict["settings"][key][val_key] = pid
                    else:
                        rl.var_dict["settings"][key][
                            val_key
                        ] = val_val  # in each other case, the settings are transferred directly
                print(rl.var_dict["settings"][key])
                if rl.var_dict["settings"][key]["enabled"] == False:
                    rl.var_dict["settings"].pop(
                        key
                    )  # if the respective laser lock is not enabled, remove it from the dictionary!
            print("updated settings: {}".format(rl.var_dict["settings"]))
            return "updated settings!"

        def iteration():
            self.lock.acquire_ch(0)
            rl.var_dict["i"] += 1

        rl.action_dict["give"] = give
        rl.action_dict["update_settings"] = update_settings
        rl.action_dict["set_dec"] = self.action_set_dec
        rl.iteration = iteration
        rl.start_loop()
        t = perf_counter() - t0
        i = rl.var_dict["i"]
        dat_list = rl.var_dict["dat_list"]
        print(i, t / i)
        if len(dat_list) > 0:
            dat_arr = np.stack(dat_list, axis=0)
            print(dat_arr.shape)
            return dat_arr.tolist()
        else:
            return "Done"  # self.lock.acquisition.tolist()


# ---------------------------------------------------------------------------
# PID  (VERBATIM from original)
# ---------------------------------------------------------------------------
class PID:
    def __init__(self, P=0, I=0, D=0, I_val=0, limit=[-1, 1]):
        self.P = P
        self.I = I
        self.D = D
        self.I_val = I_val
        self.start = I_val
        self.e_prev, self.t_prev = None, None
        self.MV = self.start
        self.limit = limit
        self.on = True

    def check_limit(self):
        max_lim = max(self.limit)
        min_lim = min(self.limit)
        if self.MV >= max_lim:
            self.MV = max_lim
            print("PID reached limit {}!".format(max_lim))
        elif self.MV <= min_lim:
            self.MV = min_lim
            print("PID reached limit {}!".format(min_lim))

    def update(self, e, t):
        if (self.e_prev == None) & (self.t_prev == None):
            self.e_prev, self.t_prev = e, t
        else:
            if self.on:
                self.I_val += self.I * e * (t - self.t_prev)
                self.MV = (
                    self.P * e
                    + self.I_val
                    + self.D * (e - self.e_prev) / (t - self.t_prev)
                )
            # check, whether PID output is within limits.
            self.check_limit()
            self.e_prev, self.t_prev = e, t

    def reset(self):
        self.I_val = self.start
        self.MV = self.start
        self.e_prev, self.t_prev = None, None


# ---------------------------------------------------------------------------
# RP  (REWRITTEN for OS 2.x)
# ---------------------------------------------------------------------------
class RP:
    """
    Hardware abstraction layer for RedPitaya STEMlab 125-14, OS 2.x.

    Replaces the mercury-based implementation with direct rp module calls.
    The public interface (gen_ramp, gen_trig, ext_trig1, ext_trig2, times, N,
    acquire(), acquire_ch(), set_dec(), trigger()) is preserved exactly so that
    RP_Lock (which inherits from RP) needs no changes.

    Supported modes
    ---------------
    "scan"    : Cavity RP — Out1=square wave trigger, Out2=ramp, In1=signal.
                Generators run in burst mode; ADC triggers on AWG positive edge.
    "lock"    : Laser RP  — Out1=PID feedback Slave1, Out2=PID feedback Slave2,
                In1=cavity signal. ADC triggers on AWG positive edge from the
                cavity RP (received on the external trigger input or via the
                internal AWG — same trigger source, no local ramp generated).
    "monitor" : Monitor RP— In1=cavity signal only, no outputs driven.
                ADC triggers on AWG positive edge from the cavity RP.
    """

    def __init__(self, mode="scan"):
        if mode not in ("scan", "lock", "monitor"):
            raise ValueError(
                "RP mode '{}' is not valid. "
                "Must be one of: 'scan', 'lock', 'monitor'.".format(mode)
            )

        # --- init rp module ---
        ret = rp.rp_Init()
        if ret != rp.RP_OK:
            raise RuntimeError("rp_Init() failed with code {}".format(ret))
        rp.rp_Reset()

        # --- proxy objects — preserve the original attribute interface ---
        # gen_trig = fpga.gen(0) → Out1 → RP_CH_1
        # gen_ramp = fpga.gen(1) → Out2 → RP_CH_2
        self.gen_trig = _GenProxy(0)
        self.gen_ramp = _GenProxy(1)

        # --- GPIO ext trigger pins ---
        # DIO0_P → ext_trig1 (enable/disable Slave1 PID)
        # DIO0_N → ext_trig2 (enable/disable Slave2 PID)
        self.ext_trig1 = _GpioProxy(rp.RP_DIO0_P)
        self.ext_trig2 = _GpioProxy(rp.RP_DIO0_N)

        # --- buffer sizes (confirmed 16384 on board) ---
        self.N = rp.ADC_BUFFER_SIZE        # 16384 — oscilloscope samples
        N_gen  = rp.DAC_BUFFER_SIZE        # 16384 — generator samples

        # --- default decimation ---
        dec = int(2 ** 4)   # 16, matching original and Cav.json / Default.json

        # --- time axis (milliseconds, same formula as original) ---
        dur = self._duration(dec)
        self.times = np.linspace(0, dur - (8e-9 * dec), self.N) * 1e3  # ms

        # --- configure acquisition (all modes) ---
        rp.rp_AcqReset()
        rp.rp_AcqSetDecimation(_DEC_MAP[dec])
        rp.rp_AcqSetGain(rp.RP_CH_1, rp.RP_HIGH)  # In1: ±20 V (HV mode)
        rp.rp_AcqSetGain(rp.RP_CH_2, rp.RP_HIGH)  # In2: ±20 V (HV mode)
        rp.rp_AcqSetAveraging(True)
        # Trigger source depends on mode:
        #   scan    — ADC triggers on local AWG positive edge (Out1 square wave
        #             fires the ADC on the same board directly).
        #   lock    — cavity RP's Out1 square wave is routed to this board's In2;
        #             ADC triggers on In2 (CHB) positive edge.
        #   monitor — same wiring as lock; triggers on In2 positive edge.
        if mode == "scan":
            rp.rp_AcqSetTriggerSrc(rp.RP_TRIG_SRC_AWG_PE)
        else:  # lock and monitor
            rp.rp_AcqSetTriggerSrc(rp.RP_TRIG_SRC_CHB_PE)

        # --- configure generators ---
        rp.rp_GenReset()
        if mode == "scan":
            # Out1: square wave trigger, Out2: piezo ramp — both in burst mode.
            self._setup_gen_scan(dec, N_gen)
        elif mode == "lock":
            # Out1 and Out2 carry PID feedback to laser current mod inputs.
            # Configured as DC continuous outputs; PID updates offset at runtime.
            self._setup_gen_lock()
        # monitor: no generator output needed; GenReset() already silences both.

        self.mode = mode

        # --- store dec for set_dec() ---
        self._dec = dec
        self._N_gen = N_gen

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _duration(self, dec):
        """Acquisition duration in seconds. Same formula as original."""
        return 8e-9 * self.N * dec

    @property
    def _trig_src(self):
        """
        Correct ADC trigger source for this board's mode.
        scan    → RP_TRIG_SRC_AWG_PE  (local Out1 square wave)
        lock    → RP_TRIG_SRC_CHB_PE  (cavity RP Out1 arrives on In2)
        monitor → RP_TRIG_SRC_CHB_PE  (same wiring as lock)
        """
        if self.mode == "scan":
            return rp.RP_TRIG_SRC_AWG_PE
        else:
            return rp.RP_TRIG_SRC_CHB_PE

    def _scan_freq_hz(self, dec):
        """
        Scan frequency in Hz for the continuous generators.

        One scan cycle = one full ADC buffer = N * dec * 8 ns.
        Both generators run at this frequency so their period matches
        the acquisition window exactly — no dead time, no offset.
        """
        return 1.0 / self._duration(dec)   # _duration returns seconds

    def _setup_gen_scan(self, dec, N_gen):
        """Configure Out1 (square wave trigger) and Out2 (ramp) for scan mode.

        Both generators run in CONTINUOUS mode at the same frequency.
        Frequency = 1 / acquisition_duration = 1 / (N * dec * 8 ns).
        At dec=16, N=16384: freq ≈ 476 Hz, period ≈ 2.1 ms.

        Out1: square wave — free-running, triggers the ADC on every positive
              edge via RP_TRIG_SRC_AWG_PE. IN2 sees a clean square wave for
              the full acquisition window.
        Out2: ramp (RAMP_DOWN = physically rising on OS 2.x) — free-running
              at the same frequency. No dead time. The PID updates its DC
              offset via gen_ramp.offset between cycles.

        No rp_GenTriggerOnly() calls are needed. trigger() just arms the ADC
        and waits for the next positive edge of the free-running Out1.
        """
        freq_hz = self._scan_freq_hz(dec)

        # --- Out1: square wave trigger (gen_trig / RP_CH_1) ---
        ch_trig = _CH[0]
        rp.rp_GenWaveform(ch_trig, rp.RP_WAVEFORM_SQUARE)
        rp.rp_GenFreqDirect(ch_trig, freq_hz)
        rp.rp_GenAmp(ch_trig, 0.9)
        rp.rp_GenOffset(ch_trig, 0.0)
        rp.rp_GenMode(ch_trig, rp.RP_GEN_MODE_CONTINUOUS)
        rp.rp_GenTriggerSource(ch_trig, rp.RP_GEN_TRIG_SRC_INTERNAL)
        rp.rp_GenOutEnable(ch_trig)

        # --- Out2: scan ramp (gen_ramp / RP_CH_2) ---
        # Frequency, amplitude and mode set here at init.
        # Waveform shape and rp_GenOutEnable are deferred to scan_output_enable()
        # so Out2 is silent until Lock.start_scan() explicitly fires it.
        ch_ramp = _CH[1]
        rp.rp_GenFreqDirect(ch_ramp, freq_hz)
        rp.rp_GenAmp(ch_ramp, 0.5)
        rp.rp_GenOffset(ch_ramp, 0.0)
        rp.rp_GenMode(ch_ramp, rp.RP_GEN_MODE_CONTINUOUS)
        rp.rp_GenTriggerSource(ch_ramp, rp.RP_GEN_TRIG_SRC_INTERNAL)
        # rp_GenWaveform + rp_GenOutEnable → called in scan_output_enable()

    def _setup_gen_lock(self):
        """
        Configure Out1 and Out2 as DC continuous outputs for laser lock mode.

        In lock mode the PID controller writes its output to gen.offset at
        runtime (via _GenProxy). The generator just needs to be alive in
        continuous mode with zero amplitude so the DC offset is all that matters.
        Out1 → Slave1 laser current mod input.
        Out2 → Slave2 laser current mod input.
        """
        for ch in (_CH[0], _CH[1]):
            rp.rp_GenWaveform(ch, rp.RP_WAVEFORM_DC)
            rp.rp_GenAmp(ch, 0.0)
            rp.rp_GenOffset(ch, 0.0)
            rp.rp_GenMode(ch, rp.RP_GEN_MODE_CONTINUOUS)
            rp.rp_GenTriggerSource(ch, rp.RP_GEN_TRIG_SRC_INTERNAL)
            rp.rp_GenOutEnable(ch)

    # ------------------------------------------------------------------
    # Scan output control  (called from RP_Server action handlers)
    # ------------------------------------------------------------------

    def scan_output_enable(self, waveform=None, amplitude=None, offset=None):
        """Write full Out2 generator config and enable its output.

        Parameters
        ----------
        waveform : rp waveform constant, optional
            Default: rp.RP_WAVEFORM_TRIANGLE
        amplitude : float, optional
            Half-swing of the waveform in volts. Default 0.5 V.
            Output swings from (offset - amplitude) to (offset + amplitude).
            Must satisfy: amplitude + abs(offset) <= 1.0  (RP clips at ±1 V).
        offset : float, optional
            DC offset that shifts the scan centre in volts. Default 0.0 V.
        """
        if waveform is None:
            waveform = rp.RP_WAVEFORM_TRIANGLE
        if amplitude is None:
            amplitude = self._scan_amplitude if hasattr(self, "_scan_amplitude") else 0.5
        if offset is None:
            offset = self._scan_offset if hasattr(self, "_scan_offset") else 0.0
        amplitude = float(amplitude)
        offset    = float(offset)
        # HV mode: generator supports up to ±6 V with RP_GAIN_2X.
        # Clamp at 6V for safety.
        if amplitude + abs(offset) > 6.0:
            amplitude = max(0.0, 6.0 - abs(offset))
            print("scan_output_enable: WARNING amplitude clamped to {:.3f} V "
                  "(amp + |offset| must be <= 6.0 V in HV mode)".format(amplitude))
        ch = _CH[1]
        freq_hz = self._scan_freq_hz(self._dec)
        rp.rp_GenWaveform(ch, waveform)
        rp.rp_GenFreqDirect(ch, freq_hz)
        rp.rp_GenAmp(ch, amplitude)
        rp.rp_GenOffset(ch, offset)
        rp.rp_GenMode(ch, rp.RP_GEN_MODE_CONTINUOUS)
        rp.rp_GenTriggerSource(ch, rp.RP_GEN_TRIG_SRC_INTERNAL)
        rp.rp_GenSetGainOut(ch, rp.RP_GAIN_2X)  # enable HV output (±6 V range)
        rp.rp_GenOutEnable(ch)
        self._scan_waveform  = waveform
        self._scan_amplitude = amplitude
        self._scan_offset    = offset
        print("scan_output_enable: Out2 ON  waveform={}  amp={:.3f} V  "
              "offset={:.3f} V  freq={:.1f} Hz".format(
              waveform, amplitude, offset, freq_hz))

    def scan_output_disable(self):
        """Zero Out2 offset and disable its output immediately."""
        rp.rp_GenOffset(_CH[1], 0.0)
        rp.rp_GenOutDisable(_CH[1])
        print("scan_output_disable: Out2 OFF")

    # ------------------------------------------------------------------
    # Scan output control
    # ------------------------------------------------------------------

    def scan_output_enable(self, waveform=None, amplitude=None, offset=None):
        """Write full Out2 generator config and enable its output.

        Parameters
        ----------
        waveform : rp waveform constant, optional
            Default: rp.RP_WAVEFORM_TRIANGLE
        amplitude : float, optional
            Half-swing of the waveform in volts. Default 0.5 V.
            Output swings from (offset-amplitude) to (offset+amplitude).
            amplitude + abs(offset) must be <= 1.0 (RP clips at ±1 V).
        offset : float, optional
            DC offset shifting the scan centre in volts. Default 0.0 V.
        """
        if waveform is None:
            waveform = rp.RP_WAVEFORM_TRIANGLE
        if amplitude is None:
            amplitude = self._scan_amplitude if hasattr(self, "_scan_amplitude") else 0.5
        if offset is None:
            offset = self._scan_offset if hasattr(self, "_scan_offset") else 0.0
        amplitude = float(amplitude)
        offset    = float(offset)
        if amplitude + abs(offset) > 1.0:
            amplitude = max(0.0, 1.0 - abs(offset))
            print("scan_output_enable: WARNING amplitude clamped to {:.3f} V"
                  .format(amplitude))
        ch = _CH[1]
        freq_hz = self._scan_freq_hz(self._dec)
        rp.rp_GenWaveform(ch, waveform)
        rp.rp_GenFreqDirect(ch, freq_hz)
        rp.rp_GenAmp(ch, amplitude)
        rp.rp_GenOffset(ch, offset)
        rp.rp_GenMode(ch, rp.RP_GEN_MODE_CONTINUOUS)
        rp.rp_GenTriggerSource(ch, rp.RP_GEN_TRIG_SRC_INTERNAL)
        rp.rp_GenOutEnable(ch)
        self._scan_waveform  = waveform
        self._scan_amplitude = amplitude
        self._scan_offset    = offset
        print("scan_output_enable: Out2 ON  waveform={}  amp={:.3f} V  "
              "offset={:.3f} V  freq={:.1f} Hz".format(
              waveform, amplitude, offset, freq_hz))

    def scan_output_disable(self):
        """Zero Out2 offset and disable its output."""
        rp.rp_GenOffset(_CH[1], 0.0)
        rp.rp_GenOutDisable(_CH[1])
        print("scan_output_disable: Out2 OFF")

    def set_scan_period(self, period_ms):
        """Set the scan period in ms by selecting the closest valid decimation.

        T = N * dec * 8 ns  (N=16384, clock=125 MHz).
        Finds the dec whose period is closest to period_ms, then updates
        ADC decimation, time axis, and both generator frequencies atomically.

        Parameters
        ----------
        period_ms : float
            Desired scan period in milliseconds.
        """
        period_s = float(period_ms) * 1e-3
        best_dec = min(_DEC_MAP.keys(),
                       key=lambda d: abs(8e-9 * self.N * d - period_s))
        actual_ms = 8e-9 * self.N * best_dec * 1e3
        print("set_scan_period: requested={:.3f} ms  dec={}  actual={:.3f} ms".format(
            period_ms, best_dec, actual_ms))
        self.set_dec(best_dec)   # updates ADC dec + time axis + gen freq

    # ------------------------------------------------------------------
    # Public interface — matching the original RP class
    # ------------------------------------------------------------------

    def set_dec(self, dec):
        """Update decimation at runtime. dec must be a power of 2 up to 65536."""
        dec = int(dec)
        if dec not in _DEC_MAP:
            raise ValueError(
                "Decimation {} is not valid. Must be one of: {}".format(
                    dec, sorted(_DEC_MAP.keys())
                )
            )
        # Update oscilloscope decimation
        rp.rp_AcqSetDecimation(_DEC_MAP[dec])

        # Update generator frequency (scan mode only — both are continuous)
        if self.mode == "scan":
            freq_hz = self._scan_freq_hz(dec)
            rp.rp_GenFreqDirect(_CH[0], freq_hz)
            rp.rp_GenFreqDirect(_CH[1], freq_hz)

        # Update time axis
        dur = self._duration(dec)
        self.times = np.linspace(0, dur - (8e-9 * dec), self.N) * 1e3
        self._dec = dec

    def trigger(self, max_polls=200000):
        """
        Arm the ADC and wait for the next trigger edge.

        Both generators run continuously — no firing needed here.
        For scan mode: ADC triggers on the next positive edge of the
        free-running Out1 square wave (RP_TRIG_SRC_AWG_PE).
        For lock/monitor modes: ADC triggers on the next positive edge
        of In2 (RP_TRIG_SRC_CHB_PE), driven by the cavity RP's Out1.

        The ADC is re-armed on every call so it catches the very next edge.

        Parameters
        ----------
        max_polls : int
            Maximum number of rp_AcqGetTriggerState() polls before giving up.
            At ~1-5 µs per poll, 200000 polls ≈ 0.2-1 s — long enough to
            survive any decimation setting, short enough that the scan loop
            can still process a "stop" command within a few seconds even if
            the trigger source is lost.
        """
        rp.rp_AcqStop()
        rp.rp_AcqSetTriggerSrc(self._trig_src)
        rp.rp_AcqSetTriggerDelay(self.N)
        rp.rp_AcqStart()

        # Wait for trigger edge — with timeout so the board loop is never
        # permanently blocked and can always process incoming stop commands.
        for _ in range(max_polls):
            _ret, state = rp.rp_AcqGetTriggerState()
            if state == rp.RP_TRIG_STATE_TRIGGERED:
                return
        # Timeout reached — log and return without data (acquire_ch returns
        # whatever is in the buffer; the caller discards stale data).
        print("trigger(): timeout waiting for ADC trigger — returning without trigger")

    def acquire(self):
        """
        Acquire both channels. Returns np.array([times, ch1_data, ch2_data]).
        Matches original return shape exactly.
        """
        self.trigger()
        _ret, trig_pos = rp.rp_AcqGetWritePointerAtTrig()

        ch1 = np.zeros(self.N, dtype=np.float32)
        ch2 = np.zeros(self.N, dtype=np.float32)
        rp.rp_AcqGetDataVNP(rp.RP_CH_1, trig_pos, ch1)
        rp.rp_AcqGetDataVNP(rp.RP_CH_2, trig_pos, ch2)

        self.acquisition = np.array([self.times, ch1, ch2])
        return self.acquisition

    def acquire_ch(self, ch):
        """
        Acquire a single channel. ch=0 → In1 (cavity), ch=1 → In2.
        Returns 1-D numpy array of float32 voltages.
        """
        self.trigger()
        _ret, trig_pos = rp.rp_AcqGetWritePointerAtTrig()

        arr = np.zeros(self.N, dtype=np.float32)
        rp.rp_AcqGetDataVNP(_ACQ_CH[ch], trig_pos, arr)
        return arr

    def close(self):
        """Release hardware resources and silence all outputs."""
        rp.rp_GenOffset(_CH[0], 0.0)
        rp.rp_GenOffset(_CH[1], 0.0)
        rp.rp_GenOutDisable(_CH[0])
        rp.rp_GenOutDisable(_CH[1])
        rp.rp_AcqStop()
        rp.rp_Release()
        print("RP.close(): all outputs disabled, hardware released")


# ---------------------------------------------------------------------------
# RP_Lock  (VERBATIM from original)
# ---------------------------------------------------------------------------
class RP_Lock(RP, reaction_loop):
    def __init__(self, addr, mode="lock"):
        RP.__init__(self, mode=mode)
        reaction_loop.__init__(self, (addr[0], 5065))

        self.action_dict["update_settings"] = self.update_settings
        self.ch = 0  # the channel detecting the cavity transmission
        self.Master_pos = 1.75  # will be initialized when the loop starts with the initial Peak position!
        # all of the methods iterate through this dict, so if its empty nothing should happen
        self.settings = {}
        # settings attribute will/must be loaded from the client side!
        # for that purpose the update_settings method is implemented
        self.t0 = perf_counter()
        self.t = 0
        self.iter_num = 0
        self.errs = dict()
        self.iteration = self.loop_iter
        self.setup = self.setup_lock
        self.mode = mode
        self.FSR_ref = None
        # This resets the outputs
        self.gen_trig.reset()
        self.gen_trig.start_trigger()
        self.gen_ramp.reset()
        self.gen_ramp.start_trigger()
        self.feedback = True  # a boolean used to filter out sudden peak jumps detected due to unexpected events in the system (such as trigger jumps etc.)
        # peak_finding stuff
        self.skipped = False
        self.invert = False

    def update_settings(self, settings):
        inverts = []
        for key, val_dict in settings.items():  # iterate through the lasers
            if key not in self.settings.keys():
                self.settings[
                    key
                ] = {}  # if settings dont exist yet, initialize dictionary
            if key == "Master":
                self.Master_pos = val_dict["lockpoint"]
                if self.mode == "scan":
                    self.settings[key][
                        "gen"
                    ] = (
                        self.gen_ramp
                    )  # if this is the cavity lock, then use output2 for master --> ramp offset!
            elif key == "Slave1" and self.mode == "lock":
                self.settings[key]["gen"] = self.gen_trig  # Slave1 locked by Output1
            elif key == "Slave2" and self.mode == "lock":
                self.settings[key]["gen"] = self.gen_ramp  # Slave2 locked by output2
            self.settings[key]["sign"] = +1  # use a default sign for the feedback.
            if key not in self.errs:
                self.errs[key] = 0
            for (
                val_key,
                val_val,
            ) in val_dict.items():  # iterate through the laser-settings
                if val_key == "PID":  # for PIDs, only the gains are sent!
                    self.update_PID(key, val_val)
                else:
                    self.settings[key][
                        val_key
                    ] = val_val  # in each other case, the settings are transferred directly
                if val_key == "peak_finder":
                    self.update_peak_finder(key, val_val)
                if val_key == "invert":
                    inverts.append(val_val)  # check invert attributes.
            if self.settings[key]["enabled"] == False:
                # reset the output when disabling the lock!
                if key == "Slave1":
                    self.gen_trig.offset = 0.0
                elif key == "Slave2":
                    self.gen_ramp.offset = 0.0
                self.settings.pop(
                    key
                )  # if the respective laser lock is not enabled, remove it from the dictionary!
                self.errs.pop(key)
        self.invert = any(
            inverts
        )  # if any of the cavity signals needs inverting, invert
        # print('update settings: {}'.format(self.settings))
        return "Updated lock setting!"

    def acquire_cav_signal(self):
        acquisition = self.acquire_ch(self.ch)
        acquisition *= (-1) ** self.invert
        return acquisition

    def update_peak_finder(self, laser, values):
        name = values.pop("name")  # remove name from peak_finder settings
        if name[:2] == "SG":  # if savitzky golay filter is involved
            self.settings[laser]["SG_m"] = SG_array(**values)  # calculate conv. matrix
        self.settings[laser]["peak_finder"] = name

    def update_PID(self, laser, val):
        # updates PID with new settings (val) for a certain laser. If lock is already running,
        # only changes the PID gains!
        if (
            "PID" in self.settings[laser]
        ):  # if lock already running, the key is already in the dicitonary
            val["I_val"] = self.settings[laser][
                "PID"
            ].I_val  # if lock running, keep the current I_val!
        pid = PID(**val)  # gains stored in yet another dictionary...
        self.settings[laser]["PID"] = pid

    def check_gpio_ext_trig(self):
        val1 = self.ext_trig1.read()
        val2 = self.ext_trig2.read()
        for laser, val in zip(["Slave1", "Slave2"], [val1, val2]):
            if laser in self.settings:
                self.settings[laser]["PID"].on = val

    def update_pos(self):
        """
        Method to obtain the current peak positions from the cavity
        and write them into the setting dictionary.

        Returns
        -------
        None.

        """
        try:
            self.update_data()
        except:
            self.skipped = True  # something failed, skip point!
            return
        self.skipped = False
        for key in self.settings.keys():  # retrieve current cavity scan
            if key == "Master":  # readout individual peak positions
                pos = self.data[key][:, 1][
                    0
                ]  # d- self.Master_pos    # Master is (currently) the only key, where multiple positions are stored!
            else:
                pos = (
                    self.data[key][0] - self.data["Master"][:, 1][0]
                )  # take the relative position!
            # check whether a quick jump occured. This procedure should ignore outliers due to unexpected jumps! locking step ignored with feedback = Flase
            if "position" in self.settings[key]:
                if np.abs(pos - self.settings[key]["position"]) < 20e-3:
                    self.feedback = True
                else:
                    self.feedback = False
                    print("skipped point!")
            self.settings[key]["position"] = pos

    def check_sign(self, iters=100):
        """
        Method to check whether the pid feedback has correct sign.
        It runs the step method a number (iters) of iterations and flips the sign,
        if the peak position error increases over that duration.

        Parameters
        ----------
        iters : int, optional
            Number of locking step interations to determine
            wheter the sign is correct. The default is 50.

        Returns
        -------
        None.

        """
        for n in range(iters):
            self.step()  # locking step
            if n == 0:
                errs_0 = self.errs  # at first step, save the intitial deviation
        for key, val in self.settings.items():
            if (
                key != "Master"
            ):  # exclusion of the master peak, since the sign is always correct.
                if (
                    abs(self.errs[key]) - abs(errs_0[key]) >= 5e-3
                ):  # if the error increased by more than 5 MHz, the sign is flipped.
                    val["sign"] = -val["sign"]
                    print("Sign for {} flipped!".format(key))

    def setup_lock(self):
        print("setting up lock")
        self.gen_ramp.offset = 0.0
        if self.mode == "lock":
            self.gen_trig.offset = (
                0.0  # if laser lock, additionally reset the second output
            )
        for key, val in self.settings.items():
            val["PID"].reset()
        # before starting the lock, do a number of acquire iterations --> steady state of successive cavity scans
        self.init_FSR_ref()  # measure the FSR for the Master laser and save the average as attribute. used for error calculation
        # reset the locking stuff
        for key, val in self.settings.items():
            if key == "Master":
                self.settings[key]["height"] = self.data[key][:, 1][
                    1
                ]  # readout current height of all peaks and save it
            else:
                self.settings[key]["height"] = self.data[key][
                    1
                ]  # readout current height of all peaks and save it
        self.check_sign()
        print("Master_pos:", self.Master_pos)

    def init_FSR_ref(self, averages=20):
        FSRs = []
        for i in range(averages):
            self.update_data()
            FSRs.append(self.FSR)
        self.FSR_ref = np.mean(
            FSRs
        )  # this is used to normalize the lockpoint --> average FSR

    def update_err(self, laser):
        """
        calculates the error (deviation of position from lockpoint)

        Parameters
        ----------
        laser : TYPE
            DESCRIPTION.
        setting : TYPE
            DESCRIPTION.
        """
        s = self.settings[laser]
        if laser == "Master":
            err = (
                s["position"] - s["lockpoint"]
            ) / self.FSR  # calculate individual errors
        else:
            # err = (s['position'] - (s['lockpoint']-self.Master_pos))/self.FSR
            err = (
                s["position"] / self.FSR
                - (s["lockpoint"] - self.Master_pos) / self.FSR_ref
            )
        self.errs[laser] = err

    def step(self):
        """
        Method for an individual locking step. During the step, check_positions
        is used to verify that the peaks have sufficient distance to their
        range borders.
        """
        self.check_gpio_ext_trig()
        self.t = perf_counter() - self.t0
        self.update_pos()  # retrieve current peak positions
        if self.feedback == True:
            for key, val in self.settings.items():
                self.update_err(key)  # calculate individual errors
                val["PID"].update(
                    self.errs[key] * val["sign"], self.t
                )  # update the corresponding PID!
                if key == "Master" and self.mode == "scan":
                    self.gen_ramp.offset = val["PID"].MV
                elif key != "Master" and self.mode == "lock":
                    val["gen"].offset = val["PID"].MV

        elif self.feedback == False:
            return
        # if np.abs(val['gen'].offset) > 0.99:
        #    self.running = False
        #    print('Error: offset limit reached!')

    def check_height(self):
        # essentially checks whether there is a peak in the range. checked by height threshhold.
        Bools = []
        for key, val in self.settings.items():
            if key == "Master":
                h = self.data[key][:, 1][1]  # current height
            else:
                h = self.data[key][1]  # current height
            # print('height', h, val['height'])
            if h < (val["height"] * 1 / 5):
                print("Peak {} too low/ dissapeared? out of range?".format(key))
                Bools.append(False)
            else:
                Bools.append(True)
        return all(Bools)

    def check_lockpoints(self):
        """
        Method for verifying, whether the current lockpoint actually lies in
        the provided range (with some free space of 15MHz). returns a boolean,
        which can be used for looop break condition.

        Returns
        -------
        Boolean
            True if lockpoints are in the range, False if not.

        """
        Bools = []
        for key, val in self.settings.items():
            if key == "Master":
                range = val["range"][1]
                i0 = self.times[range[0]]
                i1 = self.times[range[1]]
            else:
                range = val["range"]
                i0 = self.times[range[0]]
                i1 = self.times[range[1]]
            v = val["lockpoint"]
            if not (i0 < v < i1):
                print("Lockpoint {} for {} out of range!".format(v, key))
                Bools.append(False)
            else:
                Bools.append(True)
        return all(Bools)

    def check_positions(self, dmin=5e-3):
        """
        Method to check

        Parameters
        ----------
        dmin : TYPE, optional
            DESCRIPTION. The default is 15e-3.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        Bools = []
        for key, val in self.settings.items():
            if key == "Master":
                range = val["range"][1]
                i0 = self.times[range[0]] - self.Master_pos
                i1 = self.times[range[1]] - self.Master_pos
            else:
                range = val["range"]
                i0 = self.times[range[0]] - self.Master_pos
                i1 = self.times[range[1]] - self.Master_pos
            pos = val["position"]

            if (abs(pos - i0) <= dmin) or (abs(pos - i1) <= dmin):
                print("Position {} of {} too close to border!".format(pos, key))
                Bools.append(False)
            else:
                Bools.append(True)
        return all(Bools)

    def loop_iter(self, *args, **kwargs):
        self.step()  # make a locking step
        self.iter_num += 1

    def start(self, *args, **kwargs):
        self.feedback = True
        self.gen_ramp.offset = 0
        sleep(0.1)
        self.update_data()
        self.Master_pos = self.settings["Master"][
            "lockpoint"
        ]  # Set the zero position of the master laser
        self.iter_num = 0  # starting index of the while loop
        self.running = (
            self.check_lockpoints()
        )  # Set running to True, such that the loop will run!
        print(self.running)
        # initialize error dictionary
        if self.running:
            self.errs = dict()
            self.errs_times = np.array([])
            self.errs_arr = []
            self.t0 = perf_counter()
            self.start_loop()

    def acquire_peaks(self, laser, r):
        name = self.settings[laser]["peak_finder"]
        peak_finder = peak_finders[name]
        if name[:2] == "SG":  # if savitzky golay filter involved, give the matrix
            m = self.settings[laser]["SG_m"]
            return peak_finder(self.times, self.acquisition, r, m=m)
        else:
            return peak_finder(self.times, self.acquisition, r)

    def get_peaks(self, laser, setting):
        if laser == "Master":
            P_l = []
            for r in setting[
                "range"
            ]:  # 2 ranges for the master peak --> determination of frequency axis!
                P = self.acquire_peaks(laser, r)
                P_l.append(P)
            return np.stack(P_l, axis=1)
        else:
            r = setting["range"]
            P = self.acquire_peaks(laser, r)
            return P

    def update_data(self):
        """
        Method to obtain the Master and Slave peaks and create data dictionary
        """
        self.acquisition = (
            self.acquire_cav_signal()
        )  # acquire scope data and make a shortcut. only acquire the desired channel!
        self.data = dict()
        for key, val in self.settings.items():
            self.data[key] = self.get_peaks(key, val)
        self.FSR = np.abs(self.data["Master"][:, 0][0] - self.data["Master"][:, 1][0])
