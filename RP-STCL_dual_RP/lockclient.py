# -*- coding: utf-8 -*-
"""
lockclient.py  —  Combined Cav+Mon single-RP edition
=====================================================

Key addition vs the original lockclient.py
-------------------------------------------
Mode  ``"scan_mon"``  (new)
    A single RedPitaya that acts as **both** the cavity-scanning master
    (Cav) **and** the monitor (Mon).  Internally the LockClient registers
    it once in ``self.masters`` *and* once in ``self.monitors``, so all
    existing scan / lock / monitor methods work without modification.

    The Monitor classes (cavity + error) use the same dark-themed Qt5Agg
    aesthetics from the Mon-only branch:
      • Dark background  (#1e1e2e / #181825 Catppuccin palette)
      • Dual-axis layout when trigger is visible (IN1 cavity, IN2 trigger)
      • Coloured range-spans and lockpoint vlines
      • ``Monitor.show_trigger`` class flag (default True)

Original modes ``"scan"``, ``"lock"``, ``"monitor"``, ``"ext_scan"`` are
unchanged and fully backward-compatible.
"""

from communication import Sender, RP_connection, Path
import matplotlib

matplotlib.use("Qt5Agg")  # for plotting in another process
import matplotlib.pyplot as plt
import numpy as np
import json
from time import sleep, perf_counter
from general import *
import threading
import multiprocessing as mp
import queue  # for exception handling using multiprocessing.Queue
from copy import deepcopy
import sys
from scipy.constants import golden  # golden ratio
from RP_side.peak_finders import peak_finders, SG_array, SG_filter

window_size = 21
order = 1
order_range = range(order + 1)
half_window = (window_size - 1) // 2

m = SG_array(window_size, order, deriv=1)


def _patch_send_for_inline(rp_obj, sender):
    """
    Monkey-patch rp_obj.send so the while-True wait loop calls
    sender._process_sel_once() on each iteration instead of just sleep(0).
    This makes registration and processing happen in the same thread.
    """
    import functools
    from communication import RP_connection
    _orig_send = RP_connection.send

    @functools.wraps(_orig_send)
    def _inline_send(self_rp, Sender_arg, action, value="Hello World!",
                     loop_action=False, loop=False):
        if not Sender_arg.running:
            print("Event_loop not running!")
            return None
        import socket as _socket, selectors as _sel, libclient as _lc
        request = self_rp.create_request(action, value)
        if not loop:
            sock = self_rp.connect_socket(self_rp.addr)
            addr = self_rp.addr
            stop = True
        else:
            sock = self_rp.lsock
            addr = (self_rp.addr[0], 5065)
            stop = False
            if action == "stop":
                stop = True
        event_state = _sel.EVENT_READ | _sel.EVENT_WRITE
        message = _lc.Message(Sender_arg.sel, sock, addr, request, stop=stop)
        Sender_arg.sel.register(sock, event_state, data=message)
        if loop_action:
            self_rp.loop_running = True
        while True:
            # Process the selector inline — same thread as registration.
            Sender_arg._process_sel_once()
            if loop_action and self_rp.loop_running:
                if self_rp.lsock is None:
                    try:
                        key = Sender_arg.sel.get_key(sock)
                        if key.events & _sel.EVENT_READ:
                            laddr = (self_rp.addr[0], 5065)
                            sleep(2)
                            self_rp.lsock = self_rp.connect_socket(laddr)
                            sleep(0.5)
                            try:
                                self_rp.lsock.getpeername()
                            except Exception as exp:
                                print(f"Exception occured during connection: {exp}")
                                self_rp.loop_running = False
                                return "Exception occured during connection..."
                    except KeyError:
                        pass
            if message.selkey is not None:
                if self_rp.loop_running and action == "stop":
                    self_rp.loop_running = False
                break
        if message.response is None:
            result = None
        else:
            result = message.response["result"]
        if loop_action:
            self_rp.lsock = None
        return result

    # Bind to this specific rp instance only
    import types
    rp_obj.send = types.MethodType(_inline_send, rp_obj)


def _make_nonblocking_event_loop(sender):
    """
    Patch sender.event_loop to use sel.select(timeout=0) instead of timeout=1.
    This prevents the selector from blocking during socket registration from
    another thread, which is the root cause of hangs on Windows.
    """
    import selectors as _sel
    import traceback as _tb
    from time import sleep as _sleep

    def _fast_event_loop():
        sender.running = True
        try:
            while True:
                if not sender.sel.get_map():
                    _sleep(1e-4)
                else:
                    events = sender.sel.select(timeout=0)  # non-blocking
                    for key, mask in events:
                        message = key.data
                        if message is not None:
                            try:
                                if sender.mode == "monitor":
                                    message.buffersize = int(2**18)
                                else:
                                    message.buffersize = int(2**12)
                                message.process_events(mask)
                            except Exception:
                                print("Main: Error: Exception for %s:\n%s" % (
                                    message.addr, _tb.format_exc()))
                                message.close()
                if not sender.running:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            return

    sender.event_loop = _fast_event_loop


def monitor(v, *args, **kwargs):
    try:
        m = Monitor(*args, bool_var=v)
        _make_nonblocking_event_loop(m)
        m.start_event_loop()
        m.start_monitor()
    except Exception as _exc:
        import traceback
        print("[Monitor] EXCEPTION:", flush=True)
        traceback.print_exc()
    finally:
        v.value = False
    return


def monitor_errors(v, *args, **kwargs):
    try:
        m = ErrorMonitor(*args, bool_var=v, **kwargs)
        _make_nonblocking_event_loop(m)
        m.start_event_loop()
        m.start_monitor()
    except Exception as _exc:
        import traceback
        print("[ErrorMonitor] EXCEPTION:", flush=True)
        traceback.print_exc()
    finally:
        v.value = False
    return





def init_mon_dict():
    d = dict(
        running=mp.Value("i", False),  # use mp.Value for sharing between processes!
        running_err=mp.Value("i", False),  # same but for error monitor process
        queue=mp.Queue(),  # used to communicate to cavity monitor process
        queue_err=mp.Queue(),  # used to communicate to error monitor process
    )
    return d


class LockClient(Sender):
    def __init__(self, redpitayas, FSR=906, DIR=None):
        """
        This class handles the communication to all redpitayas involved in the lock.
        The redpitayas (RP_client objects) are stored in a dictionary, where the
        respective keys are used as a reference to individual devices throughout
        all of the defined methods. The free spectral range of the cavities used
        is used for scaling of monitored error signals, while a directory can be
        provided to store json files with locksettings.

        Parameters
        ----------
        redpitayas : dict
            Contains objects of class RP_client.
        FSR : float, optional
            Free spectral range [MHz] of the cavities used in the system. Only used for
            scaling of monitored errors. The default is 906.
        DIR : string, optional
            directory to save settings files. If None, a default directory from
            the repository is used. The default is None.

        New mode: ``"scan_mon"``
            Register a single RP as both the scanning master and the monitor.
            Set  ``RP_client(..., mode="scan_mon")``  in the RPs dict.
        """

        if DIR != None:  # if a directory is given, initiate use it for the Sender class
            Sender.__init__(self, DIR=DIR)
        else:  # this is the default. DIR is the directory where the modules are loaded from.
            Sender.__init__(self)
        self.FSR = FSR  # FSR of the used transfer cavity
        self.RPs = (
            redpitayas  # dictionary of RP_client objects with respective settings
        )
        self.masters = []
        self.monitors = dict()
        # load the settings from the respective json files
        for key, val in self.RPs.items():
            val.label = key  # set the key as an attribute for the RP objects!
            val.upload_current()
            if val.mode in ["scan", "ext_scan", "scan_mon"]:
                self.masters.append(key)
            if val.mode in ["monitor", "scan_mon"]:
                self.monitors[key] = init_mon_dict()
            filepath = Path(self.DIR, f"{key}.json")
            if filepath.exists():
                with open(filepath, "r") as file:
                    val.settings = json.load(file)
            else:
                self._load_default_settings(key)
            self.save_settings(key)  # afterwards, create a file with these settings!

    def start(self):
        """
        Starts up the LockClient, which includes the event loop and synchronizes
        the decimation settings on all redpitayas. This requires a connection
        on all redpitayas.
        """
        self.start_event_loop()
        # initializing the dec settings
        for m in self.masters:  # initialize the dec settings!
            dec = self.RPs[m].settings["Master"]["dec"]
            self.set_dec(m, dec)

    def close(self):
        """
        Close the LockClient and everything related to it. The order is important here:
         - first, monitors are closed.
         - next, any loops running on redpitayas (mode = lock or monitor) are closed.
         - only then loops running on master redpitayas (mode = scan or scan_mon) are closed,
             since they are triggering the other redpitayas.
         - After all redpitaya loops are stopped, the listening servers on the redpitayas
             are stopped (disconnected)
         - Finally, the event loop used for communication is closed.
        """
        # monitors
        for key, val in self.monitors.items():
            if val["running"].value:
                self.stop_monitor(key)
        # looping redpitayas
        for master_RP in self.masters:
            RPs = self.find_slave_RPs(master_RP)
            # the last entry in that list is the cavity scanning redpitaya, so it is closed last!
            for RP in RPs:
                if self.RPs[RP].loop_running:
                    self.stop_loop(RP)
        for RP in self.RPs:  # disconnect all the redpitayas
            self.disconnect(RP)
        self.stop_event_loop()  # finally, stop the event loop which handles communication!

    ################### Finding stuff ######################################

    def find_master_RP(self, RP):
        """
        Finds the master redpitaya associated with RP

        Parameters
        ----------
        RP : string
            Key of the RedPitaya in question.

        Returns
        -------
        master_RP : string
            Key of the master RedPitaya, which scans the cavity that RP refers to.
        """
        if RP not in self.masters:
            master_RP = self.RPs[RP].settings["Master"]
        else:
            master_RP = RP
        return master_RP

    def find_monitor_RP(self, RP):
        """
        Find the monitor redpitaya that watches the cavity associated with RP.

        Parameters
        ----------
        RP : string
            Key of the RedPitaya in question.

        Returns
        -------
        monitor_RP : string
            Key of the RedPitaya which monitors the cavity that RP refers to.
        """

        master_RP = self.find_master_RP(RP)
        slaves = self.find_slave_RPs(master_RP)
        # scan_mon RP is its own monitor — check it first
        if master_RP in self.monitors:
            return master_RP
        for key in slaves:
            if key in self.monitors:
                return key

    def find_slave_RPs(self, master_RP):
        """
        Finds all slave RPs that are associated with the master_RP.
        --> redpitayas that are working with the same cavity given by master_RP

        Parameters
        ----------
        master_RP : string
            Key of the RedPitaya in question.

        Returns
        -------
        RPs : list of strings
            Keys of the laserlocking RedPitayas, which refer to the cavity of master_RP
        """

        RPs = []
        for key, val in self.RPs.items():
            if val.settings["Master"] == master_RP and val.mode in ["lock", "monitor"]:
                RPs.append(key)
        # For scan_mon, the RP itself is in self.monitors but its settings["Master"]
        # is its own key (same RP) — avoid double-adding it.
        if master_RP not in RPs:
            RPs.append(master_RP)
        return RPs

    ################# decorators #######################################

    def _apply_to_monitor(func):
        # decorator for functions that change something visible on the cavity monitor
        def inner(self, RP, *args, **kwargs):
            result = func(
                self, RP, *args, **kwargs
            )  # call the function first! --> settings are changed
            # afterwards, the settings are sent to the monitor
            self.set_monitor(RP)
            return result

        return inner

    def _check_cavity_scanned(func):
        # another decorator, used to check if acquisition is possible! --> is cavity scanned/ redpitaya triggered?
        def inner(
            self, RP, *args, **kwargs
        ):  # IMPORTANT: at least RP as argument is expected!
            if self.check_cavity_scanned(RP):
                return func(
                    self, RP, *args, **kwargs
                )  # output of the function shall be returned!
            else:
                # usually an array is expected as output. If no acquisition is possible, an empty array is returned.
                return np.array([])

        return inner

    def _check_for_loop(func):
        # checks if a loop is already running before starting a new one.
        def inner(
            self, RP, *args, **kwargs
        ):  # IMPORTANT: at least RP as argument is expected!
            if self.RPs[RP].loop_running:
                # scan_mon: the monitor mp.Process is completely independent of
                # the scan loop — both run concurrently on the same board.
                # Bypass the loop guard entirely for scan_mon so start_monitor
                # and start_error_monitor are always allowed through.
                # NOTE: func.__name__ is "inner" here due to decorator stacking,
                # so we check the RP mode instead of the function name.
                if self.RPs[RP].mode == "scan_mon":
                    return func(self, RP, *args, **kwargs)
                print(
                    f"Loop currently running on {RP}! Stop it before running this function!"
                )
                return np.array([])
            else:
                return func(self, RP, *args, **kwargs)

        return inner

    def _check_update_setting(func):
        def inner(self, RP, laser, key, val):
            if (
                laser not in self.RPs[RP].settings
            ):
                print(f"There is no laser {laser} in the settings!")
                return
            elif type(self.RPs[RP].settings[laser]) == dict:
                if (
                    key not in self.RPs[RP].settings[laser]
                ):
                    var = input(f"{key} does not exist in settings! Add it? (y/n)")
                    if var == 'y':
                        pass
                    else:
                        print(f'{key} not added.')
                        return
            if laser == 'Master' and not (RP in self.masters):
                print("Master settings can not be changed with this command for a non-scanning RP. If you want to change the scanning cavity, use the method 'change_cavity'")
                return
            if not self.check_new_settings(RP, laser, key, val):
                return
            return func(self, RP, laser, key, val)

        return inner

    def check_cavity_scanned(self, RP):
        """
        used to check if the cavity associated with RP is currently scanned.
        If not, then this means that it is not triggered, and no response would arrive,
        blocking the entire script...

        For ``scan_mon`` mode: the same RP scans AND monitors, so as long as
        its loop is running (or it is ext_scan), acquisition is valid.
        """
        master = self.RPs[RP].settings["Master"]
        if type(master) == str:  # if not master RP, check if cavity is scanned first!
            if self.RPs[master].loop_running or self.RPs[master].mode == "ext_scan":
                return True
            else:
                print(f"No scanning loop running on {master}!")
                return False
        else:  # RP is the master (or scan_mon) — it scans the cavity itself!
            return True

    def check_new_settings(self, RP, laser, key, val):
        """
        Used when updating settings.
        Checks if the new settings are valid.
        """
        if key == "range":
            if not check_range(laser, val, self.get_current_dec(RP)):
                print(
                    f"range {val} will not work! pay attention to the order and limits!"
                )
                return False
            else:  # if range is fine, also adjust the lockpoint if necessary!
                print("check if lockpoint is still fine")
                return self.new_range_new_lp(RP, laser, val)
        if key == "lockpoint":
            R = self.RPs[RP].settings[laser]["range"]
            if not check_lockpoint(laser, R, val):
                print(
                    f"lockpoint {val} is not valid. Either not float or outside of range {R}! (second range for Master!)"
                )
                return False
        if key == "enabled":
            if not type(val) == bool:
                print(f"{key} has to be of type bool!")
                return False
        if key == "PID":
            if not type(val) == dict:
                print(f"{key} has to be a dictionary!")
                return False
            else:
                return check_PID(val)
        else:
            return True

    def check_range_contains_lp(self, RP, laser, R):
        lp = self.RPs[RP].settings[laser]["lockpoint"]
        if laser == "Master":
            r = R[1]
        else:
            r = R
        return r[0] < lp < r[1]

    def new_range_new_lp(self, RP, laser, R):
        if not self.check_range_contains_lp(RP, laser, R):
            query = input(
                f"range {R} does not contain the current lockpoint. If this is intended, input new lockpoint here (non-valid value to cancel):\n"
            )
            if check_lockpoint(laser, R, float(query)):
                self.RPs[RP].settings[laser]["lockpoint"] = float(query)
                return True
            else:
                print(f"canceling...")
                return False
        else:
            return True

    def _check_update_setting(func):
        def inner(self, RP, laser, key, val):
            if laser not in self.RPs[RP].settings:
                print(f"There is no laser {laser} in the settings!")
                return
            elif type(self.RPs[RP].settings[laser]) == dict:
                if key not in self.RPs[RP].settings[laser]:
                    var = input(f"{key} does not exist in settings! Add it? (y/n)")
                    if var == 'y':
                        pass
                    else:
                        print(f'{key} not added.')
                        return
            if laser == 'Master' and not (RP in self.masters):
                print("Master settings can not be changed with this command for a non-scanning RP. If you want to change the scanning cavity, use the method 'change_cavity'")
                return
            if not self.check_new_settings(RP, laser, key, val):
                return
            return func(self, RP, laser, key, val)

        return inner


    #################### Locking related functions ############################

    def stop_loop(self, RP):
        """
        Stops the loop running on the RedPitaya. This includes lock and scan loops.

        For scan-mode RPs (including scan_mon), sends "stop_scan" first to
        disable Out2 immediately.
        """
        if self.RPs[RP].mode in ["scan", "scan_mon"]:
            try:
                self.send(RP, "stop_scan")
            except Exception as e:
                print(f"stop_loop: stop_scan failed (board may be unreachable): {e}")
        return self.send(RP, "stop")

    def start_scan(self, RP, amplitude=0.5, offset=0.0, period_ms=None):
        """
        Start the cavity scan: enables Out2 triangle waveform and acquisition loop.

        Works for both ``scan`` and ``scan_mon`` modes.
        """
        if self.RPs[RP].loop_running:
            print(f"Loop already running on {RP}! Call stop_loop('{RP}') first.")
            return
        self.RPs[RP]._scan_amplitude = float(amplitude)
        self.RPs[RP]._scan_offset    = float(offset)
        value = {"amplitude": float(amplitude), "offset": float(offset)}
        if period_ms is not None:
            value["period_ms"] = float(period_ms)
        return self.start_loop(RP, "start_scan", value=value)

    @_check_for_loop
    def start_loop(self, RP, action, value="Hello world!"):
        """
        Start any kind of loop on the RedPitaya remotely using this command.
        """
        t = threading.Thread(
            target=self.send, args=(RP, action),
            kwargs=dict(loop_action=True, value=value)
        )
        t.daemon = True
        t.start()

    def set_scan_output(self, RP, amplitude=None, offset=None, period_ms=None):
        """
        Update scan output parameters while the scan is running.
        Takes effect immediately — no need to stop/restart the scan.
        """
        if not self.RPs[RP].loop_running:
            print(f"No scan running on {RP}. Start scan first.")
            return
        if amplitude is None:
            amplitude = getattr(self.RPs[RP], "_scan_amplitude", 0.5)
        if offset is None:
            offset = getattr(self.RPs[RP], "_scan_offset", 0.0)
        self.RPs[RP]._scan_amplitude = float(amplitude)
        self.RPs[RP]._scan_offset    = float(offset)
        value = {"amplitude": float(amplitude), "offset": float(offset)}
        if period_ms is not None:
            value["period_ms"] = float(period_ms)
        return self.send(RP, "set_scan_output", value=value)

    @_check_cavity_scanned  # only start lock if cavity is scanned.
    def start_lock(self, RP):
        """
        Initiate the lock with the current settings on one redpitaya.

        If the free-running scan loop is still active it is stopped first.
        start_loop() is guarded by @_check_for_loop and silently returns
        if loop_running is True, so this explicit stop is required.
        """
        # Stop the free-running scan so that the lock loop can claim port 5065.
        if self.RPs[RP].loop_running and self.RPs[RP].mode in ("scan", "scan_mon"):
            self.stop_loop(RP)
            sleep(1.0)   # board finishes scan cleanup before lock opens port 5065
        self.update_settings(RP)
        return self.start_loop(RP, "start_lock")

    ##################### settings related functions ###########################

    def _load_default_settings(self, RP):
        """
        Load default settings, if the redpitaya does not yet have any.
        """
        print(
            f"No settings for {RP} found, creating new setting file based on defaults."
        )
        filepath = Path(self.DIR, "Default.json")
        with open(filepath, "r") as file:
            default = json.load(file)
        if self.RPs[RP].mode in ["scan", "ext_scan", "scan_mon"]:
            settings = dict(Master=default["Master"])
        else:
            if len(self.masters) > 0:
                default["Master"] = self.masters[0]
            else:
                default["Master"] = "Cav"
            print(f"Master set to {default['Master']}")
            settings = default
        self.RPs[RP].settings = settings

    def change_cavity(self, RP, RP_master):
        """
        Update the cavity which the redpitaya RP corresponds to RP_master.
        """
        if RP_master not in self.masters:
            print(f"{RP_master} is not scanning a cavity.")
            return
        if RP not in self.masters:
            self.RPs[RP].settings["Master"] = RP_master

    @_apply_to_monitor
    def update_settings(self, RP):
        """
        updates all locking settings of one RedPitaya by loading them from the
        corresponding json file.
        """
        RP_ = self.RPs[RP]
        with open(Path(self.DIR, f"{RP}.json"), "r") as file:
            settings = json.load(file)
        RP_.settings = settings
        settings = self.retrieve_settings(RP)
        self.send(RP, "update_settings", value=settings)

    @_check_update_setting
    @_apply_to_monitor
    def update_setting(self, RP, laser, key, val):
        """
        Update a setting of the lock on a RedPitaya.
        """
        self.RPs[RP].settings[laser][key] = val
        self.save_settings(RP)
        settings = self.retrieve_settings(RP)
        return self.send(RP, "update_settings", value=settings)

    def save_settings(self, RP):
        """
        Saves all locking settings of one RedPitaya to a json file
        """
        with open(Path(self.DIR, f"{RP}.json"), "w") as file:
            json.dump(self.RPs[RP].settings, file, indent=4)

    def retrieve_settings(self, RP):
        """
        Used when sending settings to the RedPitaya. Mainly converts ms values
        in the ranges to indexes.
        """
        RP_ = self.RPs[RP]
        if not (RP_.mode in ["scan", "ext_scan", "scan_mon"]):
            settings = deepcopy(RP_.settings)
            settings["Master"] = deepcopy(
                self.RPs[RP_.settings["Master"]].settings["Master"]
            )
        else:
            settings = deepcopy(RP_.settings)
        # convert range values from ms to index values
        for key in settings:
            dec = settings["Master"]["dec"]
            if key == "Master":
                R = settings[key]["range"]
                settings[key]["range"] = [
                    [ms2index(x, dec) for x in r] for r in R
                ]
            else:
                R = settings[key]["range"]
                settings[key]["range"] = [ms2index(x, dec) for x in R]
        return settings

    def retrieve_monitor_settings(self, master_RP):
        """
        Retrieves all combined settings of redpitayas associated with master_RP.
        """
        settings = {}
        RPs = self.find_slave_RPs(master_RP)
        for key in RPs:
            for val_key, val_val in self.RPs[key].settings.items():
                s = deepcopy(self.retrieve_settings(key))[val_key]
                if key in self.masters and val_key == "Master":
                    settings[val_key] = s
                elif (
                    val_key != "Master" and self.RPs[key].mode == "lock"
                ):
                    settings[f"{key} : {val_key}"] = s
        return settings

    ###################### DEC / Scan frequency ###############################

    def get_current_dec(self, RP):
        """
        Get the current dec setting for the cavity scan associated with RP
        """
        settings = deepcopy(self.RPs[RP].settings)
        for key in settings:
            if key == "Master" and type(settings[key]) != str:
                dec0 = settings[key]["dec"]
            else:
                dec0 = self.RPs[settings["Master"]].settings["Master"]["dec"]
        return dec0

    def rescale_settings(self, RP, c):
        """
        Scales the settings associated with a scan time axis by a factor.
        """
        settings = self.RPs[RP].settings
        for key in settings:
            if key == "Master" and type(settings[key]) != str:
                R = settings[key]["range"]
                settings[key]["range"] = [
                    [x * c for x in r] for r in R
                ]
                settings[key]["lockpoint"] *= c
            elif key != "Master":
                R = settings[key]["range"]
                settings[key]["range"] = [x * c for x in R]
                settings[key]["lockpoint"] *= c

    @_apply_to_monitor
    def set_dec(self, master_RP, dec):
        """
        Set the dec setting for redpitayas associated with a specific master RP
        """
        if not check_dec(dec):
            return
        RPs = self.find_slave_RPs(master_RP)
        for RP in RPs:
            dec0 = self.get_current_dec(RP)
            self.rescale_settings(RP, dec / dec0)
        self.RPs[master_RP].settings["Master"]["dec"] = dec
        for RP in RPs:
            self.save_settings(RP)
            self.send(RP, "set_dec", value=dec)
        sleep(0.5)

    ################ Monitoring related functions ######################

    @_check_for_loop
    @_check_cavity_scanned
    def start_error_monitor(self, RP, tmin=10e-3):
        """
        Starts the error monitoring on the monitoring RedPitaya.

        For ``scan_mon`` mode: uses a background thread (see start_monitor
        docstring for why threading is used instead of mp.Process on Windows).
        """
        if RP in self.monitors:
            mon = self.monitors[RP]
            master_RP = self.find_master_RP(RP)
            settings = self.retrieve_monitor_settings(master_RP)
            if self.RPs[RP].mode == "scan_mon":
                rp_mon = RP_client(
                    self.RPs[RP].addr,
                    self.RPs[RP].settings,
                    mode="monitor",
                )
                rp_mon.label        = self.RPs[RP].label
                rp_mon.connected    = True
                rp_mon.loop_running = True
                rp_mon.lsock        = self.RPs[RP].lsock
                print("Starting error monitor thread (scan_mon mode)")
                t = threading.Thread(
                    target=monitor_errors,
                    args=(mon["running_err"], rp_mon, mon["queue_err"], settings),
                    kwargs=dict(FSR=self.FSR, tmin=tmin),
                    daemon=True,
                )
                t.start()
                print("Error monitor thread started")
            else:
                print("Starting background process")
                self.p = mp.Process(
                    target=monitor_errors,
                    args=(mon["running_err"], self.RPs[RP], mon["queue_err"], settings),
                    kwargs=dict(FSR=self.FSR, tmin=tmin),
                )
                self.p.daemon = True
                self.p.start()
                print("monitoring process started")

    @_check_for_loop
    @_check_cavity_scanned
    def start_monitor(self, RP):
        """
        Starts the monitoring of the cavity signal.

        For ``scan_mon`` mode: uses a background **thread** instead of a
        separate process.  On Windows, mp.Process uses "spawn" which starts
        a fresh interpreter where Sender's class-level event loop variables
        are never initialised, causing every send() call to crash.
        Threading shares the parent process so Sender.running / Sender.sel
        are already set and port 5000 is immediately usable.

        For plain ``monitor`` mode: keeps the original mp.Process behaviour.
        """
        if RP in self.monitors:
            mon = self.monitors[RP]
            master_RP = self.find_master_RP(RP)
            settings = self.retrieve_monitor_settings(master_RP)
            if self.RPs[RP].mode == "scan_mon":
                # Port 5066: dedicated monitor port on the board.
                # No Sender event loop needed — pure TCP connect/get/close.
                # Qt window setup runs on the main thread (Windows requirement).
                rp_mon = RP_client(
                    self.RPs[RP].addr,
                    self.RPs[RP].settings,
                    mode="monitor",
                )
                rp_mon.label     = self.RPs[RP].label
                rp_mon.connected = True
                m_obj = Monitor(rp_mon, mon["queue"], settings)
                m_obj.monitor_running = mon["running"]
                m_obj.running = True   # skip Sender guard in any legacy send()
                print("Setting up monitor on main thread (Qt window)...")
                m_obj.setup_monitor()
                mon["running"].value = True
                def _update_loop(m, q, v):
                    from time import sleep as _sl
                    import queue as _q
                    while v.value:
                        _sl(10e-3)
                        try:
                            query = q.get_nowait()
                            if query[0] == "stop":       v.value = False
                            elif query[0] == "settings": m.update_settings(query[1])
                            elif query[0] == "filter":   m.toggle_filter(query[1])
                        except _q.Empty:
                            pass
                        try:
                            m.update_monitor()
                        except Exception:
                            pass
                t = threading.Thread(target=_update_loop,
                                     args=(m_obj, mon["queue"], mon["running"]),
                                     daemon=True)
                t.start()
                print("Monitor started (scan_mon mode — port 5066)")
            else:
                # For monitor-mode RPs: start the board's monitor loop first so
                # that port 5066 is available before the Monitor subprocess runs.
                self.start_loop(RP, "monitor")
                sleep(1.5)   # wait for port 5066 to start listening on the board
                print("Starting background process")
                self.p = mp.Process(
                    target=monitor,
                    args=(mon["running"], self.RPs[RP], mon["queue"], settings),
                )
                self.p.daemon = True
                self.p.start()
                print("monitoring process started")

    def filter_monitor(self, RP, on=True):
        if RP in self.monitors:
            if self.monitors[RP]["running"].value:
                self.monitors[RP]["queue"].put(("filter", on))

    def set_monitor_of_type(self, monitor_RP, Type="cavity"):
        if Type == "cavity":
            queue = self.monitors[monitor_RP]["queue"]
        else:
            queue = self.monitors[monitor_RP]["queue_err"]
        master_RP = self.find_master_RP(monitor_RP)
        settings = self.retrieve_monitor_settings(master_RP)
        queue.put(("settings", settings))

    def set_monitor(self, RP):
        if len(self.monitors) == 0:
            return
        monitor_RP = self.find_monitor_RP(RP)
        if monitor_RP is None:
            return
        if self.monitors[monitor_RP]["running"].value:
            self.set_monitor_of_type(monitor_RP, Type="cavity")
        elif self.monitors[monitor_RP]["running_err"].value:
            self.set_monitor_of_type(monitor_RP, Type="errors")

    def stop_monitor(self, RP):
        """
        Stops any monitoring (error_monitor or monitor) on the RedPitaya.

        For ``monitor``-mode RPs: also stops the board's reaction_loop (port 5065)
        so that the next call to start_monitor or start_error_monitor is not
        blocked by ``_check_for_loop``.
        """
        if RP in self.monitors:
            if self.monitors[RP]["running"].value:
                self.monitors[RP]["queue"].put(("stop", None))
                # For monitor-mode RPs the board loop was started by start_monitor;
                # stop it here so loop_running returns to False before the next phase.
                if self.RPs[RP].mode == "monitor" and self.RPs[RP].loop_running:
                    sleep(0.5)
                    self.stop_loop(RP)
            elif self.monitors[RP]["running_err"].value:
                self.monitors[RP]["queue_err"].put(("stop", None))
        else:
            print("Monitor not running!")

    ############## RP related functions #######################################

    def init_SG_settings(self, RP, laser, **kwargs):
        settings = self.RPs[RP].settings[laser]["peak_finder"]
        if "window_size" not in kwargs:
            kwargs["window_size"] = settings["window_size"]
        elif "order" not in kwargs:
            kwargs["order"] = settings["order"]
        return kwargs

    def set_peakfinder(self, RP, laser, peak_finder, **kwargs):
        value = kwargs
        value["name"] = peak_finder
        if peak_finder[:2] == "SG":
            value = self.init_SG_settings(RP, laser, **value)
        if peak_finder == "SG_deriv":
            value["deriv"] = 1
            if value["order"] < 1:
                value["order"] = 1
        elif peak_finder == "SG_maximum":
            value["deriv"] = 0
            if value["order"] < 0:
                value["order"] = 0
        self.update_setting(RP, laser, "peak_finder", value)

    def show_current(self, RP):
        """
        show the current data on the inputs of redpitaya RP in a plot.
        """
        acq = self.acquire(RP)
        plt.close(RP)
        if acq.size > 0:
            plt.figure(RP)
            plt.plot(acq[0], acq[1], label="Ch1")
            plt.plot(acq[0], acq[2], label="Ch2")
            plt.legend()
            plt.grid()
            plt.xlabel("Time [ms]")
            plt.ylabel("Signal [a.u.]")

    @_check_for_loop
    @_check_cavity_scanned
    def acquire(self, RP):
        acquisition = np.array(self.send(RP, "acquire"))
        return acquisition

    @_check_for_loop
    @_check_cavity_scanned
    def acquire_ch_n(self, RP, ch, n):
        if n > 100:
            dat = np.empty((0, int(2**14)))
            n_remaining = n
            while n_remaining > 0:
                if n_remaining >= 100:
                    action, value = "acquire_ch_n", f"{ch},100"
                else:
                    action, value = "acquire_ch_n", f"{ch},{n_remaining}"
                d = np.array(self.send(RP, action, value=value))
                dat = np.concatenate([dat, d])
                n_remaining -= 100
        else:
            action, value = "acquire_ch_n", f"{ch}, {n}"
            dat = self.send(RP, action, value=value)
        return dat

    ############## communication stuff ###########################

    def send(self, RP, action, value="Hello world!", loop_action=False):
        if RP in self.RPs:
            if self.RPs[RP].mode == "ext_scan":
                return None
            loop = self.RPs[RP].loop_running
            return self.RPs[RP].send(
                self, action, value=value, loop_action=loop_action, loop=loop
            )
        else:
            print(f"{RP} not found.")
            return None

    def connect(self, RP):
        if RP in self.RPs:
            if not self.RPs[RP].connected:
                print("connecting...")
                self.RPs[RP].start_host_server()
                sleep(5)
            else:
                print(f"{RP} already connected.")
        else:
            print(f"{RP} not found.")

    def connect_all(self):
        print("connecting...")
        for RP in self.RPs:
            t = threading.Thread(
                target=self.RPs[RP].start_host_server, daemon=True
            )
            t.start()
            t.join()
        sleep(5)

    def disconnect(self, RP):
        if self.RPs[RP].connected:
            self.send(RP, "stop")
            self.RPs[RP].connected = False
        else:
            print(f"{RP} not connected.")


######################## Monitoring Classes ###################################


class Monitor(Sender):
    """
    Live cavity signal monitor with dark Catppuccin theme.

    Class flag
    ----------
    ``Monitor.show_trigger``  (default ``True``)
        ``True``  → dual-axis layout: IN1 cavity (top, 3×height) + IN2 trigger (bottom)
        ``False`` → single-axis: IN1 cavity only

    Set the flag *before* calling ``start_monitor()``:
        >>> from lockclient import Monitor
        >>> Monitor.show_trigger = False
    """

    # ── class-level flag: set to False before start_monitor() to hide trigger ──
    show_trigger = True

    def __init__(self, RP, queue, settings, bool_var=None):
        Sender.__init__(self)
        self.mode = "monitor"
        self.RP = RP
        self.queue = queue
        self._fig = None
        self.settings = settings
        self.monitor_running = bool_var  # a shared boolean variable
        self.filter = False
        # snapshot the class flag at construction time so the child process
        # gets a stable value even if the parent later changes it
        self._show_trigger = self.__class__.show_trigger

    ################ Cavity monitoring related functions ######################
    def stop_monitor(self, event):
        self.monitor_running.value = (
            False  # this is used to stop the monitor if figure is closed!
        )

    def start_monitor(self):
        """
        starts the monitoring of the cavity signal on the redpitaya RP.
        """
        self.setup_monitor()
        self.monitor_running.value = True
        while self.monitor_running.value:
            sleep(10e-3)
            try:
                query = self.queue.get_nowait()
                if query[0] == "stop":
                    self.stop_monitor(None)
                if query[0] == "settings":
                    self.update_settings(query[1])
                if query[0] == "filter":
                    self.toggle_filter(query[1])
            except queue.Empty:
                pass
            self.update_monitor()
        self.close()
        return

    def _raw_send(self, action, value="0"):
        """
        Send a request on a dedicated persistent socket to port 5065 and
        read the complete response, handling TCP fragmentation.

        On first call, opens a private socket to port 5065 (separate from
        the scan loop's lsock to avoid race conditions) and caches it as
        self._mon_sock / self._mon_sel for reuse on every subsequent call.
        This eliminates per-call selector creation overhead and the shared
        lsock race condition.
        """
        import json as _json, struct as _struct, selectors as _sel, io as _io
        import socket as _socket
        from time import time as _time

        # Use the shared lsock — port 5065 only accepts one connection.
        # A threading.Lock prevents simultaneous use by monitor + scan loop.
        import threading as _threading
        if not hasattr(self.RP, "_lsock_lock"):
            self.RP._lsock_lock = _threading.Lock()

        lsock = self.RP.lsock

        if not hasattr(self, "_mon_sel") or self._mon_sel is None:
            self._mon_sel = _sel.DefaultSelector()
            self._mon_sel.register(lsock, _sel.EVENT_READ | _sel.EVENT_WRITE)

        # Build raw request bytes
        content = _json.dumps(
            {"action": action, "value": value}, ensure_ascii=False
        ).encode("utf-8")
        jh_bytes = _json.dumps({
            "byteorder": "little", "content-type": "text/json",
            "content-encoding": "utf-8", "content-length": len(content)
        }, ensure_ascii=False).encode("utf-8")
        raw = _struct.pack(">H", len(jh_bytes)) + jh_bytes + content

        # Acquire lock for the FULL send+receive cycle.
        # This prevents the monitor thread and LockClient from interleaving
        # bytes on the single shared lsock connection to port 5065.
        with self.RP._lsock_lock:
            local_sel = _sel.DefaultSelector()
            local_sel.register(lsock, _sel.EVENT_READ | _sel.EVENT_WRITE)
            sent = False
            buf = b""
            expected_len = None

            t0 = _time()
            while _time() - t0 < 10:
                events = local_sel.select(timeout=0.05)
                for key, mask in events:
                    if mask & _sel.EVENT_WRITE and not sent:
                        lsock.send(raw)
                        local_sel.modify(lsock, _sel.EVENT_READ)
                        sent = True
                    elif mask & _sel.EVENT_READ:
                        try:
                            while True:
                                chunk = lsock.recv(2**18)
                                if not chunk:
                                    break
                                buf += chunk
                        except BlockingIOError:
                            pass

                if not sent:
                    continue

                if expected_len is None and len(buf) >= 2:
                    jh_len = _struct.unpack(">H", buf[:2])[0]
                    if len(buf) >= 2 + jh_len:
                        jh = _json.loads(buf[2:2 + jh_len].decode("utf-8"))
                        expected_len = 2 + jh_len + jh["content-length"]

                if expected_len is not None and len(buf) >= expected_len:
                    break

            local_sel.unregister(lsock)
            local_sel.close()

        if not buf or expected_len is None or len(buf) < expected_len:
            return None

        jh_len   = _struct.unpack(">H", buf[:2])[0]
        jh       = _json.loads(buf[2:2 + jh_len].decode("utf-8"))
        data     = buf[2 + jh_len: 2 + jh_len + jh["content-length"]]
        response = _json.loads(
            _io.TextIOWrapper(
                _io.BytesIO(data),
                encoding=jh["content-encoding"],
                newline=""
            ).read()
        )
        return response.get("result")

    def _close_mon_sock(self):
        """No persistent monitor socket to close — nothing to do."""
        pass

    def _port5066_send(self, ch=0):
        """Connect to port 5066, send channel byte, receive cached ADC data."""
        import socket as _s, struct as _st, json as _j
        ip = self.RP.addr[0]
        sock = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((ip, 5066))
        sock.sendall(bytes([ch]))
        hdr = b""
        while len(hdr) < 4:
            chunk = sock.recv(4 - len(hdr))
            if not chunk: break
            hdr += chunk
        if len(hdr) < 4:
            sock.close(); return None
        length = _st.unpack(">I", hdr)[0]
        payload = b""
        while len(payload) < length:
            chunk = sock.recv(min(65536, length - len(payload)))
            if not chunk: break
            payload += chunk
        sock.close()
        return _j.loads(payload.decode("utf-8")) if len(payload) == length else None

    def acquire(self):
        # Port 5066: dedicated monitor port, no Sender event loop, no conflicts.
        a = self._port5066_send(ch=0)
        if a is None:
            raise RuntimeError("Port 5066 returned no data — is start_scan running?")
        dur, self.acquisition = a
        self.times = np.linspace(0, dur, 2**14)  # in ms
        self.data = dict(
            Cavity=np.array([self.times[1:], np.array(self.acquisition)[1:]]),
        )
        if self._show_trigger:
            try:
                a1 = self._port5066_send(ch=1)
                if a1 is None: raise RuntimeError("ch1 None")
                _, ch1 = a1
                self._trigger_data = np.array(ch1)
            except Exception:
                self._trigger_data = np.zeros(len(self.acquisition))
            self.data["Trigger"] = np.array(
                [self.times[1:], self._trigger_data[1:]]
            )
        if self.filter:
            self.filter_signals()

    def toggle_filter(self, on=True):
        if on and not self.filter:
            self.filter = True
            self.acquire()
            self.plot_filtered_signals()
        elif not on and self.filter:
            self.filter = False
            self.remove_filtered_signals()

    def filter_signals(self):
        for laser in self.settings:
            kwargs = deepcopy(self.settings[laser]["peak_finder"])
            name = kwargs.pop("name")
            if laser == "Master":
                r = self.settings[laser]["range"][1]
            else:
                r = self.settings[laser]["range"]
            if name[:2] == "SG":
                m = SG_array(**kwargs)
                x, y = self.times[1:], self.acquisition[1:]
                self.data[laser + "filtered"] = SG_filter(x, y, r, m=m)

    def set_monitor_title(self):
        if type(self.settings["Master"]) == str:
            title = f'Cavity Monitor - {self.settings["Master"]}'
        else:
            title = f"Cavity Monitor - {self.RP.label}"
        try:
            dec = self.settings["Master"]["dec"]
            period_ms = 8e-9 * 16384 * dec * 1e3
            title += f"   |   dec={dec}   period={period_ms:.3f} ms"
        except Exception:
            pass
        trig_label = "  [trigger ON]" if self._show_trigger else "  [trigger OFF]"
        self._fig.canvas.manager.set_window_title(title + trig_label)
        try:
            self._fig.suptitle(
                title, color="#cdd6f4", fontsize=10, y=0.99, fontweight="bold"
            )
        except Exception:
            pass

    def _decorate_figure(self):
        # ylim from actual signal range with 15% padding
        acq = self.data["Cavity"][1]
        span = max(acq) - min(acq) if max(acq) != min(acq) else 1.0
        ymin = min(acq) - span * 0.15
        ymax = max(acq) + span * 0.15
        self._ax.set_ylim(ymin, ymax)
        self._ax.set_ylabel("IN1 — Cavity  [V]", color="#cdd6f4", fontsize=9)
        self._ax.legend(
            loc="upper right", fontsize=8,
            facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4"
        )
        if self._show_trigger and hasattr(self, "_ax2") and self._ax2 is not None:
            trig = self.data["Trigger"][1]
            t_span = max(trig) - min(trig) if max(trig) != min(trig) else 1.0
            self._ax2.set_ylim(min(trig) - t_span * 0.3, max(trig) + t_span * 0.3)
            self._ax2.set_ylabel("IN2 — Trigger  [V]", color="#cdd6f4", fontsize=9)
            self._ax2.set_xlabel("Time  [ms]", color="#cdd6f4", fontsize=9)
            self._ax2.legend(
                loc="upper right", fontsize=8,
                facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4"
            )
        else:
            self._ax.set_xlabel("Time  [ms]", color="#cdd6f4", fontsize=9)

    def _setup_figure(self):
        plt.style.use("dark_background")
        # ── dual-axis layout when trigger is enabled ──────────────────────────
        if self._show_trigger:
            self._fig, (self._ax, self._ax2) = plt.subplots(
                2, 1, figsize=(7 * golden, 7),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            )
            _axes = (self._ax, self._ax2)
        else:
            self._fig, self._ax = plt.subplots(1, 1, figsize=(7 * golden, 6))
            self._ax2 = None
            _axes = (self._ax,)
        # ── dark styling on all axes ──────────────────────────────────────────
        self._fig.patch.set_facecolor("#1e1e2e")
        for ax in _axes:
            ax.set_facecolor("#181825")
            ax.tick_params(colors="#cdd6f4", labelsize=9)
            ax.spines[:].set_color("#45475a")
            ax.grid(True, color="#313244", linewidth=0.6, linestyle="--")
        self.set_monitor_title()
        self.acquire()  # acquire the signal once
        self._lines = []
        # cavity trace — always on self._ax
        l0, = self._ax.plot(
            self.data["Cavity"][0], self.data["Cavity"][1],
            color="#4fc3f7", lw=1.0, label="IN1 — Cavity"
        )
        self._lines.append(l0)
        # trigger trace — only on self._ax2 when enabled
        if self._show_trigger and self._ax2 is not None:
            l1, = self._ax2.plot(
                self.data["Trigger"][0], self.data["Trigger"][1],
                color="#a5d6a7", lw=0.9, label="IN2 — Trigger"
            )
            self._lines.append(l1)
        self._decorate_figure()

    def setup_monitor(self):
        """
        sets the monitoring of the cavity signal up. Prepares a corresponding figure.
        Retries for up to 10s waiting for port 5066 to become available.
        """
        from time import sleep as _sl, time as _t
        t0 = _t()
        while True:
            try:
                self._setup_figure()
                break
            except Exception as _e:
                elapsed = _t() - t0
                if elapsed > 10:
                    raise RuntimeError(
                        "Monitor setup failed after 10s: {}\n"
                        "Make sure start_scan() is running first.".format(_e))
                print("Waiting for port 5066... ({:.1f}s)".format(elapsed))
                _sl(1.0)
        self.plot_settings()
        self._fig.canvas.draw()
        plt.show(block=False)
        self._bm = BlitManager(self._fig.canvas, animated_artists=self._lines)
        self._fig.canvas.mpl_connect("close_event", self.stop_monitor)

    def create_label(self, key):
        if "label" in self.settings[key]:
            label = f'{self.settings[key]["label"]} | {key}'
        else:
            label = key
        return label

    def plot_filtered_signals(self):
        for key, val in self.data.items():
            if key != "Cavity":
                l = self._ax.plot(val[0], val[1])[0]
                self._lines.append(l)
                self._bm.add_artist(l)

    def remove_filtered_signals(self):
        for j in range(len(self._lines[1:])):
            ref = self._lines[1:].pop(0)
            ref.remove()
            del ref
        for key in self.data:
            if key != "Cavity":
                self.data.pop(key)

    def plot_settings(self):
        self._setrefs = []
        ymin, ymax = self._ax.get_ylim()
        xlim = self._ax.get_xlim()
        _C_RANGE  = "#90caf9"   # pale blue — range spans
        _C_LOCKPT = "#ffb74d"   # amber     — Master lockpoint
        _C_SLAVE  = ["#ef9a9a", "#ce93d8", "#80cbc4"]
        slave_idx = 0
        for key, val in self.settings.items():
            if val["enabled"]:
                if key == "Master":
                    c_span = _C_RANGE
                    c_line = _C_LOCKPT
                    for R in val["range"]:
                        ref = self._ax.axvspan(
                            self.times[R[0]], self.times[R[1]],
                            alpha=0.18, facecolor=c_span, edgecolor="none"
                        )
                        self._setrefs.append(ref)
                else:
                    c_span = _C_SLAVE[slave_idx % len(_C_SLAVE)]
                    c_line = _C_SLAVE[slave_idx % len(_C_SLAVE)]
                    slave_idx += 1
                    R = val["range"]
                    ref = self._ax.axvspan(
                        self.times[R[0]], self.times[R[1]],
                        alpha=0.18, facecolor=c_span, edgecolor="none"
                    )
                    self._setrefs.append(ref)
                ref = self._ax.vlines(
                    [val["lockpoint"]], ymin, ymax,
                    color=c_line, lw=1.5, ls="--",
                    label=self.create_label(key)
                )
                self._setrefs.append(ref)
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ymin, ymax)
        self._ax.relim()
        self._ax.legend(
            loc="upper right", fontsize=8,
            facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4"
        )
        print("Settings added to plot", flush=True)

    def remove_settings(self):
        leg = self._ax.get_legend()
        if leg is not None:
            leg.remove()
        for j in range(len(self._setrefs)):
            ref = self._setrefs.pop(0)
            ref.remove()
            del ref

    def reset_background(self):
        for l in self._lines:
            l.set_data([], [])
        self._fig.canvas.draw()
        self._bm.on_draw(None)
        self.plot_lines()

    def update_settings(self, settings):
        self.acquire()
        self.settings = settings
        self.remove_settings()
        self.plot_settings()
        self.reset_background()

    def plot_lines(self):
        # plots the data lines — self._lines[0] is cavity, [1] is trigger (if enabled)
        self._lines[0].set_data(self.data["Cavity"][0], self.data["Cavity"][1])
        if self._show_trigger and len(self._lines) > 1 and "Trigger" in self.data:
            self._lines[1].set_data(self.data["Trigger"][0], self.data["Trigger"][1])
        self._bm.update()

    def update_monitor(self):
        self.acquire()
        self.plot_lines()

    def close(self):
        self.monitor_running.value = False
        self._close_mon_sock()
        self.stop_event_loop()


class ErrorMonitor(Sender):
    def __init__(self, RP, queue, settings, FSR=906, tmin=10e-3, bool_var=None):
        Sender.__init__(self)
        self.mode = "monitor"
        self.FSR = FSR
        self.tmin = tmin
        self.RP = RP
        self.queue = queue
        self._fig = None
        self.settings = settings
        self.monitor_running = bool_var

    def stop_monitor(self, event):
        self.monitor_running.value = False

    def close(self):
        self.monitor_running.value = False
        self.stop_event_loop()

    def start_monitor(self):
        self.setup_monitor()
        self.monitor_running.value = True
        while self.monitor_running.value:
            try:
                sleep(self.tmin)
                query = self.queue.get_nowait()
                if query[0] == "stop":
                    self.stop_monitor(None)
                if query[0] == "settings":
                    self.update_settings(query[1])
                if query[0] == "save":
                    self.save_errors(query[1])
            except queue.Empty:
                pass
            self.update_monitor()
        self.close()
        return

    def setup_monitor(self):
        self.RP.send(self, "update_settings", self.settings)
        self._setup_figure()
        self._t0 = perf_counter()
        self.times = []
        self._fig.canvas.draw()
        plt.show(block=False)
        self._bm = BlitManager(
            self._fig.canvas, animated_artists=list(self._lines.values())
        )
        self._fig.canvas.mpl_connect("close_event", self.stop_monitor)

    def _setup_figure(self):
        self._fig, self._ax = plt.subplots(1, 1, figsize=(5 * golden, 5))
        self.set_monitor_title()
        self._lines, self.errs = dict(), dict()
        for key, val in self.settings.items():
            l = self._ax.plot([], [], marker="o", label=key)[0]
            self._lines[key] = l
            self.errs[key] = []
        self._decorate_figure()

    def set_monitor_title(self):
        if type(self.settings["Master"]) == str:
            title = f'Error Monitor - {self.settings["Master"]}'
        else:
            title = f"Error Monitor - {self.RP.label}"
        self._fig.canvas.manager.set_window_title(title)

    def _decorate_figure(self):
        self._ax.set_ylim([-50, 50])
        self._ax.legend()
        self._ax.grid()
        self._ax.set_ylabel("Error [MHz]")
        self._ax.set_xlabel("Locking time [s]")

    def save_errors(self, filename):
        dat = deepcopy(self.errs)
        dat["times"] = self.times
        with open(f"{filename}.json", "w") as file:
            json.dump(dat, file, indent=4)

    def update_settings(self, settings):
        self.settings = settings
        self.RP.send(self, "update_settings", value=settings)
        return "done"

    def update_errs(self):
        new_errs = self.RP.send(self, "acquire_errs")
        for key in self.errs:
            if new_errs == "skipped":
                self.errs[key].append(np.nan)
            else:
                if key in new_errs:
                    self.errs[key].append(new_errs[key] * self.FSR)
                else:
                    self.errs[key].append(np.nan)

    def update_monitor(self):
        self.update_errs()
        self.times.append(perf_counter() - self._t0)
        for key, l in self._lines.items():
            if len(self.times) >= 300:
                i0 = -300
            else:
                i0 = 0
            l.set_xdata(self.times[i0:])
            l.set_ydata(self.errs[key][i0:])
            if len(self.times) > 2:
                self._ax.set_xlim(self.times[i0], self.times[-1])
                self._ax.relim()
        self._bm.update()


class RP_client(RP_connection):
    def __init__(self, address, settings, mode="lock"):
        # "scan_mon" is a PC-side concept — the board runs as plain "scan".
        # Pass the translated mode to RP_connection so upload_current() writes
        # RunLock.py with  RP_mode = 'scan'  (the only thing the board knows).
        _board_mode = "scan" if mode == "scan_mon" else mode
        RP_connection.__init__(self, address, mode=_board_mode)
        # Store the original PC-side mode so LockClient can still see "scan_mon".
        self.mode = mode
        self.settings = settings
        self.label = "Default"

######################## Inline event loop (Windows fix) #####################
# Windows SelectSelector is not thread-safe: a socket registered from one
# thread is invisible to another thread polling get_map(). The inline mixin
# processes the selector in the SAME thread that calls send(), bypassing the
# background thread entirely.

class _InlineEventLoopMixin:
    def start_event_loop(self):
        self.running = True   # no background thread

    def stop_event_loop(self):
        self.running = False

    def _process_sel_once(self):
        if not self.sel.get_map():
            return
        try:
            events = self.sel.select(timeout=0.01)
            for key, mask in events:
                message = key.data
                if message is not None:
                    try:
                        if self.mode == "monitor":
                            message.buffersize = int(2**18)
                        else:
                            message.buffersize = int(2**12)
                        message.process_events(mask)
                    except Exception:
                        import traceback as _tb
                        print("[InlineEventLoop] error:", flush=True)
                        _tb.print_exc()
                        message.close()
        except Exception:
            pass


class _InlineMonitor(_InlineEventLoopMixin, Monitor):
    pass


class _InlineErrorMonitor(_InlineEventLoopMixin, ErrorMonitor):
    pass
