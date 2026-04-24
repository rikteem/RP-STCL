"""
setup.py — RedPitayaSTCL compatibility and environment checker
==============================================================
All check logic lives here.  setup.ipynb imports and calls these
functions, keeping the notebook thin and readable.

Usage from a script / terminal:
    python setup.py --ip 192.168.0.101 --mode scan

Usage from the notebook:
    import setup
    setup.check_pc()
    info = setup.collect_board_info("Cav", "192.168.0.101", "scan")
    setup.check_board(info)
    setup.print_summary()
"""

import sys
import importlib
import pathlib
import subprocess
import shutil
import platform
import socket

# ── ANSI colour helpers ───────────────────────────────────────────────────────
PASS = "\033[92m  ✓\033[0m"
FAIL = "\033[91m  ✗\033[0m"
WARN = "\033[93m  ⚠\033[0m"
INFO = "\033[94m  i\033[0m"
DIV  = "─" * 68

# ── Global result stores (populated by the check functions) ───────────────────
pc_results    = []   # list of (symbol, label, detail)
board_results = {}   # board_name -> list of (symbol, label, detail)
board_infos   = {}   # board_name -> info dict


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rec_pc(sym, label, detail=""):
    """Record and print a PC check result."""
    pc_results.append((sym, label, detail))
    _print_result(sym, label, detail)


def _rec_board(name, sym, label, detail=""):
    """Record and print a board check result."""
    board_results.setdefault(name, []).append((sym, label, detail))
    _print_result(sym, label, detail)


def _print_result(sym, label, detail=""):
    line = "{} {}".format(sym, label)
    if detail:
        line += "  →  " + detail
    print(line)


def _ssh_connect(ip, user="root", password="root", port=22, timeout=8):
    """Return an open paramiko SSHClient, or None on failure."""
    try:
        import paramiko
    except ImportError:
        return None
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            ip, port=port, username=user, password=password,
            timeout=timeout,
            # Ubuntu 22.04 OpenSSH works fine with modern key algorithms;
            # disabled_algorithms kept empty for forward compatibility.
            disabled_algorithms={},
        )
        return client
    except Exception:
        return None


def _run_cmd(ssh, cmd, timeout=10):
    """Run a command over an open SSHClient and return stdout (stripped)."""
    try:
        _, stdout, _ = ssh.exec_command(cmd, timeout=timeout)
        return stdout.read().decode(errors="replace").strip()
    except Exception:
        return ""


def _ping(ip, timeout_s=2):
    """Return True if ip responds to a single ping."""
    ping_bin = shutil.which("ping")
    if not ping_bin:
        return False
    system = platform.system().lower()
    if system == "windows":
        cmd = [ping_bin, "-n", "1", "-w", str(int(timeout_s * 1000)), ip]
    elif system == "darwin":
        cmd = [ping_bin, "-c", "1", "-W", str(int(timeout_s * 1000)), ip]
    else:
        cmd = [ping_bin, "-c", "1", "-W", str(max(1, int(timeout_s))), ip]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# PC checks
# ─────────────────────────────────────────────────────────────────────────────

def check_pc():
    """
    Run all PC-side compatibility checks and populate pc_results.
    Prints each result as it runs.
    Returns True if no FAIL items were recorded, False otherwise.
    """
    pc_results.clear()

    print(DIV)
    print("  PC ENVIRONMENT CHECKS")
    print(DIV)

    # ── Python version ────────────────────────────────────────────────────────
    pv = sys.version_info
    pv_str = "{}.{}.{}".format(pv.major, pv.minor, pv.micro)
    if pv >= (3, 7):
        _rec_pc(PASS, "Python version",
                "{} (>= 3.7 required for f-strings and other PC-side syntax)".format(pv_str))
    else:
        _rec_pc(FAIL, "Python version",
                "{} — PC-side code requires Python >= 3.7. "
                "Upgrade at https://python.org".format(pv_str))

    # ── Required packages ─────────────────────────────────────────────────────
    REQUIRED = [
        ("paramiko",   "SSH/SFTP: uploads scripts and starts server on each RP"),
        ("numpy",      "Signal processing and array operations"),
        ("scipy",      "Golden-ratio figure sizing; Savitzky-Golay filter math"),
        ("matplotlib", "Qt5Agg backend for live cavity and error monitor windows"),
    ]
    for pkg, purpose in REQUIRED:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            _rec_pc(PASS, "Package: {:12s}".format(pkg),
                    "v{}  —  {}".format(ver, purpose))
        except ImportError:
            _rec_pc(FAIL, "Package: {:12s}".format(pkg),
                    "NOT FOUND  —  install with: pip install {}".format(pkg))

    # ── matplotlib Qt5Agg backend ─────────────────────────────────────────────
    try:
        import matplotlib
        try:
            matplotlib.use("Qt5Agg")
            import matplotlib.pyplot as _plt  # noqa: F401
            _rec_pc(PASS, "matplotlib backend",
                    "Qt5Agg available — live monitor windows will work")
        except Exception as e:
            _rec_pc(WARN, "matplotlib backend",
                    "Qt5Agg unavailable ({}). "
                    "Monitor windows will not open. "
                    "Install PyQt5: pip install PyQt5".format(str(e)[:100]))
    except ImportError:
        pass  # already recorded above

    # ── Repo root on sys.path ─────────────────────────────────────────────────
    repo_marker_files = [
        "lockclient.py", "communication.py", "general.py", "libclient.py",
    ]
    found_root = None
    for p in sys.path:
        if all(pathlib.Path(p, f).exists() for f in repo_marker_files):
            found_root = p
            break

    if found_root:
        _rec_pc(PASS, "Repo on sys.path", found_root)
    else:
        _rec_pc(FAIL, "Repo on sys.path",
                "lockclient.py / communication.py not found in sys.path. "
                "Add the repo root: sys.path.insert(0, '/path/to/RedPitayaSTCL')")

    # ── RP_side importable ────────────────────────────────────────────────────
    try:
        from RP_side.peak_finders import peak_finders, SG_array  # noqa: F401
        _rec_pc(PASS, "RP_side importable",
                "peak_finders module imported successfully")
    except ImportError as e:
        _rec_pc(FAIL, "RP_side importable",
                str(e) + "  —  ensure RP_side/__init__.py exists")

    # ── RP-side source files present ──────────────────────────────────────────
    rp_files = {
        "RP_Lock.py":      "Main locking loop (uploaded to board)",
        "RunLock.py":      "Entry point executed on board via SSH",
        "libserver.py":    "Server-side framed TCP protocol",
        "peak_finders.py": "Savitzky-Golay peak detection algorithms",
    }
    if found_root:
        rp_dir = pathlib.Path(found_root, "RP_side")
        for fname, desc in rp_files.items():
            fp = rp_dir / fname
            if fp.exists():
                _rec_pc(PASS, "RP_side/{:20s}".format(fname), desc)
            else:
                _rec_pc(FAIL, "RP_side/{:20s}".format(fname),
                        "Missing  —  check repo is complete (git status)")
    else:
        _rec_pc(WARN, "RP-side file check",
                "Skipped — repo root not found on sys.path")

    # ── settings directory ────────────────────────────────────────────────────
    if found_root:
        settings_dir = pathlib.Path(found_root, "settings")
        default_json = settings_dir / "Default.json"
        if default_json.exists():
            _rec_pc(PASS, "settings/Default.json",
                    "Default lock settings template present")
        else:
            _rec_pc(WARN, "settings/Default.json",
                    "Missing — LockClient will fail to create new settings files")

    print()
    failed = [r for r in pc_results if r[0] == FAIL]
    return len(failed) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Board info collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_board_info(name, ip, mode,
                       ssh_user="root", ssh_pass="root", ssh_port=22,
                       stcl_cmd_port=5000, stcl_loop_port=5065):
    """
    SSH into the board and collect all relevant hardware/software information.
    Returns an info dict.  Does NOT perform any pass/fail judgement — that is
    left to check_board().

    Parameters
    ----------
    name            : str   Label used in BOARDS dict (e.g. "Cav")
    ip              : str   Board IP address
    mode            : str   "scan", "lock", or "monitor"
    ssh_user/pass   : str   SSH credentials (default root/root)
    ssh_port        : int   SSH port (default 22)
    stcl_cmd_port   : int   STCL command port to check (default 5000)
    stcl_loop_port  : int   STCL loop port to check (default 5065)
    """
    info = {
        "name": name, "ip": ip, "mode": mode,
        "stcl_cmd_port": stcl_cmd_port,
        "stcl_loop_port": stcl_loop_port,
    }

    print(DIV)
    print("  Board: {}  ({})  mode={}".format(name, ip, mode))
    print(DIV)

    # ── ping ──────────────────────────────────────────────────────────────────
    reachable = _ping(ip)
    info["reachable"] = reachable
    if reachable:
        print("{} Ping {}  — reachable".format(PASS, ip))
    else:
        print("{} Ping {}  — UNREACHABLE".format(FAIL, ip))
        print("    Board not responding to ping. Check network / IP address.\n")
        board_infos[name] = info
        return info

    # ── SSH ───────────────────────────────────────────────────────────────────
    ssh = _ssh_connect(ip, user=ssh_user, password=ssh_pass, port=ssh_port)
    info["ssh_ok"] = ssh is not None
    if ssh:
        print("{} SSH login  root@{}".format(PASS, ip))
    else:
        print("{} SSH login  root@{}  — FAILED".format(FAIL, ip))
        print("    Check credentials and that sshd is running on port {}.\n".format(ssh_port))
        board_infos[name] = info
        return info

    # ── OS ────────────────────────────────────────────────────────────────────
    os_raw  = _run_cmd(ssh, "cat /etc/os-release")
    os_name, os_ver = "", ""
    for line in os_raw.splitlines():
        if line.startswith("PRETTY_NAME="):
            os_name = line.split("=", 1)[1].strip().strip('"')
        if line.startswith("VERSION_ID="):
            os_ver = line.split("=", 1)[1].strip().strip('"')
    info["os_name"] = os_name
    info["os_ver"]  = os_ver
    print("{} OS                : {}".format(INFO, os_name or "(unknown)"))

    # ── RP ecosystem version ──────────────────────────────────────────────────
    eco_raw  = _run_cmd(ssh, "cat /opt/redpitaya/version.txt 2>/dev/null")
    eco_ver  = ""
    for line in eco_raw.splitlines():
        if "Version:" in line:
            eco_ver = line.split("Version:", 1)[1].strip()
            break
    dot_ver  = _run_cmd(ssh, "cat /root/.version 2>/dev/null")
    info["eco_version"] = eco_ver
    info["dot_version"] = dot_ver
    print("{} RP ecosystem      : {}".format(INFO, eco_ver  or "(not found)"))
    print("{} RP .version file  : {}".format(INFO, dot_ver  or "(not found)"))

    # ── Hostname ──────────────────────────────────────────────────────────────
    hostname = _run_cmd(ssh, "hostname")
    info["hostname"] = hostname
    print("{} Hostname          : {}".format(INFO, hostname))

    # ── Uptime ───────────────────────────────────────────────────────────────
    uptime = _run_cmd(ssh, "uptime -p 2>/dev/null || uptime")
    info["uptime"] = uptime
    print("{} Uptime            : {}".format(INFO, uptime))

    # ── CPU / memory ─────────────────────────────────────────────────────────
    cpu_info = _run_cmd(ssh, "cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2")
    mem_info = _run_cmd(ssh, "free -m | awk '/Mem:/{print $2\" MB total, \"$3\" MB used\"}'")
    info["cpu_info"] = cpu_info.strip()
    info["mem_info"] = mem_info
    print("{} CPU               : {}".format(INFO, cpu_info.strip() or "(unknown)"))
    print("{} Memory            : {}".format(INFO, mem_info or "(unknown)"))

    # ── Python ────────────────────────────────────────────────────────────────
    py_ver  = _run_cmd(ssh, "python3 --version 2>&1")
    py_path = _run_cmd(ssh, "which python3")
    info["python_ver"]  = py_ver
    info["python_path"] = py_path
    print("{} Python            : {}  ({})".format(INFO, py_ver, py_path))

    # ── Python sys.path on board ──────────────────────────────────────────────
    py_syspath = _run_cmd(
        ssh, "python3 -c \"import sys; print('\\n'.join(sys.path))\" 2>/dev/null")
    info["python_syspath"] = py_syspath
    rp_lib_on_path = "/opt/redpitaya/lib/python" in py_syspath
    info["rp_lib_on_python_path"] = rp_lib_on_path
    print("{} /opt/redpitaya/lib/python on board sys.path: {}".format(
        INFO, "yes" if rp_lib_on_path else "NO"))

    # ── numpy on board ────────────────────────────────────────────────────────
    np_ver = _run_cmd(
        ssh, "python3 -c \"import numpy; print(numpy.__version__)\" 2>/dev/null")
    info["numpy_ver"] = np_ver
    print("{} numpy             : {}".format(INFO, np_ver or "not found"))

    # ── rp module (OS 2.x SWIG wrapper) ──────────────────────────────────────
    rp_mod_path = _run_cmd(
        ssh,
        "python3 -c \"import rp; print(rp.__file__)\" "
        "2>/dev/null")
    rp_mod_import = _run_cmd(
        ssh,
        "PYTHONPATH=/opt/redpitaya/lib/python/:$PYTHONPATH "
        "python3 -c \"import rp; rp.rp_Init(); rp.rp_Release(); print('ok')\" "
        "2>/dev/null")
    info["rp_mod_path"]   = rp_mod_path
    info["rp_mod_import"] = (rp_mod_import.strip() == "ok")
    print("{} rp module path    : {}".format(INFO, rp_mod_path or "not found"))
    print("{} rp module import  : {}".format(
        INFO, "ok" if info["rp_mod_import"] else "FAILED"))

    # ── /opt/redpitaya structure ──────────────────────────────────────────────
    opt_bin  = _run_cmd(ssh, "test -d /opt/redpitaya/bin  && echo yes || echo no")
    opt_fpga = _run_cmd(ssh, "test -d /opt/redpitaya/fpga && echo yes || echo no")
    info["opt_bin_ok"]  = (opt_bin  == "yes")
    info["opt_fpga_ok"] = (opt_fpga == "yes")
    print("{} /opt/redpitaya/bin : {}".format(
        INFO, "present" if info["opt_bin_ok"] else "MISSING"))
    print("{} /opt/redpitaya/fpga: {}".format(
        INFO, "present" if info["opt_fpga_ok"] else "MISSING"))

    # ── STCL files already uploaded to board ─────────────────────────────────
    stcl_files = {
        "RP_Lock.py":      "/root/RP_Lock.py",
        "RunLock.py":      "/root/RunLock.py",
        "libserver.py":    "/root/libserver.py",
        "peak_finders.py": "/root/peak_finders.py",
    }
    file_status = {}
    for fname, fpath in stcl_files.items():
        exists = _run_cmd(ssh, "test -f {} && echo yes || echo no".format(fpath))
        file_status[fname] = (exists == "yes")
        print("{} Board STCL file  : {}".format(
            PASS if exists == "yes" else WARN, fpath))
    info["stcl_files"] = file_status

    # ── Port availability ─────────────────────────────────────────────────────
    ss_out = _run_cmd(ssh, "ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null")
    info["ss_output"]      = ss_out
    info["port_cmd_free"]  = str(stcl_cmd_port)  not in ss_out
    info["port_loop_free"] = str(stcl_loop_port) not in ss_out
    print("{} Port {:4d} (cmd)  : {}".format(
        PASS if info["port_cmd_free"]  else WARN,
        stcl_cmd_port,
        "free" if info["port_cmd_free"]  else "IN USE"))
    print("{} Port {:4d} (loop) : {}".format(
        PASS if info["port_loop_free"] else WARN,
        stcl_loop_port,
        "free" if info["port_loop_free"] else "IN USE"))

    # ── Stale RunLock processes ───────────────────────────────────────────────
    stale_out = _run_cmd(ssh, "pgrep -a -f RunLock.py 2>/dev/null")
    info["stale_procs"] = stale_out if stale_out else None
    print("{} Stale RunLock     : {}".format(
        WARN if stale_out else PASS,
        stale_out if stale_out else "none"))

    # ── Active services (informational) ──────────────────────────────────────
    services = ["redpitaya_nginx", "redpitaya_scpi", "jupyter"]
    svc_status = {}
    for svc in services:
        status = _run_cmd(ssh, "systemctl is-active {} 2>/dev/null".format(svc))
        svc_status[svc] = status
        print("{} Service {:20s}: {}".format(INFO, svc, status or "unknown"))
    info["services"] = svc_status

    ssh.close()
    print()

    board_infos[name] = info
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Board compatibility checks
# ─────────────────────────────────────────────────────────────────────────────

def check_board(info, stcl_cmd_port=5000, stcl_loop_port=5065):
    """
    Evaluate the info dict returned by collect_board_info() and record
    pass/warn/fail results into board_results.
    Returns True if no FAIL items were recorded for this board.
    """
    name = info["name"]
    board_results.setdefault(name, [])

    def rb(sym, label, detail=""):
        _rec_board(name, sym, label, detail)

    print(DIV)
    print("  Compatibility checks: {}  ({})".format(name, info["ip"]))
    print(DIV)

    if not info.get("reachable"):
        rb(FAIL, "Board unreachable",
           "Check IP address ({}) and network connection".format(info["ip"]))
        print()
        return False

    if not info.get("ssh_ok"):
        rb(FAIL, "SSH login failed",
           "Check credentials (user={}) and that sshd is running".format("root"))
        print()
        return False

    # ── OS version ────────────────────────────────────────────────────────────
    os_ver = info.get("os_ver", "")
    if "22.04" in os_ver:
        rb(PASS, "OS version",
           "Ubuntu 22.04 LTS — matches the tested STCL OS 2.x configuration")
    elif "20.04" in os_ver:
        rb(WARN, "OS version",
           "Ubuntu 20.04 detected. OS 2.x STCL port targets 22.04. "
           "May work but is untested — verify the rp module is present.")
    elif os_ver:
        rb(WARN, "OS version",
           "Ubuntu {} detected. STCL OS 2.x port is tested on 22.04 only. "
           "Other versions may work but are not guaranteed.".format(os_ver))
    else:
        rb(WARN, "OS version",
           "Could not read /etc/os-release — verify manually over SSH")

    # ── RP ecosystem version ──────────────────────────────────────────────────
    eco = info.get("eco_version", "")
    dot = info.get("dot_version", "")
    version_str = eco or dot
    if version_str and (version_str.startswith("2.") or "2.0" in version_str):
        rb(PASS, "RP ecosystem version",
           "{} — OS 2.x confirmed. rp SWIG module supported.".format(version_str))
    elif version_str and "1.04" in version_str:
        rb(FAIL, "RP ecosystem version",
           "Found: '{}'. This is OS 1.04 — the OS 2.x port requires OS 2.x. "
           "Flash the SD card with OS 2.x from: "
           "https://downloads.redpitaya.com/downloads/STEMlab-125-1x/".format(version_str))
    elif version_str:
        rb(WARN, "RP ecosystem version",
           "Found: '{}' — verify this is OS 2.x compatible.".format(version_str))
    else:
        rb(WARN, "RP ecosystem version",
           "Not found at /opt/redpitaya/version.txt or /root/.version — "
           "verify OS version manually")

    # ── Python version on board ───────────────────────────────────────────────
    py_raw = info.get("python_ver", "")
    try:
        parts  = py_raw.replace("Python", "").strip().split(".")
        py_maj = int(parts[0])
        py_min = int(parts[1])
        if (py_maj, py_min) >= (3, 10):
            rb(PASS, "Board Python version",
               "{} — OS 2.x ships Python 3.10; RP-side code compatible".format(py_raw))
        elif (py_maj, py_min) >= (3, 7):
            rb(WARN, "Board Python version",
               "{} — minimum supported, but OS 2.x expects 3.10. "
               "rp module may still work; verify manually.".format(py_raw))
        else:
            rb(FAIL, "Board Python version",
               "{} — RP-side requires Python >= 3.7. "
               "This board cannot run the lock server.".format(py_raw))
    except Exception:
        rb(WARN, "Board Python version",
           "Could not parse '{}' — check python3 is installed".format(py_raw))

    # ── /opt/redpitaya/lib/python on board sys.path ───────────────────────────
    if info.get("rp_lib_on_python_path"):
        rb(PASS, "rp lib on board sys.path",
           "/opt/redpitaya/lib/python found — rp SWIG module will import correctly")
    else:
        rb(WARN, "rp lib on board sys.path",
           "/opt/redpitaya/lib/python NOT in python3 sys.path. "
           "RP_Lock.py uses PYTHONPATH=/opt/redpitaya/lib/python/:$PYTHONPATH "
           "in the SSH launch command — this is handled automatically by "
           "communication.py start_host_server(). No manual action needed unless "
           "you launch RunLock.py by hand.")

    # ── numpy on board ────────────────────────────────────────────────────────
    np_raw = info.get("numpy_ver", "")
    if np_raw:
        try:
            np_maj = int(np_raw.split(".")[0])
            if np_maj >= 2:
                rb(WARN, "Board numpy version",
                   "v{} — numpy >= 2.0 removed np.mat (used in peak_finders.py). "
                   "Confirm you are using the patched peak_finders.py "
                   "from this repo (np.mat replaced with np.array).".format(np_raw))
            else:
                rb(PASS, "Board numpy version",
                   "v{} — compatible".format(np_raw))
        except Exception:
            rb(WARN, "Board numpy version",
               "Could not parse '{}'".format(np_raw))
    else:
        rb(FAIL, "Board numpy",
           "Not found. Install: pip3 install numpy --break-system-packages")

    # ── rp module (OS 2.x SWIG wrapper) ──────────────────────────────────────
    if info.get("rp_mod_path"):
        if info.get("rp_mod_import"):
            rb(PASS, "rp module importable",
               info["rp_mod_path"] + "  (rp_Init() + rp_Release() succeeded)")
        else:
            rb(WARN, "rp module importable",
               "rp.py found at {} but import or init failed. "
               "Check FPGA overlay is loaded and board is not in use by "
               "another process.".format(info["rp_mod_path"]))
    else:
        rb(FAIL, "rp module",
           "/opt/redpitaya/lib/python/rp.py not found. "
           "This file is part of the RedPitaya OS 2.x installation. "
           "Verify the board is running OS 2.x and "
           "/opt/redpitaya/lib/python/ exists.")

    # ── /opt/redpitaya structure ──────────────────────────────────────────────
    if info.get("opt_bin_ok"):
        rb(PASS, "/opt/redpitaya/bin",  "present")
    else:
        rb(FAIL, "/opt/redpitaya/bin",
           "Missing — RP OS may not be correctly installed")
    if info.get("opt_fpga_ok"):
        rb(PASS, "/opt/redpitaya/fpga", "present")
    else:
        rb(WARN, "/opt/redpitaya/fpga",
           "Missing — FPGA bitfile directory not found")

    # ── STCL files on board ───────────────────────────────────────────────────
    for fname, present in info.get("stcl_files", {}).items():
        if present:
            rb(PASS, "Board STCL file: {}".format(fname))
        else:
            rb(WARN, "Board STCL file: {}".format(fname),
               "Not yet uploaded. Run LockClient(RPs) — "
               "upload_current() will transfer it to /root/ via SFTP automatically.")

    # ── Port availability ─────────────────────────────────────────────────────
    cmd_port  = info.get("stcl_cmd_port",  stcl_cmd_port)
    loop_port = info.get("stcl_loop_port", stcl_loop_port)

    if info.get("port_cmd_free", True):
        rb(PASS, "Port {} free (STCL command)".format(cmd_port))
    else:
        rb(WARN, "Port {} in use (STCL command)".format(cmd_port),
           "A previous server is still running. "
           "On the board run: pkill -f RunLock.py")
    if info.get("port_loop_free", True):
        rb(PASS, "Port {} free (STCL loop)".format(loop_port))
    else:
        rb(WARN, "Port {} in use (STCL loop)".format(loop_port),
           "A previous lock/scan loop is still active. "
           "Stop it before starting a new one.")

    # ── Stale processes ───────────────────────────────────────────────────────
    if info.get("stale_procs"):
        rb(WARN, "Stale RunLock.py process",
           "Running PID(s): {}  — "
           "kill with: pkill -f RunLock.py".format(info["stale_procs"]))
    else:
        rb(PASS, "No stale RunLock.py process")

    # ── Service conflicts ─────────────────────────────────────────────────────
    svcs = info.get("services", {})
    scpi_active  = svcs.get("redpitaya_scpi",  "") == "active"
    nginx_active = svcs.get("redpitaya_nginx", "") == "active"

    if scpi_active:
        rb(WARN, "redpitaya_scpi service active",
           "SCPI server and STCL both use port 5000. "
           "Stop SCPI before running STCL: "
           "systemctl stop redpitaya_scpi")
    else:
        rb(PASS, "redpitaya_scpi not running",
           "No port 5000 conflict with STCL server")

    if nginx_active:
        rb(INFO, "redpitaya_nginx active",
           "Web UI is running — not a conflict for STCL, "
           "but stop it if you need to free resources: "
           "systemctl stop redpitaya_nginx")

    print()
    failed = [r for r in board_results.get(name, []) if r[0] == FAIL]
    return len(failed) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    """
    Print a consolidated pass/warn/fail summary for PC and all boards.
    Returns True if everything passed (no FAILs anywhere).
    """
    all_ok = True

    print("\n" + "═" * 68)
    print("  SUMMARY")
    print("═" * 68)

    def _section(label, results):
        nonlocal all_ok
        fails  = [r for r in results if r[0] == FAIL]
        warns  = [r for r in results if r[0] == WARN]
        passes = [r for r in results if r[0] == PASS]
        status = FAIL if fails else (WARN if warns else PASS)
        print("\n{} {}".format(status, label))
        print("    {} passed  |  {} warnings  |  {} failed".format(
            len(passes), len(warns), len(fails)))
        if fails:
            all_ok = False
            print("    Failed:")
            for _, lbl, det in fails:
                print("      {} {}  →  {}".format(FAIL, lbl, det))
        if warns:
            print("    Warnings:")
            for _, lbl, det in warns:
                print("      {} {}  →  {}".format(WARN, lbl, det))

    _section("PC environment", pc_results)
    for bname, bresults in board_results.items():
        ip = board_infos.get(bname, {}).get("ip", "?")
        _section("Board: {}  ({})".format(bname, ip), bresults)

    print("\n" + "═" * 68)
    if all_ok:
        print("\033[92m  ✓  All checks passed — system ready for STCL operation.\033[0m")
    else:
        print("\033[91m  ✗  One or more checks FAILED — "
              "resolve the issues above before running STCL.\033[0m")
    print("═" * 68 + "\n")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RedPitayaSTCL setup and compatibility checker")
    parser.add_argument("--ip",   default=None,
                        help="Board IP address (omit to run PC checks only)")
    parser.add_argument("--name", default="Board",
                        help="Board label (default: Board)")
    parser.add_argument("--mode", default="scan",
                        choices=["scan", "lock", "monitor"],
                        help="Board mode (default: scan)")
    parser.add_argument("--user", default="root",  help="SSH username")
    parser.add_argument("--pass", default="root",  dest="password",
                        help="SSH password")
    args = parser.parse_args()

    check_pc()

    if args.ip:
        info = collect_board_info(
            args.name, args.ip, args.mode,
            ssh_user=args.user, ssh_pass=args.password)
        check_board(info)

    print_summary()
