# -*- coding: utf-8 -*-

from RP_Lock import *
import os
host, port = '192.168.0.99', 5000
Lock = RP_Server(host, port, 5065, RP_mode = 'scan_mon')
Lock.setup_server(loop=False)
Lock.start_server()
