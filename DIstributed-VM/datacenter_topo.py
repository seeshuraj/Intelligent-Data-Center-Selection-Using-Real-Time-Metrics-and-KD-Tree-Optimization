#!/usr/bin/env python3

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import random

class DataCenterTopo(Topo):
    """All hosts attach to a single OVS standalone switch for full L2 connectivity."""

    def __init__(self, num_datacenters=50):
        super(DataCenterTopo, self).__init__()

        # Single OVS switch in standalone mode
        central = self.addSwitch('s0', failMode='standalone')

        # Coordinator host
        ctr = self.addHost('ctr', ip='10.0.0.1/24')
        self.addLink(ctr, central)

        # Attach each data center host directly to central switch
        for i in range(1, num_datacenters + 1):
            dc = f'dc{i}'
            dc_ip = f'10.0.0.{i+1}/24'
            self.addHost(dc, ip=dc_ip)

            # Random link parameters
            delay = f'{random.randint(1,100)}ms'
            bw = random.randint(10,1000)
            loss = random.randint(0,5)

            self.addLink(dc, central,
                         delay=delay,
                         bw=bw,
                         loss=loss,
                         use_htb=True)

def run_datacenter_network():
    """Run the single-switch data center network with OVS standalone switch."""
    setLogLevel('info')
    topo = DataCenterTopo(num_datacenters=50)
    net = Mininet(
        topo=topo,
        link=TCLink,
        switch=OVSSwitch,
        controller=None      # no external controller needed
    )
    net.start()
    info('*** Network started: 50 DCs + coordinator on OVS standalone switch\n')
    info('*** CLI commands: pingall, ctr ping -c5 dc1, etc.\n')
    CLI(net)
    net.stop()

# For `mn --custom datacenter_topo.py --topo datacenter`
topos = {'datacenter': DataCenterTopo}

if __name__ == '__main__':
    run_datacenter_network()
