# collect_metrics.py

import subprocess
import re
from ftplib import FTP

def get_latency(ip):
    """Ping router and return average latency (ms)."""
    try:
        cmd = ["ping", "-c", "5", ip]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        match = re.search(r"rtt min/avg/max/mdev = ([\d\.]+)/([\d\.]+)/", output)
        if match:
            return float(match.group(2))  # avg latency
        return None
    except Exception as e:
        print(f"Latency error for {ip}: {e}")
        return None

def get_storage_via_ftp(ip, username="anonymous", password=""):
    """Connect via FTP and calculate used/free storage under shares/USB_Storage."""
    try:
        ftp = FTP(ip)
        ftp.login(username, password)
        ftp.cwd('shares')
        ftp.cwd('USB_Storage')

        total_gb = 57.2  # known USB size
        total_size = 0

        def crawl():
            nonlocal total_size
            items = []
            ftp.retrlines('LIST', items.append)
            for item in items:
                parts = item.split()
                name = parts[-1]
                if item.startswith('d') and name not in ['.', '..']:
                    try:
                        ftp.cwd(name)
                        crawl()
                        ftp.cwd('..')
                    except:
                        continue
                elif item.startswith('-'):
                    try:
                        total_size += int(parts[4])
                    except:
                        continue

        crawl()
        ftp.quit()

        used_gb = round(total_size / (1024 ** 3), 2)
        free_gb = round(total_gb - used_gb, 2)
        return total_gb, free_gb
    except Exception as e:
        print(f"FTP error for {ip}: {e}")
        return None, None
