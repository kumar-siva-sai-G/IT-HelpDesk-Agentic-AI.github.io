"""
scripts/seed_kb.py
Seed ChromaDB with IT incident history and fix playbooks.

Run once before starting the system:
    python scripts/seed_kb.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.rag_tools import upsert_batch, get_collection_stats
from loguru import logger


# ── Past IT Incidents (for Root-Cause Diagnoser) ──────────────────────────────

INCIDENTS = [
    {
        "id": "inc_001",
        "text": (
            "Incident: Lab server lab-server-101 became unreachable. "
            "Root cause: OOM killer terminated the lab-service process after memory usage hit 98%. "
            "The service was inactive. Ping to server succeeded but service port was closed. "
            "Resolution: Restarted service, added swap space, set memory limits. "
            "Category: service_down, memory"
        ),
        "metadata": {"category": "service_down", "resource": "lab-server", "cause": "service_stopped"},
    },
    {
        "id": "inc_002",
        "text": (
            "Incident: Multiple students could not connect to lab-server-202 via SSH. "
            "Root cause: firewall rule was accidentally blocking port 22 after system update. "
            "Ping was successful but SSH timed out. "
            "Resolution: Updated iptables to allow SSH. "
            "Category: network, firewall, access_issue"
        ),
        "metadata": {"category": "access_issue", "resource": "lab-server", "cause": "firewall_block"},
    },
    {
        "id": "inc_003",
        "text": (
            "Incident: WiFi not working in Room 301. Students could not connect. "
            "Root cause: Access point AP-301 had crashed and was not broadcasting. "
            "Other rooms worked fine. Ping to AP-301 IP failed. "
            "Resolution: Power cycled access point. Contacted network team. "
            "Category: network, wifi, hardware"
        ),
        "metadata": {"category": "network", "resource": "wifi", "cause": "network_unreachable"},
    },
    {
        "id": "inc_004",
        "text": (
            "Incident: Database server db-primary stopped responding. "
            "Root cause: Disk usage reached 100%. Write operations failed. "
            "Service was technically running but could not write. "
            "Resolution: Cleared old logs, expanded disk volume. "
            "Category: service_down, disk_full"
        ),
        "metadata": {"category": "service_down", "resource": "database", "cause": "disk_full"},
    },
    {
        "id": "inc_005",
        "text": (
            "Incident: Printer in Lab 2 not responding to print jobs. "
            "Root cause: CUPS service stopped after kernel update. "
            "Printer was online on the network but jobs were queued and not processing. "
            "Resolution: Restarted CUPS service, cleared print queue. "
            "Category: service_down, software, printer"
        ),
        "metadata": {"category": "service_down", "resource": "printer", "cause": "service_stopped"},
    },
    {
        "id": "inc_006",
        "text": (
            "Incident: Student cannot login to lab workstation. Password rejected. "
            "Root cause: LDAP authentication service was unreachable — ldap-server was down for maintenance. "
            "Local login worked but domain credentials failed. "
            "Resolution: Restored LDAP service, cleared Kerberos ticket cache. "
            "Category: access_issue, authentication, ldap"
        ),
        "metadata": {"category": "access_issue", "resource": "ldap-server", "cause": "service_stopped"},
    },
    {
        "id": "inc_007",
        "text": (
            "Incident: High CPU usage on compute-node-05 causing slow response. "
            "Root cause: Runaway student process using 100% CPU due to infinite loop in submitted code. "
            "System was responsive but extremely slow. "
            "Resolution: Identified and killed runaway process via top/kill. Added CPU limits via cgroups. "
            "Category: performance, cpu, hardware"
        ),
        "metadata": {"category": "hardware", "resource": "compute-node", "cause": "resource_exhaustion"},
    },
    {
        "id": "inc_008",
        "text": (
            "Incident: NFS share /nfsshare/users inaccessible from client machines. "
            "Root cause: NFS server service crashed due to memory leak in NFS daemon. "
            "Mount point was visible but operations hung. Ping to NFS server succeeded. "
            "Resolution: Restarted nfs-kernel-server, remounted on clients. "
            "Category: service_down, network, nfs"
        ),
        "metadata": {"category": "service_down", "resource": "nfs-server", "cause": "service_stopped"},
    },
]

# ── Fix Playbooks (for Solution Agent) ───────────────────────────────────────

PLAYBOOKS = [
    {
        "id": "play_001",
        "text": (
            "Playbook: service_stopped — OOM (Out of Memory) + service inactive\n"
            "User steps:\n"
            "1. Wait 2 minutes — admin may restart the service remotely.\n"
            "2. Try reconnecting in 5 minutes.\n"
            "3. If still down, contact IT via this bot.\n"
            "Admin steps:\n"
            "1. sudo systemctl restart <service-name>\n"
            "2. Check memory: free -h\n"
            "3. If memory low: sudo fallocate -l 2G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile\n"
            "4. Check OOM events: dmesg | grep -i 'out of memory'\n"
            "5. Set memory limits in service unit: MemoryMax=2G in [Service] section"
        ),
        "metadata": {"cause": "service_stopped", "trigger": "oom"},
    },
    {
        "id": "play_002",
        "text": (
            "Playbook: firewall_block — SSH or port blocked by firewall rule\n"
            "User steps:\n"
            "1. Check if you are on the campus VPN.\n"
            "2. Try from a different network (mobile hotspot).\n"
            "3. Report exact error message to IT.\n"
            "Admin steps:\n"
            "1. Check iptables: sudo iptables -L -n | grep <port>\n"
            "2. Allow SSH: sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT\n"
            "3. For UFW: sudo ufw allow ssh\n"
            "4. Persist rules: sudo iptables-save > /etc/iptables/rules.v4\n"
            "5. Verify with: nmap -p 22 <server-ip>"
        ),
        "metadata": {"cause": "firewall_block", "trigger": "ssh_timeout"},
    },
    {
        "id": "play_003",
        "text": (
            "Playbook: network_unreachable — Ping fails, device offline\n"
            "User steps:\n"
            "1. Check your own network connection first.\n"
            "2. Try accessing a different server to confirm your network works.\n"
            "3. If only one resource is unreachable, report to IT with exact resource name.\n"
            "Admin steps:\n"
            "1. Check physical layer: Is the server/AP powered on?\n"
            "2. Check switch port: show interface status\n"
            "3. Check VLAN config: show vlan brief\n"
            "4. Restart network service: sudo systemctl restart networking\n"
            "5. Check IP assignment: ip addr show"
        ),
        "metadata": {"cause": "network_unreachable", "trigger": "ping_fail"},
    },
    {
        "id": "play_004",
        "text": (
            "Playbook: disk_full — 100% disk usage\n"
            "User steps:\n"
            "1. Delete any large temporary files you created.\n"
            "2. Check your home directory quota.\n"
            "Admin steps:\n"
            "1. Find large files: du -sh /* 2>/dev/null | sort -rh | head -20\n"
            "2. Clean old logs: sudo journalctl --vacuum-size=500M\n"
            "3. Remove old packages: sudo apt autoremove && sudo apt clean\n"
            "4. Check for large temp files: du -sh /tmp /var/tmp\n"
            "5. Expand volume if needed (cloud): resize disk in provider console then: sudo resize2fs /dev/<device>"
        ),
        "metadata": {"cause": "disk_full", "trigger": "write_fail"},
    },
    {
        "id": "play_005",
        "text": (
            "Playbook: resource_exhaustion — High CPU / runaway process\n"
            "User steps:\n"
            "1. Save your work if possible.\n"
            "2. Close any computationally heavy programs.\n"
            "Admin steps:\n"
            "1. Identify process: top -b -n1 | head -20\n"
            "2. Kill if necessary: sudo kill -9 <PID>\n"
            "3. Set CPU limits for students: cgcreate -g cpu:students && cgset -r cpu.cfs_quota_us=50000 students\n"
            "4. Long-term: use slurm/PBS for job scheduling on compute nodes"
        ),
        "metadata": {"cause": "resource_exhaustion", "trigger": "high_cpu"},
    },
    {
        "id": "play_006",
        "text": (
            "Playbook: access_issue — LDAP/Auth service down\n"
            "User steps:\n"
            "1. Try using local credentials if available.\n"
            "2. Wait 10 minutes — auth service may be restarting.\n"
            "Admin steps:\n"
            "1. Check LDAP: sudo systemctl status slapd (or openldap)\n"
            "2. Test LDAP: ldapsearch -x -H ldap://ldap-server -b dc=lab,dc=local\n"
            "3. Restart: sudo systemctl restart slapd\n"
            "4. Clear Kerberos cache on clients: sudo kdestroy -A\n"
            "5. Check PAM config: cat /etc/pam.d/common-auth"
        ),
        "metadata": {"cause": "service_stopped", "trigger": "auth_fail"},
    },
]


# ── Seed function ─────────────────────────────────────────────────────────────

def seed_all():
    logger.info("=" * 50)
    logger.info("Seeding ChromaDB Knowledge Base")
    logger.info("=" * 50)

    # Seed incidents
    n_incidents = upsert_batch(INCIDENTS, collection_name="incidents")
    logger.success(f"✅ Seeded {n_incidents} incidents into 'incidents' collection")

    # Seed playbooks
    n_playbooks = upsert_batch(PLAYBOOKS, collection_name="playbooks")
    logger.success(f"✅ Seeded {n_playbooks} playbooks into 'playbooks' collection")

    # Stats
    stats = get_collection_stats()
    logger.info(f"KB Stats: {stats}")
    logger.success("KB seeding complete! System is ready.")


if __name__ == "__main__":
    seed_all()
