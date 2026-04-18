"""
tools/system_tools.py
Real OS-level diagnostic tools for the System Checker Agent.

Each function returns a structured dict.
ALLOW_REAL_SYSTEM_COMMANDS must be True in .env for real execution.
"""

import subprocess
import platform
import shlex
import re
from loguru import logger


def _run_cmd(cmd: str, timeout: int = 10) -> tuple[str, str, int]:
    """Run a shell command safely. Returns (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s", 1
    except FileNotFoundError as e:
        return "", f"Command not found: {e}", 127
    except Exception as e:
        return "", str(e), 1


def tool_ping(host: str, count: int = 4) -> dict:
    """
    Ping a host and return success/latency.

    Returns:
        {success, latency_ms, packet_loss_pct, details}
    """
    os_type = platform.system().lower()
    if os_type == "windows":
        cmd = f"ping -n {count} {host}"
    else:
        cmd = f"ping -c {count} -W 2 {host}"

    stdout, stderr, rc = _run_cmd(cmd)

    result = {
        "host": host,
        "success": rc == 0,
        "latency_ms": None,
        "packet_loss_pct": 100.0 if rc != 0 else 0.0,
        "details": stdout or stderr,
    }

    # Parse average latency from ping output
    avg_match = re.search(r"avg[^=]*=\s*[\d.]+/([\d.]+)", stdout)
    if avg_match:
        result["latency_ms"] = float(avg_match.group(1))

    # Parse packet loss
    loss_match = re.search(r"(\d+)%\s+packet loss", stdout)
    if loss_match:
        result["packet_loss_pct"] = float(loss_match.group(1))

    logger.debug(f"[PING] {host}: {'OK' if result['success'] else 'FAILED'}")
    return result


def tool_check_service(service_name: str) -> dict:
    """
    Check systemctl service status.

    Returns:
        {status, active, since, recent_errors, pid}
    """
    stdout, stderr, rc = _run_cmd(f"systemctl status {service_name}")

    result = {
        "service": service_name,
        "status": "unknown",
        "active": False,
        "since": None,
        "pid": None,
        "recent_errors": [],
        "raw": stdout,
    }

    if "active (running)" in stdout:
        result["status"] = "active"
        result["active"] = True
    elif "inactive" in stdout or "failed" in stdout:
        result["status"] = "inactive" if "inactive" in stdout else "failed"
        result["active"] = False

    # Parse "since" time
    since_match = re.search(r"since\s+(.+?);", stdout)
    if since_match:
        result["since"] = since_match.group(1).strip()

    # Parse PID
    pid_match = re.search(r"Main PID:\s+(\d+)", stdout)
    if pid_match:
        result["pid"] = int(pid_match.group(1))

    # Find error lines
    error_lines = [
        line.strip() for line in stdout.splitlines()
        if any(kw in line.lower() for kw in ["error", "failed", "killed", "oom"])
    ]
    result["recent_errors"] = error_lines[:5]

    logger.debug(f"[SERVICE] {service_name}: {result['status']}")
    return result


def tool_tail_logs(resource: str, lines: int = 30) -> dict:
    """
    Tail system logs for a resource.
    Checks: /var/log/syslog, journalctl, and resource-specific log paths.

    Returns:
        {tail, error_lines, warning_lines, source}
    """
    log_sources = [
        f"journalctl -u {resource} -n {lines} --no-pager",
        f"tail -n {lines} /var/log/{resource}.log",
        f"tail -n {lines} /var/log/syslog",
    ]

    for cmd in log_sources:
        stdout, stderr, rc = _run_cmd(cmd, timeout=5)
        if rc == 0 and stdout.strip():
            error_lines = [
                l.strip() for l in stdout.splitlines()
                if any(kw in l.lower() for kw in ["error", "fail", "kill", "oom", "crit"])
            ]
            warning_lines = [
                l.strip() for l in stdout.splitlines()
                if "warn" in l.lower()
            ]
            return {
                "tail": stdout[-2000:],  # cap at 2000 chars
                "error_lines": error_lines[:10],
                "warning_lines": warning_lines[:5],
                "source": cmd.split()[0],
            }

    return {
        "tail": "",
        "error_lines": [],
        "warning_lines": [],
        "source": "none",
        "error": "No readable log source found",
    }


def tool_check_disk(path: str = "/") -> dict:
    """
    Check disk usage for a given path.

    Returns:
        {path, total_gb, used_gb, free_gb, used_pct, alert}
    """
    stdout, stderr, rc = _run_cmd(f"df -BG {path}")

    result = {
        "path": path,
        "total_gb": None,
        "used_gb": None,
        "free_gb": None,
        "used_pct": None,
        "alert": False,
    }

    if rc == 0:
        lines = stdout.strip().splitlines()
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 5:
                try:
                    result["total_gb"] = float(parts[1].replace("G", ""))
                    result["used_gb"] = float(parts[2].replace("G", ""))
                    result["free_gb"] = float(parts[3].replace("G", ""))
                    pct_str = parts[4].replace("%", "")
                    result["used_pct"] = int(pct_str)
                    result["alert"] = result["used_pct"] > 85
                except (ValueError, IndexError):
                    pass

    logger.debug(f"[DISK] {path}: {result.get('used_pct', '?')}% used")
    return result


def tool_check_memory() -> dict:
    """
    Check system memory usage.

    Returns:
        {total_mb, used_mb, free_mb, available_mb, used_pct, alert}
    """
    stdout, stderr, rc = _run_cmd("free -m")

    result = {
        "total_mb": None,
        "used_mb": None,
        "free_mb": None,
        "available_mb": None,
        "used_pct": None,
        "alert": False,
    }

    if rc == 0:
        lines = stdout.strip().splitlines()
        for line in lines:
            if line.startswith("Mem:"):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        result["total_mb"] = int(parts[1])
                        result["used_mb"] = int(parts[2])
                        result["free_mb"] = int(parts[3])
                        result["available_mb"] = int(parts[6])
                        if result["total_mb"] > 0:
                            result["used_pct"] = round(
                                100 * result["used_mb"] / result["total_mb"]
                            )
                            result["alert"] = result["used_pct"] > 90
                    except (ValueError, IndexError):
                        pass
                break

    logger.debug(f"[MEMORY] {result.get('used_pct', '?')}% used")
    return result
