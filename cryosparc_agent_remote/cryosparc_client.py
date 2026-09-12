# Creates authenticated cryosparc-tools clients for API-based job operations.
import os
import re
from pathlib import Path
from typing import Dict, List, Set


CRYOSPARC_CONFIG = Path("/ssd1/linweifan/cryosparc/cryosparc_master/config.sh")
DEFAULT_CRYOSPARC_HOST = "localhost"
DEFAULT_CRYOSPARC_BASE_PORT = 61000
CRYOSPARC_PROXY_BYPASS_HOSTS = {"localhost", "127.0.0.1", "admin", "172.16.1.2"}


def read_cryosparc_config() -> Dict[str, str]:
    """Read simple exported variables from CryoSPARC config.sh."""
    values: Dict[str, str] = {}
    if not CRYOSPARC_CONFIG.exists():
        return values

    export_re = re.compile(r'^export\s+([A-Za-z_][A-Za-z0-9_]*)=(["\']?)(.*?)\2$')
    for line in CRYOSPARC_CONFIG.read_text().splitlines():
        match = export_re.match(line.strip())
        if match:
            values[match.group(1)] = match.group(3)
    return values


def ensure_cryosparc_no_proxy(host: str) -> None:
    """Bypass HTTP proxy only for CryoSPARC-local hosts in this process."""
    bypass_hosts = sorted(CRYOSPARC_PROXY_BYPASS_HOSTS | ({host} if host else set()))
    for env_key in ("NO_PROXY", "no_proxy"):
        current = os.getenv(env_key, "")
        items = [item.strip() for item in current.split(",") if item.strip()]
        seen: Set[str] = set()
        merged: List[str] = []
        for item in items + bypass_hosts:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        os.environ[env_key] = ",".join(merged)


def cryosparc_client(
    host: str = DEFAULT_CRYOSPARC_HOST,
    base_port: int = DEFAULT_CRYOSPARC_BASE_PORT,
):
    """Create a cryosparc-tools client using env vars or config.sh credentials."""
    from cryosparc.tools import CryoSPARC

    ensure_cryosparc_no_proxy(host)
    config = read_cryosparc_config()
    license_id = os.getenv("CRYOSPARC_LICENSE_ID") or config.get("CRYOSPARC_LICENSE_ID")
    email = os.getenv("CRYOSPARC_EMAIL")
    password = os.getenv("CRYOSPARC_PASSWORD")

    if email and password:
        return CryoSPARC(host=host, base_port=base_port, email=email, password=password)
    if license_id:
        return CryoSPARC(host=host, base_port=base_port, license=license_id)
    return CryoSPARC(host=host, base_port=base_port)
