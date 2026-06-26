# Creates authenticated cryosparc-tools clients for API-based job operations.
import os
import re
from pathlib import Path


CRYOSPARC_CONFIG = Path("/ssd1/linweifan/cryosparc/cryosparc_master/config.sh")
DEFAULT_CRYOSPARC_HOST = "localhost"
DEFAULT_CRYOSPARC_BASE_PORT = 61000


def read_cryosparc_config() -> dict[str, str]:
    """Read simple exported variables from CryoSPARC config.sh."""
    values: dict[str, str] = {}
    if not CRYOSPARC_CONFIG.exists():
        return values

    export_re = re.compile(r'^export\s+([A-Za-z_][A-Za-z0-9_]*)=(["\']?)(.*?)\2$')
    for line in CRYOSPARC_CONFIG.read_text().splitlines():
        match = export_re.match(line.strip())
        if match:
            values[match.group(1)] = match.group(3)
    return values


def cryosparc_client(
    host: str = DEFAULT_CRYOSPARC_HOST,
    base_port: int = DEFAULT_CRYOSPARC_BASE_PORT,
):
    """Create a cryosparc-tools client using env vars or config.sh credentials."""
    from cryosparc.tools import CryoSPARC

    config = read_cryosparc_config()
    license_id = os.getenv("CRYOSPARC_LICENSE_ID") or config.get("CRYOSPARC_LICENSE_ID")
    email = os.getenv("CRYOSPARC_EMAIL")
    password = os.getenv("CRYOSPARC_PASSWORD")

    if email and password:
        return CryoSPARC(host=host, base_port=base_port, email=email, password=password)
    if license_id:
        return CryoSPARC(host=host, base_port=base_port, license=license_id)
    return CryoSPARC(host=host, base_port=base_port)
