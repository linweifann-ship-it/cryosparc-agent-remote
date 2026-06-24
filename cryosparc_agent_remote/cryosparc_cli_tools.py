# Wraps CryoSPARC CLI and cryosparc-tools calls behind structured Python helpers.
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


CRYOSPARCM = Path("/ssd1/linweifan/cryosparc/cryosparc_master/bin/cryosparcm")
CRYOSPARCW = Path("/ssd1/linweifan/cryosparc/cryosparc_worker/bin/cryosparcw")
CRYOSPARC_CONFIG = Path("/ssd1/linweifan/cryosparc/cryosparc_master/config.sh")
DEFAULT_CRYOSPARC_HOST = "localhost"
DEFAULT_CRYOSPARC_BASE_PORT = 61000


# Command helpers keep shell output predictable for MCP callers.
def run_command(cmd: list[str], timeout: int = 60) -> dict:
    """
    运行一个 shell 命令，并把结果转成结构化 dict。
    这样 Agent 不需要直接看复杂终端输出。
    """
    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"Command timed out after {timeout} seconds",
            "success": False,
            "timed_out": True,
            "command": cmd,
        }

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
        "command": cmd,
    }


def parse_json_output(command_result: dict) -> dict:
    """
    如果命令 stdout 是 JSON，就补充 parsed_stdout；否则保留原始输出。
    """
    parsed = dict(command_result)
    try:
        parsed["parsed_stdout"] = json.loads(command_result.get("stdout") or "")
    except json.JSONDecodeError:
        parsed["parsed_stdout"] = None
    return parsed


# CryoSPARC client setup reads credentials from environment or config.sh.
def read_cryosparc_config() -> dict[str, str]:
    """
    读取 CryoSPARC config.sh 中的简单 export 变量。
    """
    values: dict[str, str] = {}
    if not CRYOSPARC_CONFIG.exists():
        return values

    export_re = re.compile(r'^export\s+([A-Za-z_][A-Za-z0-9_]*)=(["\']?)(.*?)\2$')
    for line in CRYOSPARC_CONFIG.read_text().splitlines():
        match = export_re.match(line.strip())
        if match:
            values[match.group(1)] = match.group(3)
    return values


def cryosparc_status() -> dict:
    """
    检查 CryoSPARC master 服务状态。
    """
    return run_command([str(CRYOSPARCM), "status"])


def cryosparc_version() -> dict:
    """
    查看 CryoSPARC 版本。
    """
    return run_command([str(CRYOSPARCM), "version"])


def cryosparc_worker_gpulist() -> dict:
    """
    查看当前 worker 环境可见的 GPU。

    注意：当前实例使用 SLURM cluster target 时，master/login 节点可能没有
    本机 CUDA 设备；这种情况下会返回 CryoSPARC worker CLI 的错误信息。
    """
    return parse_json_output(
        run_command([str(CRYOSPARCW), "gpulist", "--format", "json"])
    )


def cryosparc_test_workers(
    project_uid: str,
    test: str = "launch",
    target: str | None = None,
    test_pytorch: bool = False,
    timeout: int = 600,
) -> dict:
    """
    在指定 CryoSPARC project 中运行 worker 验证作业。
    """
    if test not in {"all", "launch", "ssd", "gpu"}:
        return {
            "success": False,
            "error": "test must be one of: all, launch, ssd, gpu",
        }

    cmd = [str(CRYOSPARCM), "test", "workers", project_uid, "--test", test]
    if target:
        cmd.extend(["--target", target])
    if test_pytorch:
        cmd.append("--test-pytorch")
    return run_command(cmd, timeout=timeout)


def cryosparc_client(
    host: str = DEFAULT_CRYOSPARC_HOST,
    base_port: int = DEFAULT_CRYOSPARC_BASE_PORT,
):
    """
    创建 cryosparc-tools client。
    """
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


# Direct CryoSPARC operations used by the MCP tools.
def cryosparc_create_import_movies_job(
    project_uid: str,
    workspace_uid: str,
    blob_paths: str,
    title: str = "Import Movies",
    desc: str = "Created by cryosparc_agent MCP tool.",
    params: dict[str, Any] | None = None,
    host: str = DEFAULT_CRYOSPARC_HOST,
    base_port: int = DEFAULT_CRYOSPARC_BASE_PORT,
) -> dict:
    """
    用 cryosparc-tools 创建 Import Movies job，并设置 blob_paths。
    """
    try:
        job_params: dict[str, Any] = dict(params or {})
        job_params["blob_paths"] = blob_paths

        cs = cryosparc_client(host=host, base_port=base_port)
        project = cs.find_project(project_uid)
        job = project.create_job(
            workspace_uid,
            "import_movies",
            params=job_params,
            title=title,
            desc=desc,
        )
        return {
            "success": True,
            "project_uid": project_uid,
            "workspace_uid": workspace_uid,
            "job_uid": job.uid,
            "job_type": "import_movies",
            "title": title,
            "blob_paths": blob_paths,
            "params": job_params,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "project_uid": project_uid,
            "workspace_uid": workspace_uid,
            "job_type": "import_movies",
            "blob_paths": blob_paths,
        }


if __name__ == "__main__":
    print(cryosparc_status())
