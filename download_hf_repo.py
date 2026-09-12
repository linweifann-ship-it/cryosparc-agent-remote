#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import threading
import time
import urllib.parse

import requests


CHUNK_SIZE = 8 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model repo with resume support.")
    parser.add_argument("--repo", required=True, help="Repo id like Qwen/Qwen3.6-27B")
    parser.add_argument("--target-dir", required=True, help="Local directory for downloaded files")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent file downloads")
    return parser.parse_args()


def main():
    args = parse_args()
    api_url = f"https://huggingface.co/api/models/{args.repo}"
    base_url = f"https://huggingface.co/{args.repo}/resolve/main"
    print_lock = threading.Lock()

    os.makedirs(args.target_dir, exist_ok=True)

    def log(msg):
        with print_lock:
            print(time.strftime("%F %T"), msg, flush=True)

    def remote_size(session, url):
        response = session.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        return int(response.headers.get("Content-Length", "0"))

    def download_one(name):
        url = f"{base_url}/{urllib.parse.quote(name)}"
        dest = os.path.join(args.target_dir, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        with requests.Session() as session:
            total = remote_size(session, url)
            existing = os.path.getsize(dest) if os.path.exists(dest) else 0

            if total and existing == total:
                log(f"SKIP {name} already complete ({total} bytes)")
                return

            headers = {}
            mode = "wb"
            if existing and total and existing < total:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"
                log(f"RESUME {name} from {existing}/{total}")
            else:
                if existing and total and existing > total:
                    os.remove(dest)
                    existing = 0
                log(f"START {name} total={total}")

            with session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=60) as response:
                response.raise_for_status()
                if response.status_code == 200 and mode == "ab":
                    mode = "wb"
                    existing = 0

                downloaded = existing
                last_report = existing
                with open(dest, mode) as output:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        output.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_report >= 256 * 1024 * 1024:
                            last_report = downloaded
                            log(f"PROGRESS {name} {downloaded}/{total}")

            final_size = os.path.getsize(dest)
            if total and final_size != total:
                raise RuntimeError(f"{name} size mismatch: got {final_size}, expected {total}")
            log(f"DONE {name} {final_size} bytes")

    with requests.Session() as session:
        data = session.get(api_url, timeout=60).json()
        files = sorted(sibling["rfilename"] for sibling in data["siblings"])

    small = [name for name in files if not name.endswith(".safetensors")]
    large = [name for name in files if name.endswith(".safetensors")]
    ordered = small + large
    log(f"QUEUE {len(ordered)} files")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, name): name for name in ordered}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                log(f"ERROR {name} {exc!r}")
                raise

    log("ALL_DONE")


if __name__ == "__main__":
    main()
