"""Sophon-Realtime installer. The AI Pod Host owns downloads and device access."""
import concurrent.futures
import contextvars
import copy
import functools
import hashlib
import ipaddress
import json
import os
import pathlib
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor

from control.deployment_resources import (
    assess as assess_resources,
    authorization as resource_authorization,
    authorize as authorize_resources,
    release as release_resources,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "cloud.lazycat.aipod.sophon-realtime-4b"
VERSION = (
    json.loads((ROOT / "manifests/release.json").read_text())["version"]
    if (ROOT / "manifests/release.json").is_file()
    else "development"
)
PORT = 18009
HOST = os.environ.get("AIPOD_BACKEND_URL", "http://host.lzcapp:40794").rstrip("/") + "/model-host/v1"
DATA = pathlib.Path(os.environ.get("SOPHON_REALTIME_DATA_DIR", "/lzcapp/var/sophon-realtime-4b"))
HOME = pathlib.Path(os.environ.get("SOPHON_REALTIME_DOCUMENT_ROOT", "/lzcapp/run/mnt/home"))
MODEL = (
    json.loads((ROOT / "manifests/model.json").read_text())
    if (ROOT / "manifests/model.json").is_file()
    else {}
)
MAIN_SUBDOMAIN = json.loads((ROOT / "control/main_app.json").read_text())["subdomain"]
ACTIVE = {"queued", "deploying", "starting"}
LANGUAGE = contextvars.ContextVar("language", default="zh")
LOCK = threading.RLock()
RECORDS = {}
JOBS = {}
MUTATING = set()
POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class Canceled(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.@-]{1,128}", value) or value in (".", ".."):
        raise HTTPException(400, "Invalid identifier")
    return value


def user(request):
    value = request.headers.get("X-HC-User-ID", "")
    if not value:
        raise HTTPException(401, "Authentication required")
    return safe_id(value)


def device_path(serial):
    return "/devices/" + quote(safe_id(serial), safe="")


def settings(params):
    if params is None:
        params = {}
    if not isinstance(params, dict) or set(params) - {"modelProvider", "modelSize"}:
        raise HTTPException(400, "Unsupported model parameter")
    provider = params.get("modelProvider", "modelscope")
    if provider not in ("modelscope", "huggingface"):
        raise HTTPException(400, "modelProvider must be modelscope or huggingface")
    size = params.get("modelSize", "4B-Flash")
    if size not in ("4B-Flash",):
        raise HTTPException(400, "Unsupported model size")
    return {"modelProvider": provider, "modelSize": size}


def namespace(uid, params):
    return PACKAGE + "/model/" + hashlib.sha256(uid.encode()).hexdigest()[:16] + "/" + settings(params)["modelSize"]


def service_id(uid):
    return "sophonrt4b-" + hashlib.sha256(uid.encode()).hexdigest()[:12]


def storage(params):
    return "AI 模型/Sophon-Realtime-4B-Flash/" + settings(params)["modelSize"]


def model_for(params):
    model = MODEL.get("variants", {}).get(settings(params)["modelSize"])
    if not model:
        raise HTTPException(503, "The model release has not passed its publication gates")
    return model


def files(params):
    provider = params["modelProvider"]
    from control.artifact_layout import transport_name

    result = {}
    for model in model_for(params)["models"]:
        if model.get("role") != "model" or model["role"] in result:
            raise HTTPException(503, "Invalid model role")
        source = model["sources"][provider]
        repository, revision = source["repository"], source["revision"]
        if (
            not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
            or not re.fullmatch(r"[a-f0-9]{40}", revision)
        ):
            raise HTTPException(503, "Model source is not locked")
        prefix = source.get("prefix", "")
        if prefix and (not prefix.endswith("/") or any(x in ("", ".", "..") for x in prefix.rstrip("/").split("/"))):
            raise HTTPException(503, "Invalid model prefix")
        base = "https://huggingface.co" if provider == "huggingface" else "https://modelscope.cn/models"
        entries = []
        for item in source.get("files", model["files"]):
            if (
                not isinstance(item.get("size"), int)
                or item["size"] <= 0
                or not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", ""))
            ):
                raise HTTPException(503, "Invalid model artifact integrity")
            try:
                name = transport_name(item["path"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(503, "Invalid model artifact path") from exc
            entries.append(
                {
                    "name": name,
                    "size": item["size"],
                    "sha256": item["sha256"],
                    "url": f"{base}/{repository}/resolve/{revision}/{prefix}{item['path']}",
                }
            )
        if not entries or len({x["name"] for x in entries}) != len(entries):
            raise HTTPException(503, "Invalid model artifact list")
        result[model["role"]] = entries
    return result


def runtime_manifest():
    path = ROOT / "manifests/runtimes.json"
    if not path.is_file():
        raise HTTPException(503, "The runtime release has not passed its publication gates")
    return json.loads(path.read_text())


def target(device):
    name = device.get("model", "").lower()
    if "agx thor" in name or "t5000" in name:
        return "thor"
    if "agx orin" in name:
        return "orin"
    raise HTTPException(400, "Only AGX Thor and AGX Orin are supported")


def runtime(device, params=None):
    kind = target(device)
    item = copy.deepcopy(runtime_manifest().get(kind, {}))
    profile = item.get("profiles", {}).get(settings(params)["modelSize"])
    if not profile:
        raise HTTPException(503, "Model and device profile is not validated")
    item["launchParameters"] = profile
    compatibility = item.get("compatibility", {})
    if not all(compatibility.get(k) for k in ("architecture", "l4tVersion", "jetpackVersion")):
        raise HTTPException(503, "Runtime compatibility profile is not locked")
    if (
        device.get("goArch") != compatibility["architecture"]
        or device.get("l4tVersion") != compatibility["l4tVersion"]
        or device.get("platform", {}).get("jetpackVersion") != compatibility["jetpackVersion"]
    ):
        raise HTTPException(409, "This system version needs runtime compatibility validation before deployment")
    provider = settings(params)["modelProvider"]
    source = item.get("sources", {}).get(provider, {})
    if not re.fullmatch(r"[a-f0-9]{40}", source.get("revision", "")):
        raise HTTPException(503, "Runtime download source is not ready")
    item["url"] = source.get("url", "")
    if (
        not re.fullmatch(r"sha256:[a-f0-9]{64}", item.get("imageId", ""))
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", item.get("configImageId", ""))
        or not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", ""))
        or not re.fullmatch(r"[A-Za-z0-9_./:-]+", item.get("image", ""))
        or item.get("size", 0) <= 0
        or urlparse(item.get("url", "")).scheme != "https"
    ):
        raise HTTPException(503, "Runtime release is not ready")
    return {**item, "device": kind}


def host(uid, method, path, body=None, timeout=90):
    try:
        response = requests.request(
            method, HOST + path, json=body, headers={"X-HC-User-ID": uid}, timeout=(10, timeout)
        )
        result = response.json() if response.content else {}
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError("AI Pod Host connection failed") from exc
    if not response.ok:
        raise HTTPException(response.status_code, result)
    return result


def devices(uid):
    result = host(uid, "GET", "/devices")
    return result.get("devices", []) if isinstance(result, dict) else result


def device(uid, serial):
    item = next((d for d in devices(uid) if d.get("serialNumber") == serial), None)
    if not item:
        raise HTTPException(404, "Device not found")
    if not item.get("online"):
        raise HTTPException(409, "Device is offline")
    target(item)
    return item


def base_requirements(uid, serial, params, prepared=False):
    d = device(uid, serial)
    rt = runtime(d, params)
    resources = host(uid, "GET", device_path(serial) + "/resources")
    available = resources.get("memoryAvailableBytes", 0)
    required = rt.get("requiredMemoryBytesByModel", {}).get(settings(params)["modelSize"])
    if not isinstance(required, int) or required <= 0:
        raise HTTPException(503, "Model memory budget is not validated")
    required_disk = (
        (2 << 30)
        if prepared
        else rt["size"] + rt["imageSizeBytes"] + model_for(params)["modelBytes"] + (4 << 30)
    )
    disk_ok = not resources.get("diskAvailableKnown") or resources.get("diskAvailableBytes", 0) >= required_disk
    processes = host(uid, "GET", device_path(serial) + "/process/status")
    conflicts = [
        name for name in processes.get("services", {}) if name.startswith("sophonrt4b-") and name != service_id(uid)
    ]
    blocked = available < required or not disk_ok or bool(conflicts)
    reasons = []
    if available < required:
        reasons.append("Insufficient available memory")
    if not disk_ok:
        reasons.append("Insufficient available disk space")
    if conflicts:
        reasons.append("Another Sophon-Realtime deployment occupies this endpoint")
    return {
        "online": True,
        "blocked": blocked,
        "requiresConfirmation": False,
        "requiresStop": False,
        "blockReasons": reasons,
        "requiredMemoryBytes": required,
        "availableMemoryBytes": available,
        "requiredDiskBytes": required_disk,
        "availableDiskBytes": resources.get("diskAvailableBytes", 0),
        "_processStatus": processes,
        "resources": resources,
        "resourcesEstimated": True,
        "maxConcurrency": 8,
    }


def requirements(uid, serial, params, prepared=False):
    return assess_resources(
        base_requirements(uid, serial, params, prepared), uid, serial, host, device_path, service_id(uid)
    )


def validate_request(body):
    allowed = {
        "serialNumber",
        "serialNumbers",
        "mode",
        "parameters",
        "stopExisting",
        "stopProtected",
        "protectedProcessIds",
        "modelProvider",
    }
    if set(body) - allowed or body.get("mode", "realtime-voice") != "realtime-voice":
        raise HTTPException(400, "Unsupported deployment field or mode")
    params = settings(body.get("parameters"))
    if "modelProvider" in body and body["modelProvider"] != params["modelProvider"]:
        raise HTTPException(400, "Conflicting modelProvider values")


def persist():
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "records.tmp"
    path.write_text(json.dumps(RECORDS, ensure_ascii=False, indent=2))
    os.chmod(path, 0o600)
    path.replace(DATA / "records.json")


def load():
    path = DATA / "records.json"
    if path.exists():
        RECORDS.update(json.loads(path.read_text()))
        for record in RECORDS.values():
            if record["status"] in ACTIVE:
                record.update(status="failed", error="Installer restarted; retry to resume")
        persist()


def sanitize(value):
    if isinstance(value, str):
        value = re.sub(
            r"(?:registry\.[^\s\"']+|distribution\.[^\s\"']+|local/sophonrt-[A-Za-z0-9_./:-]+|/lzcapp/[^\s\"']+|sophonrt-[a-f0-9]{12})",
            "Sophon-Realtime-4B-Flash",
            value,
        )
        return value[:6000]
    if isinstance(value, dict):
        return {
            k: v if k in ("protectedProcessIds", "serialNumber", "id", "modelId") else sanitize(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def translated(value):
    chinese = {
        "Full-duplex voice": "全双工语音对话",
        "Confirmed model services must release memory before startup": "启动前需由已确认的模型服务释放内存",
        "A model service occupies the speech endpoint": "语音服务端口被其他模型占用",
        "Insufficient preparation disk space": "准备阶段可用磁盘空间不足",
        "Total device memory is below this model profile requirement": "设备总内存低于此模型配置要求",
        "Preparation complete; waiting for confirmed model resources to be released": "准备完成，等待已确认的模型释放资源",
        "Queued": "等待部署",
        "Checking deployment": "检查部署条件",
        "Downloading and verifying artifacts": "下载并校验制品",
        "Importing runtime image": "导入运行镜像",
        "Transferring verified model": "传输已校验模型",
        "Starting Sophon-Realtime": "启动 Sophon-Realtime",
        "Sophon-Realtime is ready": "Sophon-Realtime 已就绪",
        "Sophon-Realtime stopped": "Sophon-Realtime 已停止",
        "Deployment canceled": "部署已取消",
        "Authentication required": "需要登录",
        "Insufficient available memory": "可用内存不足",
        "Insufficient available disk space": "可用磁盘空间不足",
        "Another Sophon-Realtime deployment occupies this endpoint": "已有 Sophon-Realtime 部署占用此端口",
        "Device not found": "未找到设备",
        "Device is offline": "设备离线",
        "Deployment not found": "未找到部署记录",
        "Device already has an active deployment": "此设备正在部署模型",
        "Only failed deployments can be retried": "只能重试失败的部署",
        "Cancel the active deployment first": "请先取消正在进行的部署",
        "Deployment is not stopped": "部署尚未停止",
        "Origin not allowed": "不允许此来源",
        "Only AGX Thor and AGX Orin are supported": "仅支持 AGX Thor 与 AGX Orin",
        "Select 1–4 devices": "请选择 1 至 4 台设备",
    }
    english = {
        "模型与运行镜像下载源": "Model and runtime download source",
        "模型规格": "Model size",
    }
    if isinstance(value, str):
        if value.startswith("部署失败："):
            detail = value[len("部署失败：") :]
            return ("部署失败：" + chinese.get(detail, detail)) if LANGUAGE.get() == "zh" else "Deployment failed: " + detail
        if LANGUAGE.get() == "zh":
            return chinese.get(value, value)
        return english.get(value, value)
    if isinstance(value, dict):
        return {k: translated(v) for k, v in value.items()}
    if isinstance(value, list):
        return [translated(v) for v in value]
    return value


def public(record):
    result = {k: copy.deepcopy(v) for k, v in record.items() if k not in ("userId", "serviceId", "resourceReleaseEvidence")}
    name = pathlib.PurePosixPath(result.get("currentFile") or "").name
    if name.startswith("nested-"):
        from control.artifact_layout import transport_name

        paths = (
            item["path"]
            for variant in MODEL.get("variants", {}).values()
            for entry in variant["models"]
            for item in entry["files"]
        )
        result["currentFile"] = next((path for path in paths if transport_name(path) == name), name)
    return translated(sanitize(result))


def own(uid, rid):
    record = RECORDS.get(rid)
    if not record or record["userId"] != uid:
        raise HTTPException(404, "Deployment not found")
    return record


def mutates(function):
    @functools.wraps(function)
    def guarded(request, rid, *args, **kwargs):
        uid = user(request)
        with LOCK:
            own(uid, rid)
            if rid in MUTATING:
                raise HTTPException(409, "Another operation is in progress")
            MUTATING.add(rid)
        try:
            return function(request, rid, *args, **kwargs)
        finally:
            with LOCK:
                MUTATING.discard(rid)

    return guarded


def validate_action(body, record):
    if body is None:
        return
    if set(body) - {"stopExisting", "stopProtected", "protectedProcessIds", "parameters"}:
        raise HTTPException(400, "Unsupported lifecycle field")
    if "parameters" in body and settings(body["parameters"]) != settings(record["parameters"]):
        raise HTTPException(409, "Use redeployment to change model parameters")


def update(rid, **changes):
    with LOCK:
        record = RECORDS[rid]
        if "progress" in changes:
            changes["progress"] = max(record.get("progress", 0), changes["progress"])
        if changes.get("message") and changes["message"] != record.get("message"):
            record.setdefault("logs", []).append({"timestamp": now(), "message": changes["message"]})
            record["logs"] = record["logs"][-300:]
        record.update(changes)
        persist()


def check(event):
    if event.is_set():
        raise Canceled()


def pause(event, seconds=1):
    if event.wait(seconds):
        raise Canceled()


def prepare(uid, rid, event, params, artifacts, path, start, span):
    job = host(
        uid,
        "POST",
        "/artifacts/prepare",
        {"namespace": namespace(uid, params), "storagePath": path, "artifacts": artifacts, "verifySha256": True},
    )
    try:
        while True:
            check(event)
            if job.get("status") in ("failed", "canceled"):
                raise RuntimeError(job.get("error") or "Download failed")
            percent = min(100, job.get("progress", 0))
            update(
                rid,
                phase=job.get("phase", "downloading"),
                phaseProgress=percent,
                progress=start + percent * span // 100,
                currentFile=job.get("currentFile", ""),
                currentBytes=job.get("currentBytes", 0),
                totalBytes=job.get("totalBytes", 0),
                speedBytesPerSecond=job.get("speedBytesPerSecond", 0),
                etaSeconds=job.get("etaSeconds", 0),
                message="Downloading and verifying artifacts",
            )
            if job.get("status") in ("ready", "succeeded", "completed"):
                return
            if not job.get("jobId"):
                raise RuntimeError("Host returned no download task")
            pause(event)
            job = host(uid, "GET", "/artifacts/prepare/" + quote(job["jobId"], safe=""))
    except Canceled:
        if job.get("jobId"):
            host(uid, "POST", "/artifacts/prepare/" + quote(job["jobId"], safe="") + "/cancel", {})
        raise


def upload_model(uid, rid, event, serial, params):
    base = device_path(serial)
    artifacts = files(params)
    payload = {"namespace": namespace(uid, params), "artifacts": artifacts, "verifySha256": True}
    status = host(uid, "POST", base + "/artifacts/status", payload)
    if not status.get("ready"):
        uploaded = 0
        total = model_for(params)["modelBytes"]
        for role, items in artifacts.items():
            root = (HOME / storage(params) / role).resolve()
            if not root.is_relative_to(HOME.resolve()):
                raise RuntimeError("Invalid model storage path")
            for item in items:
                check(event)
                single = {"namespace": namespace(uid, params), "artifacts": {role: [item]}, "verifySha256": True}
                if host(uid, "POST", base + "/artifacts/status", single).get("ready"):
                    uploaded += item["size"]
                    continue
                source = root / item["name"]
                if (
                    not source.resolve().is_relative_to(root)
                    or not source.is_file()
                    or source.stat().st_size != item["size"]
                ):
                    raise RuntimeError(f"Downloaded model file is missing: {item['name']}")
                with source.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                        raise RuntimeError(f"Model SHA256 mismatch: {item['name']}")
                    stream.seek(0)
                    encoder = MultipartEncoder(
                        fields={
                            "namespace": namespace(uid, params),
                            "role": role,
                            "name": item["name"],
                            "size": str(item["size"]),
                            "sha256": item["sha256"],
                            "file": (item["name"], stream, "application/octet-stream"),
                        }
                    )
                    last = [0.0]

                    def progress(monitor):
                        check(event)
                        if time.monotonic() - last[0] > 1:
                            current = uploaded + min(item["size"], monitor.bytes_read)
                            update(
                                rid,
                                phase="transferring",
                                phaseProgress=100 * current // total,
                                progress=60 + 15 * current // total,
                                currentFile=item["name"],
                                currentBytes=current,
                                totalBytes=total,
                                message="Transferring verified model",
                            )
                            last[0] = time.monotonic()

                    monitor = MultipartEncoderMonitor(encoder, progress)
                    response = requests.post(
                        HOST + base + "/artifacts/upload",
                        data=monitor,
                        headers={"X-HC-User-ID": uid, "Content-Type": monitor.content_type},
                        timeout=(10, 300),
                    )
                    if not response.ok:
                        raise RuntimeError(f"Model transfer failed: {item['name']}")
                uploaded += item["size"]
        status = host(uid, "POST", base + "/artifacts/status", payload)
    device_root = status.get("root", "")
    if (
        not status.get("ready")
        or not re.fullmatch(r"/[A-Za-z0-9_./-]+", device_root)
        or ".." in device_root.split("/")
        or device_root == "/"
    ):
        raise RuntimeError("Device model verification failed")
    return device_root


def import_image(uid, rid, event, serial, rt, params):
    base = device_path(serial)
    identities = {rt["imageId"], rt["configImageId"]}
    try:
        status = host(uid, "POST", base + "/images/status", {"image": rt["image"], "imageId": rt["imageId"]})
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        status = {}
    if not status.get("ready") and rt["configImageId"] != rt["imageId"]:
        try:
            status = host(uid, "POST", base + "/images/status", {"image": rt["image"], "imageId": rt["configImageId"]})
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
    if status.get("ready") and status.get("imageId") in identities:
        return
    path = storage(params)
    prepare(uid, rid, event, params, {"runtime-image": [{k: rt[k] for k in ("name", "size", "sha256", "url")}]}, path, 30, 20)
    expected = rt["imageId"]
    attempted = set()
    while expected not in attempted:
        attempted.add(expected)
        request = {k: rt[k] for k in ("name", "size", "sha256", "image")}
        request.update(imageId=expected, storagePath=path, **{"async": True})
        job = {}
        try:
            job = host(uid, "POST", base + "/images/load", request)
            while not job.get("loaded"):
                check(event)
                if job.get("status") in ("failed", "canceled"):
                    raise RuntimeError(job.get("error") or "Image import failed")
                if not job.get("jobId"):
                    raise RuntimeError("Host returned no image import task")
                update(
                    rid,
                    phase="loading",
                    phaseProgress=job.get("progress", 0),
                    progress=50 + min(100, job.get("progress", 0)) // 10,
                    message="Importing runtime image",
                )
                pause(event)
                job = host(uid, "GET", base + "/images/load/" + quote(job["jobId"], safe=""))
            return
        except Canceled:
            if job.get("jobId"):
                host(uid, "POST", base + "/images/load/" + quote(job["jobId"], safe="") + "/cancel", {})
            raise
        except (RuntimeError, HTTPException) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            message = json.dumps(detail, ensure_ascii=False) if isinstance(detail, dict) else str(detail)
            match = re.search(r"docker load image ID mismatch: got (sha256:[a-f0-9]{64}), want (sha256:[a-f0-9]{64})", message)
            if not match or match.group(2) != expected or match.group(1) not in identities:
                raise
            expected = match.group(1)
    raise RuntimeError("Runtime import identity mismatch")


def compose(rt, model_root, params):
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", model_root) or ".." in model_root.split("/") or model_root == "/":
        raise RuntimeError("Invalid device model root")
    from control.artifact_layout import transport_name
    from control.token_launcher import entrypoint, fingerprint as statistics_fingerprint

    model = model_for(params)
    volumes = [
        f"{model_root}/{entry['role']}/{transport_name(item['path'])}:/model/{item['path']}:ro"
        for entry in model["models"]
        for item in entry["files"]
    ]
    profile = rt["launchParameters"]
    fp = hashlib.sha256(
        json.dumps({"statistics": statistics_fingerprint(), "image": rt["imageId"], "profile": profile, "model": model}, sort_keys=True).encode()
    ).hexdigest()
    volumes.append("${LZC_AGENT_CACHE_DIR}/" + fp + ":/cache")
    cuda_path = rt.get("cudaHostPath")
    if cuda_path not in ("/usr/local/cuda/targets/sbsa-linux/lib", "/usr/local/cuda/targets/aarch64-linux/lib"):
        raise RuntimeError("Invalid CUDA runtime mount")
    volumes.append(cuda_path + ":/host-cuda/lib:ro")
    spec = {
        "image": rt["image"],
        "pull_policy": "never",
        "restart": "unless-stopped",
        "network_mode": "host",
        "runtime": "nvidia",
        "shm_size": "2g",
        "entrypoint": entrypoint(PORT),
        "environment": {
            "AIPOD_METRICS_PORT": str(PORT),
            "SOPHON_MODEL": "/model",
            "SOPHON_MODEL_ID": model["modelId"],
            "SOPHON_PROFILE": json.dumps(profile),
            "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": "/cache/huggingface",
            "TRITON_CACHE_DIR": "/cache/triton",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "volumes": volumes,
        "healthcheck": {
            "test": [
                "CMD",
                "python3",
                "-c",
                f"import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:{PORT}/health',timeout=3)); assert d.get('status')=='ready'",
            ],
            "interval": "15s",
            "timeout": "5s",
            "retries": 20,
            "start_period": "300s",
        },
        "logging": {"driver": "local", "options": {"max-size": "10m", "max-file": "3"}},
    }
    return json.dumps({"services": {"realtime": spec}})


def deploy_service(uid, rid, event, serial, rt, model_root):
    record = RECORDS[rid]
    base = device_path(serial)
    update(rid, phase="starting", progress=78, message="Starting Sophon-Realtime")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            host,
            uid,
            "POST",
            base + "/services/deploy",
            {
                "serviceId": record["serviceId"],
                "ownerPackageId": PACKAGE,
                "compose": compose(rt, model_root, record["parameters"]),
                "forceRecreate": True,
                "dirOverWrite": True,
            },
            3600,
        )
        while not pending.done():
            if event.wait(2):
                host(uid, "POST", base + "/process/stop", {"packageIds": [record["serviceId"]]})
                try:
                    pending.result()
                except Exception:
                    pass
                host(uid, "POST", base + "/process/stop", {"packageIds": [record["serviceId"]]})
                raise Canceled()
            result = host(uid, "GET", base + "/services/progress?serviceId=" + quote(record["serviceId"]))
            update(
                rid,
                phase="starting",
                phaseProgress=result.get("progress", 0),
                progress=78 + min(100, result.get("progress", 0)) * 19 // 100,
                estimated=True,
            )
        pending.result()
    result = host(uid, "POST", base + "/services/health", {"port": PORT, "path": "/health"})
    body = result.get("body", "{}")
    body = json.loads(body) if isinstance(body, str) else body
    if not result.get("healthy") or body.get("status") != "ready" or body.get("model") != model_for(record["parameters"])["modelId"]:
        raise RuntimeError("Sophon-Realtime health check failed")


def run(rid, event):
    record = copy.deepcopy(RECORDS[rid])
    uid, serial, params = record["userId"], record["serialNumber"], record["parameters"]
    created = False
    try:
        check(event)
        req = requirements(uid, serial, params)
        authorize_resources(req, record)
        caps = host(uid, "GET", "/capabilities")
        if caps.get("artifactDownloader") != "aria2" or not caps.get("capabilities", {}).get("artifactDownload"):
            raise RuntimeError("Host Aria2 capability is required")
        d = device(uid, serial)
        rt = runtime(d, params)
        update(
            rid,
            status="deploying",
            phase="checking",
            progress=3,
            deviceName=d.get("name", serial),
            deviceModel=d.get("model", ""),
            message="Checking deployment",
        )
        payload = {"namespace": namespace(uid, params), "artifacts": files(params), "verifySha256": True}
        status = host(uid, "POST", device_path(serial) + "/artifacts/status", payload)
        if not status.get("ready"):
            prepare(uid, rid, event, params, files(params), storage(params), 5, 25)
        import_image(uid, rid, event, serial, rt, params)
        model_root = upload_model(uid, rid, event, serial, params)
        compose(rt, model_root, params)
        host(
            uid,
            "POST",
            device_path(serial) + "/services/ownership",
            {"packageId": PACKAGE, "serviceId": record["serviceId"]},
            timeout=300,
        )
        release_resources(uid, serial, params, record, requirements, host, device_path, check, event, update, rid)
        check(event)
        update(rid, hasService=True)
        created = True
        deploy_service(uid, rid, event, serial, rt, model_root)
        address = str(ipaddress.ip_address(d["ip"]))
        address = "[" + address + "]" if ":" in address else address
        try:
            usage = host(uid, "GET", device_path(serial) + "/services/" + quote(record["serviceId"]) + "/resources")
        except Exception:
            usage = {}
        update(
            rid,
            status="succeeded",
            phase="ready",
            progress=100,
            phaseProgress=100,
            apiUrl=f"http://{address}:{PORT}",
            apiAccessible=True,
            finishedAt=now(),
            diskUsageBytes=usage.get("diskUsageBytes", 0),
            reclaimableBytes=usage.get("reclaimableBytes", 0),
            message="Sophon-Realtime is ready",
            error="",
        )
    except Canceled:
        if created:
            host(uid, "POST", device_path(serial) + "/process/stop", {"packageIds": [record["serviceId"]]})
        update(rid, status="canceled", phase="canceled", finishedAt=now(), message="Deployment canceled")
    except Exception as exc:
        if created:
            try:
                host(uid, "POST", device_path(serial) + "/process/stop", {"packageIds": [record["serviceId"]]})
            except Exception:
                pass
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        detail = json.dumps(detail, ensure_ascii=False) if isinstance(detail, dict) else str(detail)
        detail = re.sub(r"(?:registry\.[^\s\"']+|/lzcapp/[^\s\"']+|sophonrt-[a-f0-9]{12})", "Sophon-Realtime-4B-Flash", detail)[:3000]
        update(rid, status="failed", phase="failed", error=detail, message="部署失败：" + detail, finishedAt=now())
    finally:
        with LOCK:
            JOBS.pop(rid, None)


def launch(uid, serial, params, retry=None, approval=None):
    params = settings(params)
    with LOCK:
        if any(r["serialNumber"] == serial and r["status"] in ACTIVE for r in RECORDS.values()):
            raise HTTPException(409, "Device already has an active deployment")
        existing = next((r for r in RECORDS.values() if r["userId"] == uid and r["serialNumber"] == serial), None)
        rid = retry or (existing["id"] if existing else uuid.uuid4().hex)
        previous = copy.deepcopy(existing or {})
        record = {
            "id": rid,
            "userId": uid,
            "serialNumber": serial,
            "deviceName": serial,
            "modelName": "Sophon-Realtime-4B-Flash",
            "modelVersion": VERSION,
            "modelId": model_for(params)["modelId"],
            "mode": "realtime-voice",
            "parameters": params,
            "status": "queued",
            "phase": "queued",
            "progress": 0,
            "phaseProgress": 0,
            "startedAt": now(),
            "logs": previous.get("logs", []),
            "serviceId": service_id(uid),
            "hasService": previous.get("hasService", False),
            "message": "Queued",
            "apiKey": "",
            "apiUrl": previous.get("apiUrl", ""),
            "apiAccessible": False,
            "currentFile": "",
            "currentBytes": 0,
            "totalBytes": 0,
            "speedBytesPerSecond": 0,
            "etaSeconds": 0,
            "memoryUsageBytes": 0,
            "diskUsageBytes": 0,
            "reclaimableBytes": 0,
            "error": "",
        }
        record.update(approval or {"stopProtected": False, "protectedProcessIds": []})
        RECORDS[rid] = record
        event = threading.Event()
        JOBS[rid] = event
        persist()
        POOL.submit(run, rid, event)
        return public(record)


app = FastAPI(title="Sophon-Realtime installer", docs_url=None, redoc_url=None)


@app.on_event("startup")
def startup():
    if not MODEL or not any(k in runtime_manifest() for k in ("thor", "orin")):
        raise RuntimeError("A complete, verified release manifest is required")
    load()


@app.middleware("http")
async def cors(request, call_next):
    LANGUAGE.set("zh" if request.headers.get("Accept-Language", "zh").lower().startswith("zh") else "en")
    origin = request.headers.get("origin", "")
    hostname = request.headers.get("host", "").split(":")[0]
    expected = "https://" + MAIN_SUBDOMAIN + "." + hostname.split(".", 1)[1] if "." in hostname else ""
    allowed = os.environ.get("AIPOD_UI_ORIGIN", "")
    if origin and origin not in (expected, allowed, "https://" + hostname):
        return JSONResponse({"error": "Origin not allowed"}, status_code=403)
    response = Response(status_code=204) if request.method == "OPTIONS" else await call_next(request)
    if origin:
        response.headers.update(
            {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Vary": "Origin",
                "Access-Control-Allow-Headers": "Content-Type, X-HC-User-ID, Accept-Language",
                "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
            }
        )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(HTTPException)
async def api_error(request, exc):
    return JSONResponse(
        translated(sanitize(exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)})),
        status_code=exc.status_code,
    )


@app.get("/", response_class=HTMLResponse)
def landing():
    return (ROOT / "control/index.html").read_text().replace("__AIPOD_SUBDOMAIN__", MAIN_SUBDOMAIN)


@app.get("/api/v1/health")
def health():
    return {"ok": True, "package": PACKAGE, "version": VERSION, "protocolVersion": "1.1"}


@app.get("/api/v1/descriptor")
def descriptor():
    return translated(
        {
            "protocolVersion": "1.1",
            "minHostProtocolVersion": "1.0",
            "package": PACKAGE,
            "name": "Sophon-Realtime-4B-Flash",
            "model": {
                "name": "Sophon-Realtime-4B-Flash",
                "taskType": "realtime-voice",
                "networkDrivePath": "/AI 模型/Sophon-Realtime-4B-Flash/",
                "reuseAcrossDevices": True,
                "maxConcurrency": 8,
            },
            "supportedDevices": ["AGX Thor", "AGX Orin"],
            "modes": [{"id": "realtime-voice", "title": "Full-duplex voice"}],
            "capabilities": {
                "prebuiltImages": True,
                "networkDriveReuse": True,
                "unifiedProgress": True,
                "cancelDeployment": True,
                "restartStopped": True,
                "modelResources": True,
                "piAgentConfig": False,
                "runtimeModelDownload": True,
                "modelSourceSelection": True,
                "descriptorParameters": True,
                "ociRuntimePull": False,
                "managedUninstall": True,
                "delete": True,
                "persistentData": True,
            },
            "parameters": [
                {
                    "id": "modelSize",
                    "label": "模型规格",
                    "type": "select",
                    "default": "4B-Flash",
                    "fixed": True,
                    "options": [{"value": "4B-Flash", "label": "4B-Flash"}],
                },
                {
                    "id": "modelProvider",
                    "label": "模型与运行镜像下载源",
                    "type": "select",
                    "default": "modelscope",
                    "fixed": False,
                    "options": [
                        {"value": "modelscope", "label": "ModelScope"},
                        {"value": "huggingface", "label": "Hugging Face"},
                    ],
                },
            ],
            "modelSources": {
                "policy": "user-selected",
                "detection": "installation-ui",
                "regionProviders": {"CN": "modelscope", "default": "huggingface"},
                "integrity": "size-and-sha256",
                "cache": "device-persistent",
                "totalSize": model_for(settings({}))["modelBytes"],
                "models": [
                    {
                        "role": entry["role"],
                        "name": entry["name"],
                        "destination": storage(settings({})) + "/" + entry["role"],
                        "fileCount": len(entry["files"]),
                        "size": sum(f["size"] for f in entry["files"]),
                        "providers": entry["sources"],
                    }
                    for entry in model_for(settings({}))["models"]
                ],
            },
            "artifacts": files(settings({})),
            "runtimeImages": runtime_manifest(),
        }
    )


@app.get("/api/v1/devices")
def list_devices(request: Request):
    result = []
    for item in devices(user(request)):
        try:
            target(item)
            result.append(item)
        except HTTPException:
            pass
    return result


@app.get("/api/v1/deployments/requirements")
def preflight(request: Request, serialNumber: str, modelProvider: str = "modelscope", modelSize: str = "4B-Flash"):
    return translated(
        sanitize(
            requirements(
                user(request), safe_id(serialNumber), settings({"modelProvider": modelProvider, "modelSize": modelSize})
            )
        )
    )


@app.get("/api/v1/deployments")
def listing(request: Request):
    uid = user(request)
    with LOCK:
        return [public(r) for r in RECORDS.values() if r["userId"] == uid]


@app.post("/api/v1/deployments")
def create(request: Request, body: dict):
    validate_request(body)
    approval = resource_authorization(body)
    uid = user(request)
    params = settings(body.get("parameters"))
    serials = body.get("serialNumbers") or [body.get("serialNumber")]
    if not isinstance(serials, list) or not 1 <= len(serials) <= 4:
        raise HTTPException(400, "Select 1–4 devices")
    serials = list(dict.fromkeys(safe_id(s) for s in serials))
    for serial in serials:
        req = requirements(uid, serial, params)
        authorize_resources(req, approval)
    return {"records": [launch(uid, serial, params, approval=approval) for serial in serials]}


@app.get("/api/v1/deployments/{rid}")
def get_record(request: Request, rid: str):
    uid = user(request)
    with LOCK:
        return public(own(uid, rid))


@app.patch("/api/v1/deployments/{rid}")
def rename(request: Request, rid: str, body: dict):
    uid = user(request)
    name = body.get("displayName", "")
    if not isinstance(name, str) or not 1 <= len(name) <= 80:
        raise HTTPException(400, "Invalid display name")
    with LOCK:
        own(uid, rid)
        update(rid, displayName=name)
        return public(RECORDS[rid])


@app.post("/api/v1/deployments/{rid}/cancel")
def cancel(request: Request, rid: str):
    uid = user(request)
    with LOCK:
        own(uid, rid)
        event = JOBS.get(rid)
        if event:
            event.set()
    return {"ok": True}


@app.post("/api/v1/deployments/{rid}/retry")
@mutates
def retry(request: Request, rid: str, body: dict = None):
    uid = user(request)
    with LOCK:
        record = copy.deepcopy(own(uid, rid))
        if record["status"] != "failed":
            raise HTTPException(409, "Only failed deployments can be retried")
    validate_action(body, record)
    return launch(uid, record["serialNumber"], record["parameters"], retry=rid, approval=resource_authorization(body or record))


@app.post("/api/v1/deployments/{rid}/stop")
@mutates
def stop(request: Request, rid: str):
    uid = user(request)
    with LOCK:
        record = copy.deepcopy(own(uid, rid))
    if record["status"] in ACTIVE:
        raise HTTPException(409, "Cancel the active deployment first")
    host(uid, "POST", device_path(record["serialNumber"]) + "/process/stop", {"packageIds": [record["serviceId"]]})
    update(rid, status="stopped", phase="stopped", apiAccessible=False, message="Sophon-Realtime stopped")
    return public(RECORDS[rid])


@app.post("/api/v1/deployments/{rid}/start")
@mutates
def start(request: Request, rid: str, body: dict = None):
    uid = user(request)
    with LOCK:
        record = copy.deepcopy(own(uid, rid))
    if record["status"] != "stopped":
        raise HTTPException(409, "Deployment is not stopped")
    validate_action(body, record)
    req = requirements(uid, record["serialNumber"], record["parameters"])
    approve = resource_authorization(body or record)
    authorize_resources(req, approve)
    return launch(uid, record["serialNumber"], record["parameters"], retry=rid, approval=approve)


@app.delete("/api/v1/deployments/{rid}")
@mutates
def delete(request: Request, rid: str):
    uid = user(request)
    raw_params = request.query_params.getlist("deleteData")
    if len(raw_params) > 1:
        raise HTTPException(400, "Duplicate deleteData parameter")
    delete_data = False
    if raw_params:
        if raw_params[0] not in ("true", "false"):
            raise HTTPException(400, "deleteData must be true or false")
        delete_data = raw_params[0] == "true"

    with LOCK:
        record = copy.deepcopy(own(uid, rid))
        event = JOBS.get(rid)
        if event:
            event.set()
            raise HTTPException(409, "Deployment cancellation in progress; retry deletion")
    if record.get("hasService"):
        if delete_data:
            host(uid, "DELETE", device_path(record["serialNumber"]) + "/services/" + quote(record["serviceId"]))
        else:
            host(uid, "POST", device_path(record["serialNumber"]) + "/process/stop", {"packageIds": [record["serviceId"]]})
    with LOCK:
        RECORDS.pop(rid, None)
        persist()
    return {"deleted": True}


@app.get("/api/v1/deployments/{rid}/pi-agent-config")
def pi_agent_config(request: Request, rid: str):
    own(user(request), rid)
    raise HTTPException(404, "Sophon-Realtime does not provide a Pi Agent chat configuration")
