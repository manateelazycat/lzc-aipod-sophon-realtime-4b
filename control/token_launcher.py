"""Ship a small inference adapter with the LPK, preserving the runtime image."""
import base64
import hashlib
import json
import zlib
from pathlib import Path


def sources():
    root = Path(__file__).resolve().parents[1]
    runtime = root / "runtime"
    if (runtime / "server.py").is_file():
        out = {
            "server": (runtime / "server.py").read_text(),
            "metrics": (runtime / "token_metrics.py").read_text(),
        }
        thor_patch = runtime / "thor_patch.py"
        if thor_patch.is_file():
            out["thor_patch"] = thor_patch.read_text()
        thor_yaml = runtime / "config" / "sophon_realtime_flash_4b.yaml"
        if thor_yaml.is_file():
            out["thor_yaml"] = thor_yaml.read_text()
        orin_yaml = runtime / "config" / "sophon_realtime_flash_4b_orin.yaml"
        if orin_yaml.is_file():
            out["orin_yaml"] = orin_yaml.read_text()
        return out
    return {"server": (root / "control/inference_server.py").read_text(),
            "metrics": (root / "control/inference_token_metrics.py").read_text()}


def fingerprint():
    return hashlib.sha256(json.dumps(sources(), sort_keys=True).encode()).hexdigest()


def entrypoint(port, python="python3", module="aipod_inference"):
    encoded = base64.b64encode(
        zlib.compress(json.dumps(sources()).encode("utf-8"), 9)
    ).decode("ascii")
    code = (
        "import base64,json,sys,types,zlib;"
        f"s=json.loads(zlib.decompress(base64.b64decode({encoded!r})).decode('utf-8'));"
        "m=types.ModuleType('token_metrics');sys.modules[m.__name__]=m;"
        "exec(compile(s['metrics'],'aipod-token-metrics.py','exec'),m.__dict__);"
        f"a=types.ModuleType({module!r});a.__file__='aipod-inference.py';sys.modules[a.__name__]=a;"
        "a._THOR_PATCH_SRC=s.get('thor_patch','');a._THOR_DEPLOY_YAML=s.get('thor_yaml','');a._ORIN_DEPLOY_YAML=s.get('orin_yaml','');"
        "exec(compile(s['server'],a.__file__,'exec'),a.__dict__);"
        f"import uvicorn;uvicorn.run(a.app,host='0.0.0.0',port={int(port)},workers=1)"
    )
    return [python, "-c", code]
