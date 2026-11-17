"""Ship a small inference adapter with the LPK, preserving the runtime image."""
import base64
import hashlib
import json
from pathlib import Path


def sources():
    root = Path(__file__).resolve().parents[1]
    runtime = root / "runtime"
    if (runtime / "server.py").is_file():
        return {"server": (runtime / "server.py").read_text(),
                "metrics": (runtime / "token_metrics.py").read_text()}
    return {"server": (root / "control/inference_server.py").read_text(),
            "metrics": (root / "control/inference_token_metrics.py").read_text()}


def fingerprint():
    return hashlib.sha256(json.dumps(sources(), sort_keys=True).encode()).hexdigest()


def entrypoint(port, python="python3", module="aipod_inference"):
    encoded = base64.b64encode(json.dumps(sources()).encode()).decode("ascii")
    code = (
        "import base64,json,sys,types;"
        f"s=json.loads(base64.b64decode({encoded!r}));"
        "m=types.ModuleType('token_metrics');sys.modules[m.__name__]=m;"
        "exec(compile(s['metrics'],'aipod-token-metrics.py','exec'),m.__dict__);"
        f"a=types.ModuleType({module!r});a.__file__='aipod-inference.py';sys.modules[a.__name__]=a;"
        "exec(compile(s['server'],a.__file__,'exec'),a.__dict__);"
        f"import uvicorn;uvicorn.run(a.app,host='0.0.0.0',port={int(port)},workers=1)"
    )
    return [python, "-c", code]
