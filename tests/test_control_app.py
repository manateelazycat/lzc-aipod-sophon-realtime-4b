import json
import pathlib
from fastapi.testclient import TestClient

import control.app as app_mod


def test_health_and_descriptor(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DATA", tmp_path)
    monkeypatch.setattr(app_mod, "RECORDS", {})
    # Provide a minimal runtimes.json for descriptor test
    rt_path = tmp_path / "runtimes.json"
    rt_path.write_text(json.dumps({
        "thor": {
            "image": "local/sophon-realtime-4b-thor:0.1.0",
            "imageId": "sha256:" + "a" * 64,
            "configImageId": "sha256:" + "a" * 64,
            "size": 1000,
            "imageSizeBytes": 2000,
            "sha256": "b" * 64,
            "name": "sophon-realtime-4b-thor-0.1.0.tar.gz",
            "sources": {
                "modelscope": {"revision": "c" * 40, "url": "https://modelscope.cn/models/test/test/resolve/" + "c" * 40 + "/runtime.tar.gz"},
                "huggingface": {"revision": "d" * 40, "url": "https://huggingface.co/test/test/resolve/" + "d" * 40 + "/runtime.tar.gz"},
            },
            "profiles": {"4B-Flash": {"flow_steps": 4}},
            "requiredMemoryBytesByModel": {"4B-Flash": 64 << 30},
            "compatibility": {"architecture": "arm64", "l4tVersion": "39.2.0", "jetpackVersion": "7.2"},
            "cudaHostPath": "/usr/local/cuda/targets/sbsa-linux/lib",
        }
    }))
    monkeypatch.setattr(app_mod, "runtime_manifest", lambda: json.loads(rt_path.read_text()))

    client = TestClient(app_mod.app)
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    d = client.get("/api/v1/descriptor")
    assert d.status_code == 200
    body = d.json()
    assert body["package"] == "cloud.lazycat.aipod.sophon-realtime-4b"
    assert body["capabilities"]["persistentData"] is True
    assert body["capabilities"]["delete"] is True
    assert body["modelSources"]["models"][0]["fileCount"] == 36


def test_delete_data_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DATA", tmp_path)
    monkeypatch.setattr(app_mod, "RECORDS", {
        "r1": {
            "id": "r1",
            "userId": "u1",
            "serialNumber": "dev1",
            "serviceId": "sophonrt4b-abc",
            "hasService": True,
            "status": "succeeded",
            "parameters": {"modelProvider": "modelscope", "modelSize": "4B-Flash"},
        },
        "r2": {
            "id": "r2",
            "userId": "u1",
            "serialNumber": "dev1",
            "serviceId": "sophonrt4b-abc",
            "hasService": True,
            "status": "succeeded",
            "parameters": {"modelProvider": "modelscope", "modelSize": "4B-Flash"},
        },
    })
    calls = []
    monkeypatch.setattr(app_mod, "host", lambda uid, method, path, body=None, timeout=90: calls.append((method, path, body)) or {})
    client = TestClient(app_mod.app)
    headers = {"X-HC-User-ID": "u1"}

    # Invalid deleteData rejected with 400
    assert client.delete("/api/v1/deployments/r1?deleteData=yes", headers=headers).status_code == 400
    assert client.delete("/api/v1/deployments/r1?deleteData=true&deleteData=false", headers=headers).status_code == 400

    # Default (no param) -> stop only, preserve data
    res = client.delete("/api/v1/deployments/r1", headers=headers)
    assert res.status_code == 200
    assert res.json() == {"deleted": True}
    assert calls[-1][0] == "POST" and calls[-1][1].endswith("/process/stop")

    # Explicit deleteData=true -> destructive DELETE service
    res2 = client.delete("/api/v1/deployments/r2?deleteData=true", headers=headers)
    assert res2.status_code == 200
    assert calls[-1][0] == "DELETE" and "/services/" in calls[-1][1]
