"""Read-only authorization and late, cancellable, selective resource release."""
import re
import time
from fastapi import HTTPException

_INFRA = re.compile(r'traefik|caddy|postgres|lzc[-_]?ai[-_]?agent|serviceforward|network|cluster|oom|thermal|monitor|jtop', re.I)


def authorization(body):
    body = body or {}
    for name in ('stopExisting', 'stopProtected'):
        if name in body and type(body[name]) is not bool:
            raise HTTPException(400, 'Invalid stop authorization')
    ids = body.get('protectedProcessIds', [])
    if (not isinstance(ids, list) or len(ids) > 100 or any(not isinstance(x, str) or
            not 1 <= len(x) <= 200 or any(ord(c) < 32 for c in x) for x in ids)):
        raise HTTPException(400, 'Invalid protected process identifiers')
    return {'stopExisting': body.get('stopExisting', False), 'stopProtected': body.get('stopProtected', False),
            'protectedProcessIds': sorted(set(ids))}


def running_ids(status):
    return set(status.get('services', {})) | {x['pkgId'] for x in status.get('ollamaPs', []) if x.get('pkgId')}


def assess(req, uid, serial, host, device_path, own_service):
    req = dict(req)
    resources = req['resources']
    status = req.pop('_processStatus', None)
    if status is None:
        status = host(uid, 'GET', device_path(serial) + '/process/status')
    services = status.get('services', {})
    physical = max(0, int(resources.get('memoryAvailableBytes', 0)))
    required = req['requiredMemoryBytes']
    total = int(resources.get('memoryTotalBytes', 0))
    own_memory = max(0, int(services.get(own_service, {}).get('memoryUsage', 0)))
    prefix = own_service.rsplit('-', 1)[0] + '-'
    endpoints = {name for name in running_ids(status) if name != own_service and name.startswith(prefix)}
    candidates = {name for name in running_ids(status) if name != own_service and not _INFRA.search(name)}
    insufficient = physical + own_memory < required
    selected = set(endpoints)
    if insufficient:
        deficit = required - physical - own_memory
        for name in sorted(candidates, key=lambda x: int(services.get(x, {}).get('memoryUsage', 0)), reverse=True):
            selected.add(name)
            deficit -= max(0, int(services.get(name, {}).get('memoryUsage', 0)))
            if deficit <= 0:
                break
    disk_bad = bool(resources.get('diskAvailableKnown') and req['availableDiskBytes'] < req['requiredDiskBytes'])
    impossible = bool(total and total < required)
    reasons = []
    if impossible:
        reasons.append('Total device memory is below this model profile requirement')
    elif insufficient:
        reasons.append('Confirmed model services must release memory before startup')
    if endpoints:
        reasons.append('A model service occupies the speech endpoint')
    if disk_bad:
        reasons.append('Insufficient preparation disk space')
    req.update(blocked=insufficient or bool(endpoints) or disk_bad or impossible,
               requiresConfirmation=bool(selected) and not disk_bad and not impossible,
               requiresStop=bool(selected), protectedProcessIds=sorted(selected), blockReasons=reasons,
               insufficientMemory=insufficient, physicalCapacityInsufficient=impossible,
               preparationBlocked=disk_bad, physicalAvailableMemoryBytes=physical,
               ownServiceMemoryReuseBytes=own_memory, resourceReleaseStage='starting',
               endpointProcessIds=sorted(endpoints), ownService=own_service,
               ownServiceRunning=own_service in running_ids(status))
    return req


def authorize(req, approval):
    if not req['blocked']:
        return req
    ids = set(req['protectedProcessIds'])
    if (not req['requiresConfirmation'] or not ids or not approval.get('stopProtected') or
            not ids.issubset(approval.get('protectedProcessIds', []))):
        raise HTTPException(409, req)
    return req


def release(uid, serial, params, approval, requirements, host, device_path, check, event, update, rid):
    check(event)
    req = requirements(uid, serial, params, prepared=True)
    authorize(req, approval)
    own_service = req['ownService']
    must_replace = req['ownServiceRunning']
    pending = list(req['endpointProcessIds'])
    if must_replace:
        pending.insert(0, own_service)
    pending += [x for x in req['protectedProcessIds'] if x not in pending]
    if not pending and req['physicalAvailableMemoryBytes'] >= req['requiredMemoryBytes']:
        return req
    caps = host(uid, 'GET', '/capabilities')
    if not caps.get('capabilities', {}).get('selectiveProcessStop'):
        raise RuntimeError('Selective process stopping is required')
    stopped = set()
    evidence = {'stage': 'starting', 'authorizedIds': list(approval.get('protectedProcessIds', [])), 'samples': [], 'stopCalls': []}
    deadline = time.monotonic() + 120
    update(rid, phase='stopping', message='Preparation complete; waiting for confirmed model resources to be released')
    while time.monotonic() < deadline:
        check(event)
        status = host(uid, 'GET', device_path(serial) + '/process/status')
        resources = host(uid, 'GET', device_path(serial) + '/resources')
        running = running_ids(status)
        physical = int(resources.get('memoryAvailableBytes', 0))
        endpoint_live = set(req['endpointProcessIds']) & running
        own_live = {own_service} & running if must_replace else set()
        evidence['samples'].append({'unix': time.time(), 'availableMemoryBytes': physical, 'requiredMemoryBytes': req['requiredMemoryBytes'], 'confirmedRunningIds': sorted(running & set(pending))})
        update(rid, resourceReleaseEvidence=evidence)
        if not endpoint_live and not own_live and not (stopped & running) and physical >= req['requiredMemoryBytes']:
            return req
        if not (stopped & running):
            next_id = next((x for x in pending if x in running and x not in stopped), None)
            if next_id:
                if next_id != own_service and (not approval.get('stopProtected') or next_id not in approval.get('protectedProcessIds', [])):
                    raise HTTPException(409, {'error': 'A new resource conflict requires confirmation',
                                               'requiresConfirmation': True, 'protectedProcessIds': [next_id]})
                check(event)
                host(uid, 'POST', device_path(serial) + '/process/stop', {'packageIds': [next_id]})
                stopped.add(next_id)
                evidence['stopCalls'].append({'unix': time.time(), 'packageIds': [next_id]})
                update(rid, resourceReleaseEvidence=evidence)
        if event.wait(.5):
            check(event)
    raise RuntimeError('Resource release timed out: stopped processes or actual available memory did not reach the profile budget')
