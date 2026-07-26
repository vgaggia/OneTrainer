#!/usr/bin/env bash
# Echo the current pod's SSH args. The public SSH port changes on every pod (and on every
# restart), so hardcoding it is how the last session ended up with "Connection refused" that
# looked like a dead pod.
#
#   POD=$(bash perf_tests/pod_ssh.sh) && ssh $POD "nvidia-smi"
[ -n "${RUNPOD_KEY:-}" ] || { echo "RUNPOD_KEY not set" >&2; exit 1; }
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query":"query{myself{pods{id desiredStatus runtime{ports{ip publicPort privatePort type}}}}}"}' \
  | python -c "
import json,sys
pods=json.load(sys.stdin)['data']['myself']['pods']
for p in pods:
    rt=p.get('runtime') or {}
    for prt in (rt.get('ports') or []):
        if prt['privatePort']==22 and prt['type']=='tcp':
            print(f\"-o StrictHostKeyChecking=no -o BatchMode=yes -p {prt['publicPort']} root@{prt['ip']}\")
            sys.exit(0)
sys.exit('no running pod with an ssh port yet')
"
