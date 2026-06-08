import json, pathlib
from agent_customer_support.models import Flow

def test_seed_flow_valid():
    p = pathlib.Path("seeds/flows/pyc_su_co.json")
    flow = Flow.model_validate(json.loads(p.read_text()))
    assert flow.id and flow.steps and "escalate" in flow.outcomes
