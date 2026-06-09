"""
Quick integration test — exercises all Person B endpoints.
Run with: python test_endpoints.py
Requires the server running at localhost:8000.
"""

import requests
import json
import sys

BASE = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    data = r.json()
    print(f"✓ Health: {data}")
    return data


def test_text_capture():
    r = requests.post(f"{BASE}/api/capture/text", json={
        "content": "Observed corrosion on pipe section B-14, approximately 30% wall thickness loss. "
                   "Flange bolts on valve V-203 appear loose. No visible leaks at time of inspection.",
        "section_hint": "findings"
    })
    assert r.status_code == 200
    obs = r.json()
    print(f"✓ Text capture: type={obs['input_type']}, content={obs['raw_content'][:80]}...")
    return obs


def test_text_capture_2():
    r = requests.post(f"{BASE}/api/capture/text", json={
        "content": "Electrical panel EP-4 has exposed wiring. Arc flash labels missing. "
                   "Lockout/tagout procedure posted but appears outdated (rev 2019).",
        "section_hint": "findings"
    })
    assert r.status_code == 200
    obs = r.json()
    print(f"✓ Text capture 2: type={obs['input_type']}")
    return obs


def test_session_flow():
    # 1. Create session
    r = requests.post(f"{BASE}/api/sessions")
    assert r.status_code == 200
    session = r.json()
    sid = session["session_id"]
    print(f"✓ Session created: {sid}")

    # 2. Add observations
    obs1 = test_text_capture()
    r = requests.post(f"{BASE}/api/sessions/{sid}/observations", json=obs1)
    assert r.status_code == 200
    print(f"✓ Added observation 1")

    obs2 = test_text_capture_2()
    r = requests.post(f"{BASE}/api/sessions/{sid}/observations", json=obs2)
    assert r.status_code == 200
    print(f"✓ Added observation 2")

    # 3. Check session
    r = requests.get(f"{BASE}/api/sessions/{sid}")
    assert r.status_code == 200
    session = r.json()
    print(f"✓ Session has {len(session['observations'])} observations")

    # 4. Generate report
    print("  Generating report (this calls the LLM, may take 10-30s)...")
    r = requests.post(f"{BASE}/api/sessions/{sid}/generate", json={
        "observation_session_id": sid,
        "inspector_name": "Yashvi",
        "site_name": "Industrial Plant Alpha",
        "site_location": "Dubai, UAE",
        "report_type": "field_inspection",
    })

    if r.status_code == 200:
        report = r.json()
        print(f"✓ Report generated: {report['title']}")
        print(f"  Sections: {len(report['sections'])}")
        for s in report["sections"]:
            print(f"    - {s['title']} ({len(s['cited_claims'])} citations)")
        print(f"  RAG chunks used: {len(report['rag_chunks_used'])}")
        print(f"  Metadata: {report.get('generation_metadata', {})}")

        # 5. Retrieve report
        rid = report["report_id"]
        r = requests.get(f"{BASE}/api/reports/{rid}")
        assert r.status_code == 200
        print(f"✓ Report retrievable at /api/reports/{rid}")
    else:
        print(f"✗ Report generation failed: {r.status_code} — {r.text}")
        print("  (This is expected if no LLM API key is configured)")


def test_quick_generate():
    """Test the one-shot generate endpoint."""
    print("\nTesting quick generate...")
    r = requests.post(
        f"{BASE}/api/generate-quick",
        params={
            "inspector_name": "Test Inspector",
            "site_name": "Test Site",
            "report_type": "safety_audit",
        },
        json=[
            {
                "input_type": "text",
                "raw_content": "Emergency exit sign on floor 3 is not illuminated. "
                               "Fire extinguisher in corridor B expired 6 months ago.",
                "section_hint": "safety",
            }
        ],
    )
    if r.status_code == 200:
        report = r.json()
        print(f"✓ Quick report: {report['title']} ({len(report['sections'])} sections)")
    else:
        print(f"✗ Quick generate: {r.status_code} — {r.text[:200]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Person B — Endpoint Integration Tests")
    print("=" * 60)

    try:
        test_health()
    except Exception as e:
        print(f"✗ Server not reachable at {BASE}: {e}")
        print("  Start the server first: uvicorn app.main:app --reload")
        sys.exit(1)

    test_session_flow()
    test_quick_generate()

    print("\n" + "=" * 60)
    print("Done. Check /docs for the interactive API explorer.")
    print("=" * 60)
