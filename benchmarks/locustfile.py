"""Optional Locust profile for the fake-provider gateway.

The numbers in the README come from ``python -m llm_guard.cli bench``, which
does not require Locust. This file is the same workload for an external run:

    locust -f benchmarks/locustfile.py --headless -u 20 -r 20 -t 30s --host http://127.0.0.1:8080
"""

from locust import HttpUser, between, task

_PAYLOAD = {
    "model": "latency",
    "temperature": 0,
    "messages": [
        {
            "role": "system",
            "content": "The platinum card APR is 2.9 percent for the first twelve months.",
        },
        {"role": "user", "content": "What is the platinum card APR for the first twelve months?"},
    ],
    "llm_guard": {"cache": False},
}


class GatewayUser(HttpUser):
    wait_time = between(0.0, 0.01)

    @task
    def chat(self) -> None:
        self.client.post(
            "/v1/chat/completions",
            json=_PAYLOAD,
            headers={"Authorization": "Bearer lg_test_regulated_demo_key_0001"},
            name="/v1/chat/completions",
        )
