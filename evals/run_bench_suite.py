import os
import subprocess
import sys
import time
import urllib.request

CONFIGS = [
    {"label": "groq-off", "provider": "groq", "memory": "false"},
    {"label": "groq-on", "provider": "groq", "memory": "true"},
    {"label": "ollama-off", "provider": "ollama", "memory": "false"},
    {"label": "ollama-on", "provider": "ollama", "memory": "true"},
]

def wait_for_server(timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health") as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def run():
    os.makedirs("evals/logs", exist_ok=True)
    os.makedirs("evals/results", exist_ok=True)

    for cfg in CONFIGS:
        label = cfg["label"]
        provider = cfg["provider"]
        memory = cfg["memory"]

        log_file = f"evals/logs/{label}-recall2.txt"
        out_file = f"evals/results/{label}-recall2.jsonl"

        print(f"\n==================================================")
        print(f"Starting Benchmark: {label} (Provider={provider}, Memory={memory})")
        print(f"==================================================")

        if os.path.exists(out_file):
            os.remove(out_file)

        env = dict(os.environ)
        env["LLM_PROVIDER"] = provider
        env["MEMORY_ENABLED"] = memory

        log_fp = open(log_file, "w", encoding="utf-8")
        server_proc = subprocess.Popen(
            ["uv", "run", "uvicorn", "app.main:app", "--port", "8000"],
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT
        )

        if not wait_for_server():
            print(f"ERROR: Server failed to start for {label}", file=sys.stderr)
            server_proc.kill()
            log_fp.close()
            continue

        print(f"Server started successfully. Running bench_memory.py...")

        bench_cmd = [
            "uv", "run", "python", "evals/bench_memory.py",
            "--label", label,
            "--only", "recall_project",
            "--reps", "3",
            "--pause", "5",
            "--log", log_file,
            "--out", out_file
        ]

        bench_proc = subprocess.run(bench_cmd)

        print(f"Finished benchmark for {label} (exit code: {bench_proc.returncode})")

        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        log_fp.close()

        time.sleep(2)

if __name__ == "__main__":
    run()
