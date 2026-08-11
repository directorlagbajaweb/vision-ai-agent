"""
system_control/code_exec.py
Sandboxed-ish Python code execution for VISION. Runs code in a
subprocess with a timeout so a bad script can't hang or crash VISION.
"""

import subprocess
import tempfile
from pathlib import Path


def execute_python(code: str, timeout: int = 10) -> dict:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            temp_path = f.name

        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Code timed out after {timeout} seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)