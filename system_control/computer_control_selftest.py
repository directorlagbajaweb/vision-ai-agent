"""
system_control/computer_control_selftest.py
Standalone verification for the Computer Use feature: dispatcher wiring,
coordinate math, and the app-whitelist confirmation logic. Run with:

    python3 -m system_control.computer_control_selftest

Deliberately does NOT perform real clicks/keystrokes — that would move the
user's actual cursor or type into whatever's currently focused with no
warning. Anything that needs a real click prints a manual checklist instead.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import Quartz
from system_control import dispatcher, computer_control


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main():
    ok = True

    print("\n-- Session / dispatcher wiring --")
    r = dispatcher.dispatch("computer_control", action="take_screenshot")
    ok &= check("computer_control before any session is rejected", r["success"] is False)

    r = dispatcher.dispatch("start_computer_use", goal="selftest")
    frontmost = computer_control._frontmost_app_name()
    if r.get("status") == "confirmation_required":
        print(f"    (frontmost app '{frontmost}' is not on BENIGN_APPS — confirmation required, as expected)")
        confirm_result = dispatcher.confirm(r["confirmation_token"], approved=True)
        ok &= check("confirming start_computer_use succeeds", confirm_result.get("success") is True)
    else:
        print(f"    (frontmost app '{frontmost}' IS on BENIGN_APPS — task started with zero confirmation)")
        ok &= check("frictionless start_computer_use succeeds", r.get("success") is True)

    computer_control.record_capture_size(1568, 980)
    r = dispatcher.dispatch("computer_control", action="take_screenshot")
    ok &= check("take_screenshot after session start succeeds", r.get("success") is True)

    print("\n-- Step limit --")
    for i in range(14):
        dispatcher.dispatch("computer_control", action="take_screenshot")
    r = dispatcher.dispatch("computer_control", action="take_screenshot")
    ok &= check("16th action in the same session hits the step cap", r["success"] is False and "step limit" in r["error"])

    r = dispatcher.dispatch("end_computer_use")
    ok &= check("end_computer_use releases the session", r.get("ended") is True)
    r = dispatcher.dispatch("computer_control", action="take_screenshot")
    ok &= check("computer_control after end_computer_use is rejected again", r["success"] is False)

    print("\n-- Coordinate math --")
    dispatcher.dispatch("start_computer_use", goal="coord check")
    computer_control.record_capture_size(1568, 980)
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    lx, ly = computer_control._to_logical_coords(784, 490)  # center of the fake image
    expected_x, expected_y = bounds.size.width / 2, bounds.size.height / 2
    ok &= check(
        f"image center maps to real screen center (got {lx:.0f},{ly:.0f}, expected ~{expected_x:.0f},{expected_y:.0f})",
        abs(lx - expected_x) < 1 and abs(ly - expected_y) < 1,
    )
    clamped_x, clamped_y = computer_control._to_logical_coords(999999, 999999)
    ok &= check(
        "out-of-range coordinate clamps within screen bounds",
        clamped_x <= bounds.origin.x + bounds.size.width and clamped_y <= bounds.origin.y + bounds.size.height,
    )

    print("\n-- App whitelist --")
    print(f"    Current frontmost app: {computer_control._frontmost_app_name()}")
    print(f"    BENIGN_APPS: {sorted(computer_control.BENIGN_APPS)}")
    print(f"    needs_confirmation_for_start() -> {computer_control.needs_confirmation_for_start()}")
    ok &= check(
        "close_app/run_shortcut/execute_python are still always gated (unaffected by callable support)",
        dispatcher.ACTION_REGISTRY["close_app"]["requires_confirmation"] is True
        and dispatcher.ACTION_REGISTRY["run_shortcut"]["requires_confirmation"] is True
        and dispatcher.ACTION_REGISTRY["execute_python"]["requires_confirmation"] is True,
    )

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    print("""
Manual checklist (needs a real click/keystroke on your actual screen —
not run automatically by this script):
  [ ] left_click on a real button actually activates it
  [ ] right_click opens a context menu where one exists
  [ ] scroll('down', 3) actually scrolls a long page/list
  [ ] type_text into a real focused text field produces the right text
  [ ] press_key('cmd+a') / press_key('enter') behave as expected
  [ ] With a BENIGN_APPS app frontmost, start_computer_use runs with no
      confirmation prompt at all — confirm this feels acceptable in practice
""")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
