#!/usr/bin/env python3
r"""
Simple script that waits for a keypress, checking once per second.

This script is useful for monitoring background system activity with
resource_monitor.py. Run it as:

    python resource_monitor.py -d \output -- python waiter.py

Then press any key when you want to stop monitoring.
"""
import sys
import time

# Windows-specific keyboard input handling
if sys.platform == 'win32':
    import msvcrt

    def key_pressed() -> bool:
        """Return True if a key has been pressed."""
        return msvcrt.kbhit()

    def get_key() -> str:
        """Read and return the pressed key."""
        return msvcrt.getch().decode('utf-8', errors='ignore')
else:
    # Unix/Linux fallback (non-blocking input)
    import select

    def key_pressed() -> bool:
        """Return True if input is available on stdin."""
        return select.select([sys.stdin], [], [], 0)[0] != []

    def get_key() -> str:
        """Read and return a single character from stdin."""
        return sys.stdin.read(1)


def main() -> int:
    """Wait for a keypress, checking once per second."""
    print("Press any key to continue...", file=sys.stderr)

    while True:
        if key_pressed():
            key = get_key()
            print(f"\nKey pressed: {repr(key)}")
            break
        time.sleep(1.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
