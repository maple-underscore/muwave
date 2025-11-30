#!/usr/bin/env python3
"""
Simple program to convert an input string into its Unicode code points.

Usage:
  python3 tools/unicode_convert.py "Hello, 世界"
  echo "Hi 👋" | python3 tools/unicode_convert.py
"""

import sys


def to_code_points(s: str) -> str:
    """Return a space-separated string of Unicode code points for s.

    Example: "A😀" -> "U+0041 U+1F600"
    """
    return " ".join(f"U+{ord(ch):04X}" for ch in s)


def main() -> int:
    # If arguments are provided, join them as the input string.
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        # Otherwise read from stdin.
        text = sys.stdin.read()
    text = text.rstrip("\n")

    codes = to_code_points(text)
    # Also print each character with its code for clarity.
    print(codes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
