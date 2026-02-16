import os
import select
import sys
import termios
import tty

import litellm


def flush_stdin():
    """Discard any buffered keystrokes."""
    fd = sys.stdin.fileno()
    while select.select([fd], [], [], 0)[0]:
        os.read(fd, 1024)


def check_interject(timeout=0.5):
    """Wait briefly for Space press. Returns True if user wants to interject."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        flush_stdin()
        print("  [Space to interject] ", end="", flush=True)
        ready, _, _ = select.select([fd], [], [], timeout)
        if ready:
            key = os.read(fd, 1)
            return key == b" "
        return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        # clear the prompt
        print("\r" + " " * 40 + "\r", end="", flush=True)


def model_turn(model, history, label):
    """Run one model turn, or let the user interject. Returns the response text."""
    print(f"\n{'=' * 60}")
    print(f"[{label} - {model}]")
    print(f"{'=' * 60}")

    if check_interject():
        print(f"\n  You (as {label}): ", end="", flush=True)
        response = input()
        print()
    else:
        response = ""
        resp = litellm.completion(model=model, messages=history, stream=True)
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            response += delta
        print()

    return response


def run_dialog(
    seed: str,
    model_a: str = "anthropic/claude-opus-4-6",
    model_b: str = "anthropic/claude-opus-4-6",
) -> None:
    system_a = "You are a helpful assistant"
    system_b = "You are a helpful assistant"

    history_a: list[dict] = [{"role": "system", "content": system_a}]
    history_b: list[dict] = [{"role": "system", "content": system_b}]

    current_message = seed

    while True:
        # Model A's turn
        history_a.append({"role": "user", "content": current_message})
        response_a = model_turn(model_a, history_a, "Model A")
        history_a.append({"role": "assistant", "content": response_a})
        history_b.append({"role": "user", "content": response_a})

        # Model B's turn
        response_b = model_turn(model_b, history_b, "Model B")
        history_b.append({"role": "assistant", "content": response_b})
        history_a.append({"role": "user", "content": response_b})

        current_message = response_b


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run main.py <seed phrase>")
        print('Example: uv run main.py "What is consciousness?"')
        sys.exit(1)

    seed_phrase = sys.argv[1]

    print(f"Seed: {seed_phrase}")
    print("Running forever. Ctrl+C to stop.")
    print("Press Space between turns to impersonate a model.\n")
    try:
        run_dialog(seed_phrase)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
