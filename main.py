import sys

import litellm


def run_dialog(
    seed: str,
    model_a: str = "anthropic/claude-opus-4-6",
    model_b: str = "anthropic/claude-opus-4-6",
) -> None:
    system_a = "You are a helpful assistant"
    system_b = "You are a helpful assistant"

    history_a: list[dict] = [{"role": "system", "content": system_a}]
    history_b: list[dict] = [{"role": "system", "content": system_b}]

    # Seed message comes from the "user" to model A
    current_message = seed
    turn = 0

    while True:
        # Model A responds
        history_a.append({"role": "user", "content": current_message})

        print(f"\n{'=' * 60}")
        print(f"[Model A - {model_a}]")
        print(f"{'=' * 60}")

        response_a = ""
        resp = litellm.completion(model=model_a, messages=history_a, stream=True)
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            response_a += delta
        print()

        history_a.append({"role": "assistant", "content": response_a})
        history_b.append({"role": "user", "content": response_a})

        # Model B responds
        print(f"\n{'=' * 60}")
        print(f"[Model B - {model_b}]")
        print(f"{'=' * 60}")

        response_b = ""
        resp = litellm.completion(model=model_b, messages=history_b, stream=True)
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            response_b += delta
        print()

        history_b.append({"role": "assistant", "content": response_b})
        history_a.append({"role": "user", "content": response_b})
        current_message = response_b
        turn += 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run main.py <seed phrase>")
        print('Example: uv run main.py "What is consciousness?"')
        sys.exit(1)

    seed_phrase = sys.argv[1]

    print(f"Seed: {seed_phrase}")
    print("Running forever. Ctrl+C to stop.\n")
    try:
        run_dialog(seed_phrase)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
