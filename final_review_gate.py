# final_review_gate.py
import sys
import os

if __name__ == "__main__":
    # Try to make stdout unbuffered for more responsive interaction.
    # This might not work on all platforms or if stdout is not a TTY,
    # but it's a good practice for this kind of interactive script.
    try:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    except Exception:
        pass  # Ignore if unbuffering fails, e.g., in certain environments

    try:
        sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
    except Exception:
        pass  # Ignore

    print("--- FINAL REVIEW GATE ACTIVE ---", flush=True)
    print("AI has completed its primary actions. Awaiting your review or further sub-prompts.", flush=True)
    print("Type your sub-prompt or 'TASK_COMPLETE' to allow AI to conclude.", flush=True)

    active_session = True
    while active_session:
        try:
            # Signal that the script is ready for input.
            print("REVIEW_GATE_AWAITING_INPUT:", end="", flush=True)

            line = sys.stdin.readline()

            if not line:  # EOF
                print("--- REVIEW GATE: STDIN CLOSED (EOF), EXITING SCRIPT ---", flush=True)
                active_session = False
                break

            user_input = line.strip()

            if user_input.upper() == 'TASK_COMPLETE':
                print("--- REVIEW GATE: USER CONFIRMED TASK COMPLETE ---", flush=True)
                active_session = False
                break
            elif user_input:  # Any non-empty input other than TASK_COMPLETE
                print(f"USER_REVIEW_SUB_PROMPT: {user_input}", flush=True)

        except KeyboardInterrupt:
            print("--- REVIEW GATE: SESSION INTERRUPTED BY USER (KeyboardInterrupt) ---", flush=True)
            active_session = False
            break
        except Exception as e:
            print(f"--- REVIEW GATE SCRIPT ERROR: {e} ---", flush=True)
            active_session = False
            break

    print("--- FINAL REVIEW GATE SCRIPT EXITED ---", flush=True)
