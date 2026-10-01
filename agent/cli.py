from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage

from agent.orchestrator import graph


# Project root
project_root = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(project_root / ".env")


SYSTEM_PROMPT = project_root / "prompt" / "system_prompt.txt"

with open(SYSTEM_PROMPT, "r", encoding="utf-8") as f:
    system_prompt_content = f.read()


def main():

    print("=" * 60)
    print("        TripMate - AI Travel Assistant")
    print("=" * 60)
    print("Ask me anything about Bangkok, Barcelona, Reykjavik,")
    print("or Tokyo.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    # One conversation/thread
    config = {
        "configurable": {
            "thread_id": "tripmate-cli"
        }
    }

    # Initialize conversation with system prompt
    graph.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt_content)
            ]
        },
        config=config
    )

    while True:

        try:
            user_input = input("\nYou: ").strip()

            # Empty input
            if not user_input:
                print("TripMate: Please enter a question.")
                continue

            # Exit
            if user_input.lower() in ["exit", "quit"]:
                print("\nTripMate: Goodbye! Have a great trip! ✈️")
                break

            # Send user query
            response = graph.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config
            )

            # Get final assistant response
            messages = response["messages"]

            for message in reversed(messages):
                if message.type == "ai" and message.content:
                    print(f"\nTripMate: {message.content}")
                    break

        except KeyboardInterrupt:
            print("\n\nTripMate: Goodbye!")
            break

        except Exception as e:
            print(f"\nTripMate: Sorry, something went wrong: {e}")


if __name__ == "__main__":
    main()