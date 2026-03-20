import asyncio, json, os, sys
from contextlib import AsyncExitStack
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set in your .env file.")

ai_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are an intelligent Gmail assistant with access to a rich set of
Gmail Analytics tools. When the user asks about their email, use the appropriate tool(s)
to fetch real data and summarise it clearly. Always be concise, helpful, and proactive
in surfacing insights the user might not have asked for but would find valuable."""


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._tools: list[dict] = []

    async def connect_to_server(self, server_script_path: str) -> None:
        if not (server_script_path.endswith(".py") or server_script_path.endswith(".js")):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if server_script_path.endswith(".py") else "node"
        params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        response = await self.session.list_tools()
        self._tools = [self._convert_tool(t) for t in response.tools]
        print(f"\nConnected — {len(self._tools)} tools available.")
        print("Tools:", ", ".join(t["name"] for t in self._tools))

    @staticmethod
    def _convert_tool(tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }

    async def process_query(self, query: str, history: list[dict]) -> tuple[str, list[dict]]:
        messages = history + [{"role": "user", "content": query}]

        while True:
            response = ai_client.chat.completions.create(
                model="gpt-5.3",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                tools=self._tools,
                tool_choice="auto"
            )

            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return msg.content or "(no response)", messages

            tool_results = []
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                print(f"\nCalling tool: {name}({json.dumps(args)[:120]}…)")

                result = await self.session.call_tool(name, args)
                result_text = "\n".join(
                    c.text if hasattr(c, "text") else str(c)
                    for c in result.content
                )

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_text,
                })

            messages.extend(tool_results)

    async def chat_loop(self) -> None:
        print("\nGmail Analytics MCP Client (type 'quit' to exit)")

        history: list[dict] = []

        while True:
            try:
                query = input("\nYou: ").strip()
                if not query:
                    continue
                if query.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if query.lower() == "history":
                    print(f"Conversation turns: {len(history) // 2}")
                    continue
                if query.lower() == "clear":
                    history = []
                    print("History cleared.")
                    continue

                response, history = await self.process_query(query, history)
                print(f"\nAssistant:\n{response}")

            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as exc:
                print(f"\nError: {exc}")

    async def cleanup(self) -> None:
        await self.exit_stack.aclose()


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
