# Part 4: Model Context Protocol (MCP)

The Model Context Protocol (MCP) is an open standard for connecting AI assistants to external data sources and tools. It enables seamless integration between LLMs and various services, databases, and APIs through a standardized protocol.


```python
%pip install mcp
```


```python
from google import genai
from google.genai import types
import sys
import os
import asyncio
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
else:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY',None)

# Create client with api key
MODEL_ID = "gemini-2.5-flash-preview-05-20"
client = genai.Client(api_key=GEMINI_API_KEY)
```

## What is MCP?

Model Context Protocol (MCP) is a revolutionary approach to extending AI capabilities. Unlike traditional function calling where you define functions locally in your code, MCP allows AI models to connect to remote servers that provide tools and resources.


- **🔌 Plug-and-Play Integration**: Connect to any MCP-compatible service instantly
- **🌐 Remote Capabilities**: Access tools and data from anywhere on the internet
- **🔄 Standardized Protocol**: One protocol works with all MCP servers
- **🔒 Centralized Security**: Control access and permissions at the server level
- **📈 Scalability**: Share resources across multiple AI applications
- **🛠️ Rich Ecosystem**: Growing library of MCP servers for various use case

## 1. Working with Stdio MCP Servers

Stdio (Standard Input/Output) servers run as local processes and communicate through pipes. This is perfect for:
- Development and testing
- Local tools and utilities
- Lightweight integrations


## 1. Working with MCP Servers

Let's use the DeepWiki MCP server, which provides access to Wikipedia data and search capabilities:


```python
# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="npx",  # Executable
    args=["-y", "@philschmid/weather-mcp"],  # MCP Server
    env=None,  # Optional environment variables
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Prompt to get the weather for the current day in London.
            prompt = f"What is the weather in London in {datetime.now().strftime('%Y-%m-%d')}?"
            # Initialize the connection between client and server
            await session.initialize()
            # Send request to the model with MCP function declarations
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],  # uses the session, will automatically call the tool
                    # Uncomment if you **don't** want the sdk to automatically call the tool
                    # automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                    #     disable=True
                    # ),
                ),
            )
            print(response.text)

await run()
```

    OK. The weather in London on 2025-05-30 will be: The temperature in Celsius will be 15.6 at 00:00, 15.3 at 01:00, 15.3 at 02:00, 15.3 at 03:00, 15.1 at 04:00, 15.3 at 05:00, 14.6 at 06:00, 15.7 at 07:00, 17 at 08:00, 17.8 at 09:00, 19.4 at 10:00, 20.9 at 11:00, 22.1 at 12:00, 23.3 at 13:00, 24 at 14:00, 23.7 at 15:00, 23.7 at 16:00, 23.1 at 17:00, 22.8 at 18:00, 21 at 19:00, 20.2 at 20:00, 19.3 at 21:00, 18.5 at 22:00, 17.9 at 23:00.


## !! Exercise: Build Your Own MCP CLI Agent !!

Create an interactive command-line interface (CLI) chat agent that connects to the DeepWiki MCP server (a remote server providing access to Wikipedia-like data). The agent should allow users to ask questions about GitHub repositories, and it will use the DeepWiki server to find answers.

Task:
- Use `mcp.client.streamable_http.streamablehttp_client` to establish a connection to the remote URL.
- Inside the `async with streamablehttp_client(...)` block, create an `mcp.ClientSession`.
- Initialize the session using `await session.initialize()`.
- Create a `genai.types.GenerateContentConfig` with `temperature=0` and pass the `session` object in the `tools` list. This configures the chat to use the MCP server.
- Create an asynchronous chat session using `client.aio.chats.create()`, passing the `MODEL_ID` (e.g., "gemini-2.5-flash-preview-05-20") and the `config` you created.
- Implement an interactive loop to chat with the model using `input()` to get the user's input.


```python
remote_url = "https://mcp.deepwiki.com/mcp"

async def run():
    async with streamablehttp_client(remote_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Initialize conversation history using simple tuples
            config = genai.types.GenerateContentConfig(
                temperature=0,
                tools=[session],
            )
            print("Agent is ready. Type 'exit' to quit.")
            chat = client.aio.chats.create(model="gemini-2.5-flash-preview-05-20", config=config)
            while True:
                user_input = input("You: ")
                if user_input.lower() == "exit":
                    print("Exiting chat.")
                    break

                # Append user message to history
                response = await chat.send_message(user_input)
                if len(response.automatic_function_calling_history) > 0:
                    if (
                        response.automatic_function_calling_history[0].parts[0].text
                        == user_input
                    ):
                        response.automatic_function_calling_history.pop(0)
                    for call in response.automatic_function_calling_history:
                        if call.parts[0].function_call:
                            print(f"Function call: {call.parts[0].function_call}")
                        elif call.parts[0].function_response:
                            print(
                                f"Function response: {call.parts[0].function_response.response['result'].content[0].text}"
                            )
                print(f"Assistant: {response.text}")

await run()
```

## Recap & Next Steps

**What You've Learned:**
- Understanding the Model Context Protocol (MCP) and its advantages over traditional function calling
- Connecting to remote MCP servers using both stdio and HTTP protocols
- Building interactive chat agents that leverage MCP capabilities

**Key Takeaways:**
- MCP enables plug-and-play integration with external services and data sources
- Remote capabilities provide access to tools and data from anywhere on the internet
- Standardized protocols ensure compatibility across different AI applications
- Centralized security and permissions improve enterprise deployment scenarios
- The MCP ecosystem is rapidly growing with servers for various use cases

🎉 **Congratulations!** You've completed the Gemini 2.5 AI Engineering Workshop

**More Resources:**
- [MCP with Gemini Documentation](https://ai.google.dev/gemini-api/docs/function-calling?example=weather#model_context_protocol_mcp)
- [Function Calling Documentation](https://ai.google.dev/gemini-api/docs/function-calling?lang=python)
- [MCP Official Specification](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Server Directory](https://github.com/modelcontextprotocol/servers)
