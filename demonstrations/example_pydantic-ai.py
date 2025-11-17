#!/usr/bin/env -S uv run --quiet --script
"""
Example of using Scorable MCP with Pydantic-AI
run it with: OPENAI_API_KEY=... OPENAI_BASE_URL=... uv run example_pydantic-ai.py
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#    "pydantic-ai-slim[mcp,openai]==0.0.54",
# ]
# ///
import os

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

agent_prompt = """
    <instructions>
    You are a support agent handling customer hotel reservations. 
    You will be given a customer's question, a policy, and the current state of the hotel. 
    Your need to strictly comply with all instructions in the policy and verify your answer with the evaluation tools at your disposal.
    Think carefully and reflect on the results of your tool calls before responding.
    Your final response is only cosidered correct if it meets the acceptance criteria.
    </instructions>

    <acceptance_criteria>
    - Response candidate must score above 0.7 as indicated by Scorable evaluators. Use the contents of the policy and current_state tags as the context parameter.
    - At least 2 evaluators from the list of evaluators have been used on your response candidate
    - If evaluators are not available or give errors, respond to the customer with a temporary apology
    </acceptance_criteria>

    <policy>
    1. Do not mention our competitor ACME
    2. Always start with with a greeting
    3. Be brief
    </policy>

    <current_state>
    - Today is 2025-04-10
    - We are fully booked on 2025-04-10
    - We are fully booked on 2025-04-11
    - We are fully booked on 2025-04-12
    - We have 5 rooms left for 2025-04-13
    </current_state>

    <user_question>
    {question}
    </user_question>
    """.strip()

# Assumes the MCP server is already running
scorable_server = MCPServerHTTP(url="http://localhost:9090/sse")

provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)  # set your proxy if needed, e.g. openrouter or litellm
model = OpenAIModel(
    provider=provider, model_name="gpt-4.1"
)  # set your model here, including custom models


class RoomBooking(BaseModel):
    response: str
    booking_success: bool
    evaluation_score: float


agent = Agent(
    model,
    system_prompt=agent_prompt,
    mcp_servers=[scorable_server],
    result_type=RoomBooking,
    end_strategy="exhaustive",  # this allows the agent do do multiple tool calls before responding
)


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run(
            "Hello! I would like to book a room for tomorrow - what are my options? Should I check with ACME too?"
        )
    print(f"Agent Response: {result.data.response}")
    print(f"Booking Success: {result.data.booking_success}")
    print(f"Evaluation Score of the response: {result.data.evaluation_score}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
