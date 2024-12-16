from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


@app.get("/", tags=["Health"])
def root():
    return JSONResponse(content={"message": "Testing out Pydantic AI"}, status_code=200)


@app.get("/ai-agent", tags=["Pydnatic-AI"])
async def ai_agent(query: str):
    try:
        model = GeminiModel(
            "gemini-1.5-flash",
            api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
        )
        agent = Agent(
            model,
            system_prompt="""You are a personal productivity assistant. Your role is to help users manage their tasks, schedules, and reminders effectively. 
When a user provides a command:
1. Identify the task type (e.g., reminder, meeting, task scheduling).
2. Extract important details like time, date, participants, and description.
3. Provide clear feedback or confirm the action.
4. Suggest optimal scheduling options if the user provides incomplete details.
5. Offer productivity tips if requested.

Examples:
- User: "Remind me to call Sarah at 2 PM tomorrow."
  Response: "Sure! I've set a reminder for you to call Sarah at 2 PM tomorrow."

- User: "Schedule a meeting with the team on Friday at 3 PM."
  Response: "Got it! I've scheduled a meeting with the team on Friday at 3 PM."

- User: "What are some productivity tips for managing my time better?"
  Response: "Here are some tips: 1) Prioritize your tasks using the Eisenhower matrix. 2) Break larger tasks into smaller, manageable parts. 3) Use time-blocking to allocate focused time for each task."

Ensure your response is concise, helpful, and actionable.
""",
        )
        response = await agent.run(query)
        return JSONResponse(content={"data": response.data}, status_code=200)

    except Exception as e:
        raise
