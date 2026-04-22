import os
import json
import sqlite3
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = """
AutoStream Pricing:
- Basic Plan: $29/month — 10 videos/month, 720p resolution
- Pro Plan: $79/month — Unlimited videos, 4K resolution, AI captions, 24/7 priority support

Policies:
- No refund after 7 days of purchase
- 24/7 live support available for Pro users only
- Basic users have access to email support (48hr response time)

Best fit guide:
- Basic Plan → Hobbyists, beginners, 1-2 uploads per week
- Pro Plan → Full-time creators, agencies, daily uploaders, those needing AI tools
"""

SYSTEM_PERSONA = """You are Aria, a friendly and helpful sales assistant for AutoStream — a video hosting platform for content creators.

Your personality:
- Warm, conversational, and encouraging
- Never robotic or overly formal
- Use light emojis where natural (not every sentence)
- Keep responses concise but helpful
- If a user seems confused, patiently clarify

Always stay on topic. If asked something off-topic, gently steer back to AutoStream.
"""

# ─────────────────────────────────────────────
# 2. MODEL
# ─────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

# ─────────────────────────────────────────────
# 3. MOCK TOOL — Lead Capture
# ─────────────────────────────────────────────
def mock_lead_capture(name: str, email: str, platform: str):
    """Simulates saving lead to CRM/database."""
    print(f"\n{'='*50}")
    print(f"LEAD CAPTURED SUCCESSFULLY")
    print(f"    Name     : {name}")
    print(f"    Email    : {email}")
    print(f"    Platform : {platform}")
    print(f"{'='*50}\n")

# ─────────────────────────────────────────────
# 4. STATE
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    intent: str          # "greeting" | "inquiry" | "high_intent" | "completed"
    lead_info: dict      # {"name": ..., "email": ..., "platform": ...}

# ─────────────────────────────────────────────
# 5. NODES
# ─────────────────────────────────────────────

def intent_classifier(state: AgentState):
    """
    Classifies the user's LATEST message.
    IMPORTANT: If we're already in lead capture flow, preserve that intent.
    """
    current_intent = state.get("intent", "")
    lead_info = state.get("lead_info", {})

    # KEY FIX: Don't re-classify if we're mid lead capture flow
    # (lead_info exists but is incomplete → keep collecting)
    if current_intent in ("high_intent",) and lead_info:
        missing = [k for k in ["name", "email", "platform"] if not lead_info.get(k)]
        if missing:
            return {"intent": "high_intent"}  # Stay in lead capture

    last_msg = state["messages"][-1].content

    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an intent classifier for a SaaS chatbot. "
            "Classify the user message into EXACTLY ONE of these categories:\n"
            "- greeting    → simple hello/hi/hey\n"
            "- inquiry     → asking about features, pricing, policy, comparisons\n"
            "- high_intent → wants to buy, sign up, try, get started, subscribe, upgrade\n\n"
            "Respond with ONLY the category word. Nothing else."
        )),
        ("human", "{msg}")
    ])

    result = (classify_prompt | llm).invoke({"msg": last_msg})
    raw = result.content.strip().lower()

    # Map to clean intent
    if "greeting" in raw:
        intent = "greeting"
    elif "high_intent" in raw or any(kw in last_msg.lower() for kw in
                                      ["buy", "sign up", "get started", "subscribe", "try", "want pro", "want basic", "start"]):
        intent = "high_intent"
    else:
        intent = "inquiry"

    return {"intent": intent}


def rag_node(state: AgentState):
    """
    Handles greetings and general inquiries using the knowledge base.
    Has access to full conversation history for context.
    """
    # Build conversation history for context
    history = state["messages"][:-1]  # All except the latest
    last_msg = state["messages"][-1].content

    history_text = ""
    for msg in history[-6:]:  # Last 3 turns (6 messages)
        role = "User" if isinstance(msg, HumanMessage) else "Aria"
        history_text += f"{role}: {msg.content}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PERSONA + "\n\nKnowledge Base:\n" + KNOWLEDGE_BASE),
        ("human", (
            "Conversation so far:\n{history}\n\n"
            "Latest message: {question}\n\n"
            "Respond naturally as Aria. Use the knowledge base if relevant. "
            "Keep it concise and friendly. Don't mention you're an AI unless asked."
        ))
    ])

    response = (prompt | llm).invoke({
        "history": history_text or "None",
        "question": last_msg
    })

    return {"messages": [AIMessage(content=response.content)]}


def lead_capture_node(state: AgentState):
    """
    Step-by-step lead collection: name → email → platform → save.
    Handles edge cases like user confusion or off-topic replies.
    """
    lead_data = state.get("lead_info") or {"name": None, "email": None, "platform": None}
    last_user_msg = state["messages"][-1].content

    # ── Step 1: Extract any info from user's message ──
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Extract contact info from the user message. "
            "Current collected data is given. Only update fields if new info is present.\n"
            "Return ONLY a valid JSON object. No prose. No backticks. No explanation.\n"
            "JSON keys: name, email, platform (values can be null if not found)\n"
            "For platform, only accept: YouTube, Instagram, TikTok, Facebook, Twitter, Other"
        )),
        ("human", (
            "Current data: {current_data}\n"
            "User message: '{user_msg}'\n"
            "Return updated JSON:"
        ))
    ])

    try:
        raw = (extract_prompt | llm).invoke({
            "current_data": json.dumps(lead_data),
            "user_msg": last_user_msg
        }).content

        clean = raw.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(clean)

        for key in ["name", "email", "platform"]:
            val = extracted.get(key)
            if val and str(val).lower() not in ("null", "none", ""):
                lead_data[key] = val

    except Exception as e:
        print(f"[Extraction skipped: {e}]")

    # ── Step 2: Check what's still missing & respond accordingly ──
    
    # Build a context-aware response using LLM for naturalness
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PERSONA),
        ("human", (
            "You are collecting lead info step by step. Current collected data: {lead_data}\n"
            "User just said: '{user_msg}'\n\n"
            "Based on what's still missing, do ONE of these:\n"
            "1. If name is missing → warmly ask for their name\n"
            "2. If email is missing → ask for email (if user questions why, briefly explain it's for account setup)\n"
            "3. If platform is missing → ask which platform they create content on\n"
            "4. If all collected → say you're all set and team will reach out soon 🎉\n\n"
            "Be natural and friendly. Don't repeat what they already told you. "
            "If user seems confused/resistant, handle it gracefully before asking again."
        ))
    ])

    if not lead_data.get("name"):
        response = (response_prompt | llm).invoke({
            "lead_data": json.dumps(lead_data),
            "user_msg": last_user_msg
        })
        return {"messages": [AIMessage(content=response.content)], "lead_info": lead_data}

    elif not lead_data.get("email"):
        response = (response_prompt | llm).invoke({
            "lead_data": json.dumps(lead_data),
            "user_msg": last_user_msg
        })
        return {"messages": [AIMessage(content=response.content)], "lead_info": lead_data}

    elif not lead_data.get("platform"):
        response = (response_prompt | llm).invoke({
            "lead_data": json.dumps(lead_data),
            "user_msg": last_user_msg
        })
        return {"messages": [AIMessage(content=response.content)], "lead_info": lead_data}

    else:
        #All info collected — fire the tool!
        mock_lead_capture(lead_data["name"], lead_data["email"], lead_data["platform"])

        farewell = (
            f"You're all set, {lead_data['name']}!\n"
            f"Our team will reach out to you at {lead_data['email']} shortly to get your Pro plan activated. "
            f"Welcome to AutoStream!"
        )
        return {
            "messages": [AIMessage(content=farewell)],
            "lead_info": lead_data,
            "intent": "completed"
        }


# ─────────────────────────────────────────────
# 6. GRAPH ASSEMBLY
# ─────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("classifier", intent_classifier)
workflow.add_node("rag", rag_node)
workflow.add_node("lead_capture", lead_capture_node)

workflow.set_entry_point("classifier")


def router(state: AgentState):
    intent = state.get("intent", "inquiry")
    if intent == "high_intent":
        return "lead_capture"
    return "rag"  # greeting + inquiry both go to RAG (RAG handles both naturally)


workflow.add_conditional_edges("classifier", router, {
    "lead_capture": "lead_capture",
    "rag": "rag"
})
workflow.add_edge("rag", END)
workflow.add_edge("lead_capture", END)

# ─────────────────────────────────────────────
# 7. MEMORY — SQLite Checkpointer
# ─────────────────────────────────────────────
conn = sqlite3.connect("state_db.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)

# ─────────────────────────────────────────────
# 8. CHAT RUNNER
# ─────────────────────────────────────────────
def run_chat():
    thread_id = "user_demo_001"  # In prod: use phone number / user ID
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "="*50)
    print("AutoStream AI Assistant — Aria")
    print("="*50)
    print("  Type 'exit' or 'quit' to end the chat\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            print("Aria: Thanks for chatting! Have a great day")
            break

        for event in app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values"
        ):
            msgs = event.get("messages", [])
            if msgs:
                last = msgs[-1]
                if isinstance(last, AIMessage):
                    print(f"\nAria: {last.content}\n")
                    break  # Only print once per turn


if __name__ == "__main__":
    run_chat()