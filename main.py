# 📋 main.py · [실습 6-1] 심화 ① 접근 코드로 내 API 지키기 — 헤더 X-Access-Code (POST 한 번에 요청 1번)
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.5-flash-lite"
MAX_TURNS = 6                                 # 최근 몇 턴을 기억할지
SYSTEM_PROMPT = "너는 '폴리봇'이야. 인공지능소프트웨어과 1학년의 파이썬 · AI 공부 도우미야. 존댓말로 3문장 이내로 답한다. 개인정보를 묻지 않는다."
app = FastAPI(title="AI 용어 설명 API", description="용어를 보내면 Gemini 가 쉬운 말로 설명해 줍니다.")


class Question(BaseModel):
    word: str = Field(min_length=1, max_length=30, description="궁금한 용어")
    style: str = Field(default="초등학생도 알아듣게 쉽게", description="설명 말투")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500, description="이번에 보낼 말")
    history: list[dict] = Field(default_factory=list, description="지난 대화 기록 (이전 응답의 history 를 그대로 보냄)")


class ChatReply(BaseModel):
    answer: str
    history: list[dict]                       # 다음 요청 때 그대로 다시 보낼 기록


class Answer(BaseModel):                      # 응답(JSON)의 모양 — /docs 에도 그대로 나옴
    word: str
    answer: str
    tokens: int | None = None


def get_client():
    """키는 서버의 환경 변수 GEMINI_API_KEY 에서 읽음 — 코드 · 저장소에 적지 않음"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="서버에 GEMINI_API_KEY 가 없습니다 — 환경 변수를 확인하세요")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60_000, retry_options=types.HttpRetryOptions(http_status_codes=[500, 503])),
    )


def check_access(access_code):
    """서버 환경 변수 APP_ACCESS_CODE 가 있으면, 요청 헤더의 접근 코드가 같아야만 통과"""
    expected = os.environ.get("APP_ACCESS_CODE")
    if expected and access_code != expected:
        raise HTTPException(status_code=401, detail="접근 코드가 틀렸습니다 — 헤더 X-Access-Code 를 확인하세요")


def trim_history(history, max_turns):
    """최근 max_turns 턴만 남긴 새 리스트를 돌려준다 (11주차와 같음)"""
    user_positions = [index for index, step in enumerate(history) if step.get("type") == "user_input"]
    if len(user_positions) <= max_turns:
        return history
    return history[user_positions[-max_turns]:]


def ask_gemini(client, system_instruction, contents):
    """Gemini 에 묻고, 실패하면 알맞은 상태 번호와 한글 안내로 바꿔 돌려보냄"""
    try:
        return client.interactions.create(model=MODEL_NAME, store=False, system_instruction=system_instruction, input=contents)
    except Exception as error:
        code = getattr(error, "status_code", None)
        if code == 429:
            raise HTTPException(status_code=429, detail="Gemini 무료 한도를 넘었습니다 — 1분 뒤 다시 요청하세요")
        raise HTTPException(status_code=502, detail=f"Gemini 요청 실패 (상태 번호 {code})")


@app.get("/")
def home():
    return {"message": "AI 용어 설명 API 입니다. /docs 에서 사용법을 보세요."}


@app.post("/explain", response_model=Answer)
def explain(question: Question, x_access_code: str | None = Header(default=None)):   # 헤더 X-Access-Code 를 받음
    check_access(x_access_code)               # ★ Gemini 를 부르기 전에 먼저 검사 (틀리면 요청이 나가지 않음)
    client = get_client()
    reply = ask_gemini(client, f"너는 AI 용어 설명 도우미야. {question.style} 3문장 이내로 설명한다.", question.word)
    return Answer(word=question.word, answer=reply.output_text, tokens=reply.usage.total_tokens)


@app.post("/chat", response_model=ChatReply)
def chat(request: ChatRequest, x_access_code: str | None = Header(default=None)):
    """서버는 대화를 기억하지 않음 — 부르는 쪽이 history 를 들고 와서 다시 보냄 (11주차 stateless 방식)"""
    check_access(x_access_code)
    client = get_client()
    send_list = trim_history(request.history + [{"type": "user_input", "content": [{"type": "text", "text": request.message}]}], MAX_TURNS)
    reply = ask_gemini(client, SYSTEM_PROMPT, send_list)
    return ChatReply(answer=reply.output_text, history=send_list + [step.model_dump(mode="json") for step in reply.steps])
