#创建 session → 发送“老人少走路”修改需求 → RAG 辅助意图识别 → 保存 pending_revision_summary

import argparse
import json
import sys

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8000/api/trip"


def make_payload() -> dict:
    request = {
        "city": "北京",
        "start_date": "2026-08-01",
        "end_date": "2026-08-02",
        "travel_days": 2,
        "transportation": "公共交通",
        "accommodation": "经济型酒店",
        "preferences": ["历史文化"],
        "free_text_input": "",
    }

    plan = {
        "city": "北京",
        "start_date": "2026-08-01",
        "end_date": "2026-08-02",
        "days": [
            {
                "date": "2026-08-01",
                "day_index": 0,
                "description": "第一天参观北京核心历史文化景点。",
                "transportation": "公共交通",
                "accommodation": "经济型酒店",
                "attractions": [
                    {
                        "name": "故宫博物院",
                        "address": "景山前街4号",
                        "location": {
                            "longitude": 116.397005,
                            "latitude": 39.919278,
                        },
                        "visit_duration": 180,
                        "description": "历史文化景点",
                        "category": "历史文化",
                        "ticket_price": 60,
                    },
                    {
                        "name": "八达岭长城",
                        "address": "G6京藏高速58号出口",
                        "location": {
                            "longitude": 116.016802,
                            "latitude": 40.356029,
                        },
                        "visit_duration": 240,
                        "description": "游览强度较高的历史景点",
                        "category": "历史文化",
                        "ticket_price": 40,
                    },
                ],
                "meals": [
                    {"type": "breakfast", "name": "早餐"},
                    {"type": "lunch", "name": "午餐"},
                    {"type": "dinner", "name": "晚餐"},
                ],
            },
            {
                "date": "2026-08-02",
                "day_index": 1,
                "description": "第二天继续参观博物馆和城市景点。",
                "transportation": "公共交通",
                "accommodation": "经济型酒店",
                "attractions": [
                    {
                        "name": "中国国家博物馆",
                        "address": "东长安街16号天安门广场东侧",
                        "location": {
                            "longitude": 116.397755,
                            "latitude": 39.903182,
                        },
                        "visit_duration": 180,
                        "description": "大型博物馆",
                        "category": "历史文化",
                        "ticket_price": 0,
                    }
                ],
                "meals": [
                    {"type": "breakfast", "name": "早餐"},
                    {"type": "lunch", "name": "午餐"},
                    {"type": "dinner", "name": "晚餐"},
                ],
            },
        ],
        "weather_info": [],
        "overall_suggestions": "建议提前预约热门景点，并合理安排交通时间。",
        "budget": None,
    }

    return {
        "request": request,
        "plan": plan,
    }


def post_json(client: httpx.Client, url: str, payload: dict) -> dict:
    response = client.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def print_json(title: str, data: dict):
    print(f"\n{title}:")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Verify trip session chat and optional revision over HTTP."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--revise",
        action="store_true",
        help="Also call /revise after the chat endpoint saves pending_revision_summary.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    user_message = "我带老人去北京，希望每天轻松一点，不要走太多路"

    try:
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            created = post_json(
                client,
                f"{base_url}/sessions",
                make_payload(),
            )
            session_id = created["data"]["id"]
            print(f"Created session: {session_id}")

            chat_result = post_json(
                client,
                f"{base_url}/sessions/{session_id}/chat",
                {"message": user_message},
            )
            print_json("Chat result", {
                "success": chat_result.get("success"),
                "assistant_message": chat_result["data"]["messages"][-1]["content"],
                "intent": chat_result.get("intent"),
                "pending_revision_summary": chat_result["data"].get(
                    "pending_revision_summary"
                ),
            })

            if not args.revise:
                print("\nSkipped revise step. Add --revise to verify full re-planning.")
                return

            if not chat_result["data"].get("pending_revision_summary"):
                print("\nNo pending_revision_summary found; revise step was not called.")
                return

            revised = post_json(
                client,
                f"{base_url}/sessions/{session_id}/revise",
                {},
            )
            print_json("Revision result", {
                "success": revised.get("success"),
                "message": revised.get("message"),
                "pending_revision_summary": revised["data"].get(
                    "pending_revision_summary"
                ),
                "plan_versions_count": len(revised["data"].get("plan_versions", [])),
                "overall_suggestions": revised["data"]["current_plan"].get(
                    "overall_suggestions"
                ),
            })

    except httpx.ConnectError:
        print(
            "Cannot connect to backend. Start it first with:\n"
            "cd backend\n"
            "$env:PYTHONPATH='.'\n"
            "..\\.venv\\python.exe -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000"
        )
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        print(f"HTTP request failed: {exc}")
        print("Response body:")
        print(exc.response.text)
        sys.exit(1)


if __name__ == "__main__":
    main()
