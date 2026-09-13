"""
Lab #4: System Prompt Engineering & Tool Calling Engine.

The project runs in mock mode by default: ChatbotBaseline answers without
tools, while ToolCallingAgent performs a deterministic tool-calling loop.
"""

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from tools import TOOL_DEFINITIONS, TOOL_MAP


# ---------------------------------------------------------------------------
# Production System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""
Bạn là VinAssistant — trợ lý AI cho các sản phẩm và dịch vụ Vingroup.

## PERSONA
- Vai trò: tư vấn VinFast/Vinpearl và ghi nhận yêu cầu hỗ trợ.
- Phong cách: chuyên nghiệp, thân thiện, ngắn gọn và chính xác.

## AVAILABLE TOOLS
{json.dumps(TOOL_DEFINITIONS, indent=2, ensure_ascii=False)}

## CORE RULES
1. Không bịa tên, giá, tình trạng sản phẩm, chính sách hoặc mã ticket.
2. Phải dùng search_product_catalog khi người dùng muốn tra cứu catalog.
3. Chỉ dùng submit_support_ticket khi có tên khách hàng và mô tả vấn đề.
4. Mỗi yêu cầu chỉ được tạo một ticket; chỉ xác nhận khi tool thành công.
5. Câu trả lời phải dựa trên Observation và tôn trọng max_iterations.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ nội dung thuộc hệ sinh thái Vingroup.
- Không dùng tool ngoài danh sách, không suy đoán dữ liệu cá nhân.
- Với yêu cầu ngoài phạm vi, giải thích giới hạn một cách lịch sự.

## OUTPUT CONTRACT
- Thought/Decision: lưu intent định tuyến ngắn gọn trong trace.
- Action: tên tool và JSON arguments hợp lệ.
- Observation: kết quả nguyên bản của tool.
- Final Answer: câu trả lời tiếng Việt; không công khai suy luận nội bộ.
""".strip()


def _fold_text(text: str) -> str:
    """Lowercase Vietnamese text and remove accents for keyword matching."""

    normalized = unicodedata.normalize("NFD", text)
    text_without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return text_without_marks.replace("đ", "d").replace("Đ", "D").lower()


def _contains_any(text: str, phrases: Tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


# ---------------------------------------------------------------------------
# Baseline chatbot
# ---------------------------------------------------------------------------


class ChatbotBaseline:
    """Mock chatbot without tool calling, used to illustrate hallucination."""

    def query(self, user_input: str) -> Dict[str, Any]:
        prompt = "" if user_input is None else str(user_input).strip()
        return {
            "answer": (
                "[Chatbot Baseline] Câu trả lời mô phỏng này không được kiểm "
                f"chứng bằng tool: {prompt}"
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline",
        }


# ---------------------------------------------------------------------------
# Deterministic tool-calling agent
# ---------------------------------------------------------------------------


class ToolCallingAgent:
    """Detect intents, execute each required tool once, and compose an answer."""

    CATALOG_TERMS = (
        "cho toi xem",
        "muon xem",
        "tim xe",
        "tim resort",
        "goi y",
        "tu van",
        "danh sach",
        "gia duoi",
        "gia toi da",
        "ngan sach",
    )
    TICKET_TERMS = (
        "tao ticket",
        "mo ticket",
        "ghi nhan phan hoi",
        "ghi nhan su co",
        "phan anh",
        "khieu nai",
        "yeu cau ho tro",
        "can ho tro",
    )
    ISSUE_TERMS = (
        "bi loi",
        "loi he thong",
        "bi hong",
        "khong hoat dong",
        "am moc",
        "su co",
    )

    def __init__(self, max_iterations: int = 5):
        try:
            parsed_limit = int(max_iterations)
        except (TypeError, ValueError):
            parsed_limit = 5
        self.max_iterations = max(0, parsed_limit)
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def _extract_categories(folded_input: str) -> List[str]:
        categories = []
        if _contains_any(folded_input, ("xe dien", "vinfast", "xe vinfast")):
            categories.append("xe_dien")
        if _contains_any(
            folded_input,
            ("vinpearl", "resort", "du lich", "khach san", "ky nghi"),
        ):
            categories.append("du_lich")
        return categories

    @staticmethod
    def _number_to_vnd(number_text: str, unit: str) -> Optional[int]:
        number_text = number_text.replace(" ", "")
        multipliers = {
            "ty": 1_000_000_000,
            "trieu": 1_000_000,
            "tr": 1_000_000,
            "nghin": 1_000,
            "ngan": 1_000,
            "k": 1_000,
            "vnd": 1,
            "dong": 1,
            "d": 1,
            "": 1,
        }
        multiplier = multipliers.get(unit, 1)
        try:
            last_group = re.split(r"[.,]", number_text)[-1]
            is_decimal = (
                multiplier > 1
                and number_text.count(".") + number_text.count(",") == 1
                and len(last_group) <= 2
            )
            normalized = (
                number_text.replace(",", ".")
                if is_decimal
                else re.sub(r"[.,]", "", number_text)
            )
            return int(float(normalized) * multiplier)
        except ValueError:
            return None

    @classmethod
    def _extract_max_price(cls, folded_input: str) -> Optional[int]:
        # Currency units make an amount unambiguous and also support
        # expressions such as "1 tỷ 200 triệu".
        amount_pattern = re.compile(
            r"(?<![a-z0-9])(\d[\d.,]*)\s*"
            r"(ty|trieu|tr|nghin|ngan|k|vnd|dong|d)\b"
        )
        amounts = [
            cls._number_to_vnd(match.group(1), match.group(2))
            for match in amount_pattern.finditer(folded_input)
        ]
        valid_amounts = [amount for amount in amounts if amount is not None]
        if valid_amounts:
            return sum(valid_amounts)

        # A bare number is accepted only after an explicit price expression.
        bare_price = re.search(
            r"(?:gia(?:\s+(?:duoi|toi da|khong qua|den|tam|khoang))?"
            r"|duoi|toi da(?:\s+la)?|khong qua|ngan sach(?:\s+la)?|tam|khoang)"
            r"\s*:?\s*(\d[\d.,]*)",
            folded_input,
        )
        if bare_price:
            return cls._number_to_vnd(bare_price.group(1), "")
        return None

    @staticmethod
    def _extract_customer_name(user_input: str) -> Tuple[Optional[str], int]:
        folded = _fold_text(user_input)
        patterns = (
            r"\btoi\s+ten\s+([^,.;:\n]+)",
            r"\bten\s+(?:cua\s+)?toi\s+la\s+([^,.;:\n]+)",
            r"\bten\s+khach\s+hang\s+(?:la\s+)?([^,.;:\n]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, folded)
            if not match:
                continue
            start, end = match.span(1)
            candidate = user_input[start:end].strip(" -–—")
            stop = re.search(
                r"\s+(?:va|xe|phong|dang|hien|can)\b", _fold_text(candidate)
            )
            if stop:
                candidate = candidate[: stop.start()].strip()
                end = start + stop.start()
            if candidate and len(candidate.split()) <= 8:
                return candidate, end
        return None, -1

    @classmethod
    def _extract_issue(cls, user_input: str, name_end: int) -> Optional[str]:
        candidates = []
        if name_end >= 0:
            candidates.append(user_input[name_end:].lstrip(" ,.;:-–—"))
        candidates.extend(re.split(r"(?<=[.!?])\s+", user_input))

        issue = next(
            (
                text
                for text in candidates
                if _contains_any(_fold_text(text), cls.ISSUE_TERMS)
            ),
            "",
        )
        if not issue:
            return None

        folded_issue = _fold_text(issue)
        cut_markers = (
            ", muc do",
            ", uu tien",
            ". day la van de",
            ". day la su co",
            ". can xu ly",
        )
        cut_positions = [folded_issue.find(marker) for marker in cut_markers]
        cut_positions = [position for position in cut_positions if position >= 0]
        if cut_positions:
            issue = issue[: min(cut_positions)]

        colon = issue.find(":")
        if colon >= 0 and _contains_any(
            _fold_text(issue[:colon]), cls.TICKET_TERMS
        ):
            issue = issue[colon + 1 :]

        issue = re.sub(r"\bcủa\s+tôi\b\s*", "", issue, flags=re.IGNORECASE)
        issue = re.sub(
            r"\bphòng\s+tôi\s+đặt\b", "phòng đặt", issue, flags=re.IGNORECASE
        )
        issue = re.sub(r"\s+", " ", issue).strip(" ,.;:-")
        return issue[0].upper() + issue[1:] if issue else None

    @staticmethod
    def _extract_priority(folded_input: str) -> str:
        if _contains_any(
            folded_input, ("khong gap", "muc do thap", "uu tien thap")
        ):
            return "low"
        if _contains_any(
            folded_input,
            ("nghiem trong", "khan cap", "xu ly gap", "can gap", "uu tien cao"),
        ):
            return "high"
        return "medium"

    @classmethod
    def _detect_intents(cls, user_input: str) -> Dict[str, Any]:
        folded = _fold_text(user_input)
        categories = cls._extract_categories(folded)
        original_lower = user_input.lower()
        asks_if_available = bool(
            re.search(r"\bcó\s+(?:xe|resort)\b", original_lower)
            or re.match(r"^\s*co\s+(?:xe|resort)\b", folded)
        )
        warranty_faq = "bao hanh" in folded and _contains_any(
            folded, ("xe dien", "vinfast", "vf ")
        )
        explicit_catalog = bool(categories) and (
            _contains_any(folded, cls.CATALOG_TERMS) or asks_if_available
        )
        needs_catalog = explicit_catalog and not (
            warranty_faq
            and not _contains_any(
                folded, ("cho toi xem", "muon xem", "tim xe")
            )
            and not asks_if_available
        )

        customer_name, name_end = cls._extract_customer_name(user_input)
        has_issue = _contains_any(folded, cls.ISSUE_TERMS)
        needs_ticket = _contains_any(folded, cls.TICKET_TERMS) or bool(
            customer_name and has_issue
        )
        issue = cls._extract_issue(user_input, name_end) if needs_ticket else None

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": warranty_faq,
            "categories": categories if needs_catalog else [],
            "max_price": cls._extract_max_price(folded),
            "customer_name": customer_name,
            "issue_description": issue,
            "priority": cls._extract_priority(folded),
        }

    @staticmethod
    def _missing_ticket_fields(intents: Dict[str, Any]) -> List[str]:
        missing = []
        if not intents.get("customer_name"):
            missing.append("tên khách hàng")
        if not intents.get("issue_description"):
            missing.append("mô tả vấn đề")
        return missing

    def _build_actions(self, intents: Dict[str, Any]) -> List[Dict[str, Any]]:
        actions = []
        for category in intents["categories"]:
            arguments = {"category": category}
            if intents["max_price"] is not None:
                arguments["max_price"] = intents["max_price"]
            actions.append({"tool": "search_product_catalog", "arguments": arguments})

        if intents["needs_ticket"] and not self._missing_ticket_fields(intents):
            actions.append(
                {
                    "tool": "submit_support_ticket",
                    "arguments": {
                        "customer_name": intents["customer_name"],
                        "issue_description": intents["issue_description"],
                        "priority": intents["priority"],
                    },
                }
            )
        return actions

    @staticmethod
    def _format_price(value: Any) -> str:
        try:
            return f"{int(value):,}".replace(",", ".") + " VNĐ"
        except (TypeError, ValueError):
            return "giá chưa xác định"

    @classmethod
    def _format_catalog(cls, result: Any, category: str) -> str:
        label = "xe điện" if category == "xe_dien" else "dịch vụ du lịch"
        if (
            isinstance(result, list)
            and len(result) == 1
            and isinstance(result[0], dict)
            and result[0].get("error")
        ):
            return "Không thể tra cứu catalog lúc này: " + str(result[0]["error"])

        products = (
            [item for item in result if isinstance(item, dict) and item.get("name")]
            if isinstance(result, list)
            else []
        )
        if not products:
            return f"Rất tiếc, không tìm thấy {label} phù hợp với yêu cầu."

        products.sort(key=lambda item: item.get("price_vnd", float("inf")))
        lines = [f"Các {label} phù hợp:"]
        lines.extend(
            f"- {item['name']}: {cls._format_price(item.get('price_vnd'))}."
            for item in products
        )
        return "\n".join(lines)

    @staticmethod
    def _format_ticket(result: Any) -> str:
        if not isinstance(result, dict) or result.get("error"):
            detail = result.get("error") if isinstance(result, dict) else "dữ liệu không hợp lệ"
            return f"Chưa thể tạo phiếu hỗ trợ: {detail}."
        ticket_id = result.get("ticket_id")
        if not ticket_id:
            return "Chưa thể tạo phiếu hỗ trợ; hệ thống chưa cấp mã ticket."
        return (
            f"Đã tạo phiếu hỗ trợ {ticket_id} cho "
            f"{result.get('customer_name', 'khách hàng')}; "
            f"mức ưu tiên {result.get('priority', 'medium')}, "
            f"trạng thái {result.get('status', 'open')}."
        )

    @staticmethod
    def _faq_answer() -> str:
        return (
            "Theo dữ liệu của bài lab, pin xe điện VinFast được bảo hành 10 năm. "
            "Điều kiện cụ thể có thể phụ thuộc mẫu xe và hợp đồng."
        )

    def _compose_answer(
        self,
        user_input: str,
        intents: Dict[str, Any],
        observations: List[Dict[str, Any]],
    ) -> str:
        parts = []
        for observation in observations:
            if observation["tool"] == "search_product_catalog":
                parts.append(
                    self._format_catalog(
                        observation["result"], observation["arguments"]["category"]
                    )
                )
            else:
                parts.append(self._format_ticket(observation["result"]))

        missing = self._missing_ticket_fields(intents)
        if intents["needs_ticket"] and missing:
            parts.append(
                "Để tạo phiếu hỗ trợ, vui lòng bổ sung " + " và ".join(missing) + "."
            )
        if intents["is_faq"]:
            parts.append(self._faq_answer())
        if parts:
            return "\n\n".join(parts)
        if not user_input:
            return "Bạn vui lòng nhập yêu cầu cần VinAssistant hỗ trợ."
        if _fold_text(user_input).strip(" !.,") in {"xin chao", "chao", "hello", "hi"}:
            return "Xin chào! Tôi có thể hỗ trợ bạn về VinFast và Vinpearl."
        return (
            "Tôi chỉ hỗ trợ các sản phẩm và dịch vụ thuộc hệ sinh thái Vingroup."
        )

    def _finish(
        self,
        answer: str,
        iterations: int,
        status: str,
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        self.trace.append(
            {
                "step": "final_answer",
                "iteration": iterations,
                "status": status,
                "answer": answer,
            }
        )
        return {
            "answer": answer,
            "trace": self.trace,
            "tool_calls": tool_calls,
            "iterations": iterations,
            "status": status,
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Main entry point for the bounded agent loop."""

        prompt = "" if user_input is None else str(user_input).strip()
        self.trace = []
        intents = self._detect_intents(prompt)
        self.trace.append(
            {
                "step": "decision",
                "iteration": 0,
                "user_input": prompt,
                "intents": intents,
            }
        )

        if self.max_iterations == 0:
            return self._finish(
                "Lỗi: đã đạt giới hạn số bước tối đa.",
                0,
                "max_iterations_reached",
                [],
            )

        actions = self._build_actions(intents)
        observations = []
        tool_calls = []
        iterations = 0

        for action in actions:
            if iterations >= self.max_iterations:
                return self._finish(
                    "Lỗi: đã đạt giới hạn số bước tối đa; một số thao tác chưa thực hiện.",
                    iterations,
                    "max_iterations_reached",
                    tool_calls,
                )

            iterations += 1
            tool_name = action["tool"]
            arguments = dict(action["arguments"])
            self.trace.append(
                {
                    "step": "action",
                    "iteration": iterations,
                    "tool": tool_name,
                    "arguments": arguments,
                }
            )
            tool_calls.append({"name": tool_name, "arguments": arguments})

            try:
                tool = TOOL_MAP[tool_name]
                result = tool(**arguments)
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}"}

            observation = {
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            }
            observations.append(observation)
            self.trace.append(
                {
                    "step": "observation",
                    "iteration": iterations,
                    **observation,
                }
            )

        # FAQ/direct responses still count as one completed decision iteration.
        if not actions:
            iterations = 1
        answer = self._compose_answer(prompt, intents, observations)
        return self._finish(answer, iterations, "completed", tool_calls)


def main() -> None:
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."
    print("=== RUNNING CHATBOT BASELINE ===")
    print(ChatbotBaseline().query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
