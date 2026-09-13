import json
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# Tra cứu catalog theo danh mục và mức giá tối đa.
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    if not isinstance(category, str) or not category.strip():
        return []

    try:
        price_limit = int(max_price)
    except (TypeError, ValueError):
        return []

    if price_limit < 0:
        return []

    with open(catalog_file, "r", encoding="utf-8") as file:
        products = json.load(file)

    if not isinstance(products, list):
        raise ValueError("Product catalog must contain a JSON array.")

    normalized_category = category.strip().lower()
    return [
        product
        for product in products
        if isinstance(product, dict)
        and isinstance(product.get("category"), str)
        and product["category"].lower() == normalized_category
        and isinstance(product.get("price_vnd"), (int, float))
        and not isinstance(product["price_vnd"], bool)
        and product["price_vnd"] <= price_limit
    ]


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# Tạo ticket mới và lưu nối tiếp vào kho dữ liệu hiện có.
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    if not isinstance(customer_name, str) or not customer_name.strip():
        raise ValueError("customer_name must be a non-empty string.")
    if not isinstance(issue_description, str) or not issue_description.strip():
        raise ValueError("issue_description must be a non-empty string.")
    if not isinstance(priority, str):
        raise ValueError("priority must be one of: low, medium, high.")

    normalized_priority = priority.strip().lower()
    if normalized_priority not in {"low", "medium", "high"}:
        raise ValueError("priority must be one of: low, medium, high.")

    existing_tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        with open(tickets_file, "r", encoding="utf-8") as file:
            loaded_tickets = json.load(file)
        if not isinstance(loaded_tickets, list):
            raise ValueError("Support ticket data must contain a JSON array.")
        existing_tickets = loaded_tickets

    # Use the largest existing suffix instead of len(...) so IDs remain unique
    # even if an earlier ticket was removed or the file contains unrelated data.
    sequence_numbers = []
    for ticket in existing_tickets:
        if not isinstance(ticket, dict):
            continue
        existing_id = ticket.get("ticket_id")
        if not isinstance(existing_id, str):
            continue
        suffix = existing_id.rsplit("-", 1)[-1]
        if suffix.isdigit():
            sequence_numbers.append(int(suffix))

    now = datetime.now(timezone(timedelta(hours=7)))
    sequence = max(sequence_numbers, default=0) + 1
    ticket_id = f"TK-{now.strftime('%Y%m%d')}-{sequence:03d}"

    normalized_name = customer_name.strip()
    normalized_issue = issue_description.strip()
    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": normalized_name,
        "issue_description": normalized_issue,
        "priority": normalized_priority,
        "status": "open",
        "created_at": now.isoformat(),
        "category": "general",
    }
    existing_tickets.append(new_ticket)

    with open(tickets_file, "w", encoding="utf-8") as file:
        json.dump(existing_tickets, file, indent=2, ensure_ascii=False)

    return {
        "ticket_id": ticket_id,
        "customer_name": normalized_name,
        "priority": normalized_priority,
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công.",
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm hoặc dịch vụ Vingroup theo danh mục và mức giá tối đa."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Danh mục cần tra cứu.",
                    "enum": ["xe_dien", "du_lich"],
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ.",
                    "minimum": 0,
                    "default": 999999999999,
                },
            },
            "required": ["category"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_support_ticket",
        "description": "Tạo yêu cầu hỗ trợ mới cho khách hàng.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ và tên khách hàng.",
                    "minLength": 1,
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả vấn đề khách hàng cần hỗ trợ.",
                    "minLength": 1,
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên của yêu cầu hỗ trợ.",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required": ["customer_name", "issue_description"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
