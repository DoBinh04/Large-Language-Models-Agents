# Vì sao LangGraph dùng StateGraph thay vì chain đơn giản?

## 1. Chain là gì và nó thiếu gì

Một "chain" (LCEL `prompt | llm | parser`, hay `SequentialChain`) là **hàm hợp tuyến tính**: `f3(f2(f1(x)))`.
Dữ liệu chảy một chiều, mỗi bước chạy đúng một lần, và chain kết thúc khi bước cuối trả về.

Agent ở bài 3.3 lại không có dạng đó:

```
LLM → (cần tool?) → tool → LLM → (cần tool?) → tool → LLM → ... → trả lời
```

| Agent cần | Chain | StateGraph |
|---|---|---|
| **Vòng lặp** với số vòng không biết trước | Không có. DAG không có chu trình, phải viết `while` bên ngoài chain | Có cạnh `tools → agent`, nên chu trình là thứ tự nhiên của graph |
| **Rẽ nhánh** theo output của LLM (trả lời / tool an toàn / tool nguy hiểm) | `RunnableBranch` lồng nhau, khó đọc | `add_conditional_edges(agent, route_after_agent)` |
| **State chung** nhiều bước cùng đọc/ghi (messages, steps, approvals...) | Mỗi bước chỉ thấy output của bước trước; phải tự nhét mọi thứ vào dict truyền tay | `TypedDict` có schema, mỗi field có **reducer** (`messages: Annotated[list, operator.add]`) nên node chỉ trả về phần thay đổi |
| **Dừng giữa chừng chờ người**, rồi chạy tiếp, có thể ở process khác hoặc vài giờ sau | Không làm được: chain là một lời gọi hàm, dừng nghĩa là mất hết | `interrupt()` + checkpointer: state được lưu sau mỗi bước, và `Command(resume=...)` chạy tiếp đúng chỗ |
| Khôi phục sau lỗi, xem lại lịch sử, "time travel" | Không | Mỗi super-step là một checkpoint, xem được bằng `get_state_history(config)` và chạy lại được từ checkpoint cũ |
| Streaming từng bước, quan sát node nào đang chạy | Chỉ stream token | `stream_mode="updates" / "values" / "messages"` |
| Nhiều agent cùng phối hợp (bài 3.6) | Rất khó | Mỗi agent là một node (hoặc subgraph), giao tiếp qua state |

## 2. Mô hình bên dưới: Pregel / Bulk Synchronous Parallel

`StateGraph.compile()` không tạo ra một hàm, mà tạo ra một **máy trạng thái** chạy theo từng *super-step*:

1. Chọn các node được kích hoạt (theo cạnh từ bước trước).
2. Chạy chúng (song song nếu có nhiều node), mỗi node đọc **snapshot** của state.
3. Gộp các update bằng reducer của từng field. Nhờ đó 2 node song song cùng ghi `messages` không đè lên nhau.
4. **Lưu checkpoint** (nếu có checkpointer), rồi sang super-step tiếp theo.

Chính bước 4 làm human-in-the-loop khả thi. Khi node `human_approval` gọi `interrupt(payload)`:

- LangGraph ném một exception đặc biệt, dừng super-step, rồi lưu checkpoint cùng `payload` vào task đang chờ.
- `graph.invoke` trả về. Process có thể thoát luôn, vì state đã nằm trong checkpointer (Memory/SQLite/Postgres).
- Sau đó `graph.invoke(Command(resume=decision), config)` với cùng `thread_id`: LangGraph nạp checkpoint và **chạy lại node `human_approval` từ đầu**. Lần này `interrupt()` không dừng nữa mà trả về `decision`.

Hệ quả cần nhớ: code **trước** `interrupt()` trong cùng node sẽ chạy 2 lần, nên phải là code thuần, không có side effect. Vì vậy bài này tách riêng node `human_approval` (chỉ hỏi) và node `tools` (mới thực sự ghi file / gửi mail).

## 3. Áp dụng trong bài này

```
START → agent ─┬─ (không còn tool_use) ──────────────→ END
               ├─ (chỉ tool an toàn) ──→ tools ──→ agent
               ├─ (có write_file/send_email) → human_approval → tools ──→ agent
               └─ (quá MAX_STEPS) ──→ max_steps → END
```

- `route_after_agent` là logic điều phối. Ở bài 3.3 nó nằm lẫn trong vòng `for`; ở đây nó là một hàm thuần, test riêng được.
- Chỉ những lượt có tool nguy hiểm mới dừng. Nếu dùng `compile(interrupt_before=["tools"])` thì graph dừng **trước mọi tool**, kể cả `calculate`, gây phiền. `interrupt()` động cho phép quyết định theo nội dung tool call.
- Người duyệt có 3 lựa chọn: duyệt, từ chối kèm lý do (lý do được gửi lại cho model dưới dạng `tool_result` có `is_error: true` để nó điều chỉnh), hoặc **sửa tham số** rồi duyệt.
- Thay `InMemorySaver` bằng `SqliteSaver`/`PostgresSaver` (`langgraph-checkpoint-sqlite`/`-postgres`) là có thể duyệt từ một process khác, ví dụ một web UI.

## 4. So sánh với bản from scratch (3.3)

| | 3.3 (tự viết) | 3.4 (LangGraph) |
|---|---|---|
| Vòng lặp | `for i in range(max_steps)` | cạnh `tools → agent` + `recursion_limit` |
| Gọi tool | parse text `Action:`/`Action Input:` | native `tool_use` block (JSON schema) |
| State | biến cục bộ trong `run()` | `AgentState` được checkpoint |
| Dừng chờ người | phải tự lưu/khôi phục `messages` | `interrupt()` / `Command(resume=)` |
| Quan sát | tự `print` | `stream_mode="updates"`, `get_state_history` |
| Chi phí | không phụ thuộc gì, dễ hiểu từng dòng | thêm dependency, phải hiểu mô hình super-step |

Kết luận: chain phù hợp với pipeline cố định (RAG một lượt: retrieve → prompt → LLM). Agent thì cần **chu trình + rẽ nhánh + state bền vững**, và đó chính là định nghĩa của một state machine. LangGraph chọn StateGraph vì nó mô hình hoá đúng bản chất bài toán.
