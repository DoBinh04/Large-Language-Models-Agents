# MCP wire protocol: ghi chú từ việc tự cài đặt

MCP = **JSON-RPC 2.0** + một tập method được quy ước (`initialize`, `tools/list`, `tools/call`, `resources/*`, `prompts/*`...) + một lớp **transport** (stdio hoặc Streamable HTTP).

## 1. Ba loại message JSON-RPC

| Loại | Có `id`? | Có `method`? | Ví dụ |
|---|---|---|---|
| Request | có | có | `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{...}}` |
| Response | có (trùng với request) | **không**; có `result` **hoặc** `error` | `{"jsonrpc":"2.0","id":3,"result":{...}}` |
| Notification | **không** | có | `{"jsonrpc":"2.0","method":"notifications/initialized"}` |

Bên nhận **không bao giờ** trả lời notification. Response được ghép với request **qua `id`**, không qua thứ tự. Cả hai phía đều có thể gửi request: server có thể gửi `sampling/createMessage` hay `elicitation/create` cho client, nếu client khai báo capability đó.

## 2. Phiên làm việc thực tế (lấy từ `wire.log` / `mcp_client_demo.py`)

```jsonc
// --> client: đề nghị protocol version + capability của client
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"phase3-custom-client","version":"1.0.0"}}}
// <-- server: version CHỐT (bằng version client nếu server hỗ trợ), capability của server
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"phase3-fs-python","version":"1.0.0"},"instructions":"File tools are sandboxed to ..."}}
// --> client: báo đã xong handshake (notification, không có id, server không trả lời)
{"jsonrpc":"2.0","method":"notifications/initialized"}

// --> khám phá tool
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
// <-- mỗi tool có name, description, inputSchema (JSON Schema)
{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"read_file","description":"...","inputSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}, ...]}}

// --> gọi tool
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"execute_python","arguments":{"code":"print(6*7)"}}}
// <-- kết quả là danh sách content block (text / image / audio / resource_link / embedded resource)
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"exit_code: 0\n--- stdout ---\n42\n"}],"isError":false}}
```

Hãy để ý: `\n` bên trong chuỗi được escape thành hai ký tự `\` `n`, nên mỗi message vẫn nằm gọn trên **một dòng** vật lý. Đó là toàn bộ "framing" của transport stdio.

## 3. Hai loại lỗi: phải phân biệt

| | Khi nào | Dạng |
|---|---|---|
| **Lỗi protocol** | method không tồn tại, tool không tồn tại, params sai schema, JSON hỏng | `"error": {"code": -32601 / -32602 / -32700, "message": ...}` |
| **Lỗi tool** | tool chạy nhưng thất bại: file không có, path ngoài root, code Python raise | `"result": {"content": [...], "isError": true}` |

Lỗi tool đi trong `result` vì nó dành cho **LLM đọc và tự sửa** (ví dụ đổi path). Lỗi protocol dành cho **code của client** (có bug). Client sẽ không đưa lỗi protocol cho model như một kết quả bình thường.

## 4. Transport

**stdio** (Claude Desktop dùng cách này):
- Client spawn server như subprocess. Mỗi message là một dòng UTF-8 kết thúc bằng `\n`, không chứa newline thật.
- **stdout chỉ dành cho protocol.** Một lệnh `print()` debug thôi cũng làm hỏng stream. Mọi log phải ra stderr (Claude Desktop ghi stderr vào `%APPDATA%\Claude\logs\mcp-server-<name>.log`).
- Kết thúc phiên: client đóng stdin, server đọc được EOF rồi thoát.

**Streamable HTTP** (`--http PORT`):
- Một endpoint duy nhất `POST /mcp`. Body là một message JSON-RPC.
- Request có response trả về: `200` với `Content-Type: application/json` (bản này), hoặc `text/event-stream` (SSE) nếu server muốn stream nhiều message.
- Chỉ có notification/response trả về: `202 Accepted`, không body.
- Server cấp `Mcp-Session-Id` trong header response của `initialize`. Client phải gửi lại header này ở mọi request sau, và cả `MCP-Protocol-Version`.
- `GET /mcp` để mở SSE stream cho message server chủ động gửi. Server này trả `405` vì không hỗ trợ. `DELETE /mcp` kết thúc session.
- Bảo mật: bind `127.0.0.1` và kiểm tra header `Origin` để chống DNS rebinding.

Thử bằng PowerShell:

```powershell
python mcp_server.py --http 8765          # cửa sổ 1

# cửa sổ 2
$h = @{ "Accept" = "application/json, text/event-stream" }
$init = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"ps","version":"1"}}}'
$r = Invoke-WebRequest -Uri http://127.0.0.1:8765/mcp -Method Post -Body $init -ContentType "application/json" -Headers $h
$r.Content
$h["Mcp-Session-Id"] = $r.Headers["Mcp-Session-Id"]
$h["MCP-Protocol-Version"] = "2025-11-25"
Invoke-WebRequest -Uri http://127.0.0.1:8765/mcp -Method Post -ContentType "application/json" -Headers $h -Body '{"jsonrpc":"2.0","method":"notifications/initialized"}' | Select-Object StatusCode   # 202
(Invoke-WebRequest -Uri http://127.0.0.1:8765/mcp -Method Post -ContentType "application/json" -Headers $h -Body '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_directory","arguments":{}}}').Content
```

## 5. So với native tool use của Claude API

| | Claude API tool use | MCP |
|---|---|---|
| Ai định nghĩa tool | code gọi API (gửi `tools=[...]` mỗi request) | server, client hỏi qua `tools/list` |
| Schema field | `input_schema` | `inputSchema` |
| Kết quả | block `tool_result` trong user message | `result.content[]` + `isError` |
| Ai chạy tool | code của bạn | process server (có thể ở máy khác qua HTTP) |

`mcp_agent.py` chính là "cây cầu" giữa hai bên: `tools/list` → `input_schema`, `tool_use` → `tools/call`, `result` → `tool_result`.
