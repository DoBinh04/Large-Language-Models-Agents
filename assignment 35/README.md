# Assignment 3.5: MCP server tự viết (JSON-RPC over stdio / HTTP)

📄 **Báo cáo kết quả thực nghiệm: [REPORT.md](REPORT.md)**

Server **không dùng SDK `mcp`**. Toàn bộ protocol được viết tay bằng thư viện chuẩn Python để thấy rõ từng message. Ghi chú chi tiết về wire protocol nằm trong [PROTOCOL.md](PROTOCOL.md).

## Cấu trúc

```
assignment 35/
├── mcp_server.py        # server: JSON-RPC dispatcher + stdio; HTTP demo tối giản
├── mcp_client.py        # client stdio tự viết (spawn server, request/notify, ghép response theo id)
├── mcp_client_demo.py   # in từng message JSON-RPC đi/về (không cần API key)
├── mcp_agent.py         # agent Claude dùng tool của server qua MCP
├── claude_desktop_config.example.json
├── sandbox/             # --root mặc định: tool chỉ truy cập được bên trong
└── tests/test_server.py
```

## 3 tools

| Tool | Tham số | Ghi chú |
|---|---|---|
| `read_file` | `path`, `max_bytes?` | chỉ file UTF-8, tối đa 200 KB |
| `list_directory` | `path?` | thư mục trước, kèm kích thước file |
| `execute_python` | `code`, `timeout?` | process riêng `python -I`, cwd = root, timeout ≤ 60s, **không** truyền biến môi trường chứa KEY/TOKEN/SECRET |

Mọi path đều được `resolve()` và kiểm tra nằm trong `--root`, nên `../` hay đường dẫn tuyệt đối ra ngoài đều bị chặn.
⚠️ `execute_python` **không phải sandbox bảo mật**: code vẫn chạy với quyền của user hiện tại. Dùng `--no-exec` để tắt, hoặc chạy server trong container.

## Chạy

```bat
cd "C:\phase3\assignment 35"
python -m tests.test_server          :: test dispatcher
python mcp_client_demo.py            :: xem wire protocol (không cần API key)
python mcp_agent.py "Đọc README.txt rồi dùng Python đếm số từ"      :: cần ANTHROPIC_API_KEY (.env của assignment 33 cũng được)
python mcp_agent.py --wire --confirm-exec "Tạo file primes.txt chứa 100 số nguyên tố đầu tiên"
python mcp_server.py --http 8765     :: HTTP demo tối giản; xem PROTOCOL.md mục 4 để thử bằng PowerShell
```

## Kết nối Claude Desktop

1. Mở file config ngay trong Claude Desktop (nên dùng cách này để không sửa nhầm file):
   - Giao diện mới: **Settings → Desktop app → Extensions**, kéo xuống mục **Local MCP servers** → bấm **Edit config**. (Nút *Advanced settings* ở mục Extensions là cho extension `.mcpb/.dxt`, không cần dùng.)
   - Bản cũ: **Settings → Developer → Edit Config**.

   Nếu muốn mở file trực tiếp: bản cài từ trang chủ lưu ở `%APPDATA%\Claude\claude_desktop_config.json`; bản cài từ Microsoft Store (MSIX) lưu ở `%LOCALAPPDATA%\Packages\Claude_<id>\LocalCache\Roaming\Claude\claude_desktop_config.json`. Log cũng nằm trong thư mục `logs\` cạnh file đó.
2. Chép khối `mcpServers` từ `claude_desktop_config.example.json`. Sửa `command` thành đường dẫn `python.exe` của bạn (`where python`). Dùng đường dẫn **tuyệt đối** cho mọi thứ.
3. Thoát hẳn Claude Desktop (cả icon ở khay hệ thống) rồi mở lại. Tool sẽ xuất hiện trong menu công cụ của khung chat.
4. Hỏi: *"List the files in the sandbox and run Python to compute 2**100"*. Claude Desktop hỏi quyền trước mỗi lần gọi tool.
5. Debug:
   - `wire.log`: mọi message Claude Desktop gửi/nhận (nhờ `--wire-log`).
   - `logs\mcp-server-phase3-fs-python.log` (cùng thư mục cha với file config ở bước 1): stderr của server.

Nên xem trong `wire.log` thứ tự `initialize` → `notifications/initialized` → `tools/list` mà Claude Desktop gửi, và `protocolVersion` mà nó đề nghị.

## Phạm vi tương thích

Phần đã được chứng minh bằng artifact trong repository là **MCP qua stdio**: `wire.log` và ảnh trong `screenshots/` cho thấy Claude Desktop đã khám phá và gọi thành công cả ba tool. Đây là phần đáp ứng trực tiếp yêu cầu đề bài.

Tùy chọn `--http` hữu ích để học cấu trúc Streamable HTTP (endpoint, JSON-RPC body, session ID, `202` cho notification), nhưng chưa được chạy trong đợt thực nghiệm và không nên mô tả là một implementation Streamable HTTP đầy đủ. Cụ thể, nó chưa kiểm tra `Accept`, `Content-Type` và `MCP-Protocol-Version`, chưa có SSE hay xác thực. Vì vậy không dùng endpoint này cho môi trường thật.

## Những điểm dễ sai khi tự viết server (đã xử lý)

- In bất cứ thứ gì ra **stdout** ngoài JSON-RPC sẽ làm client parse lỗi. Log luôn ra stderr.
- Trả lời một **notification**: sai spec, client có thể báo lỗi "unknown id".
- Trên Windows, stdout ở text mode đổi `\n` thành `\r\n`. Server mở lại stdout với `newline="\n"`.
- Lỗi tool phải trả bằng `isError: true` trong `result`, không phải JSON-RPC `error`. Có vậy LLM mới thấy lỗi và tự sửa.
- `initialize`: nếu hỗ trợ version client đề nghị thì trả lại **đúng version đó**, nếu không thì trả version mới nhất của server. Client sẽ tự ngắt kết nối nếu không chấp nhận được.
