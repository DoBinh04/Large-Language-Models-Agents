# Vận chuyển, thanh toán và huỷ đơn – TechViet Store

## 1. Phương thức vận chuyển

| Gói | Thời gian | Phí |
|---|---|---|
| Tiêu chuẩn | 3–5 ngày làm việc | 30.000đ; **miễn phí** đơn từ 500.000đ |
| Nhanh | 1–2 ngày làm việc (nội thành Hà Nội, TP.HCM giao trong 4 giờ) | 50.000đ |

- Đơn vị vận chuyển: GHN. Mã vận đơn dạng `GHN-xxxxxxxx`, tra cứu trên ứng dụng GHN.
- Không giao hàng vào Chủ nhật đối với gói Tiêu chuẩn.
- Khách được **đồng kiểm** (mở hộp kiểm tra ngoại quan) trước khi thanh toán COD, nhưng không được kích hoạt sản phẩm.

## 2. Trạng thái đơn hàng

- `processing`: đã xác nhận, đang chuẩn bị hàng (thường dưới 24 giờ).
- `shipping`: đã bàn giao cho GHN.
- `delivered`: đã giao thành công.
- `cancelled`: đã huỷ.
- `returned`: đã hoàn trả / hoàn tiền.

## 3. Thanh toán

- COD (thanh toán khi nhận hàng) cho đơn dưới 50 triệu đồng.
- Chuyển khoản ngân hàng, thẻ tín dụng/ghi nợ (Visa, Mastercard, JCB), ví điện tử.
- **Trả góp 0%** qua thẻ tín dụng hoặc công ty tài chính cho đơn từ 3 triệu đồng, kỳ hạn 6 hoặc 12 tháng, không phí chuyển đổi với kỳ hạn 6 tháng.
- Xuất hoá đơn VAT điện tử theo yêu cầu trong vòng 7 ngày kể từ ngày đặt hàng.

## 4. Huỷ đơn và đổi địa chỉ

- Đơn `processing` được huỷ miễn phí; tiền đã thanh toán hoàn trong 3–5 ngày làm việc.
- Đơn `shipping` không thể huỷ trên hệ thống; khách từ chối nhận hàng khi shipper giao, phí vận chuyển chiều về do TechViet chịu lần đầu.
- Đổi địa chỉ nhận hàng chỉ thực hiện được khi đơn còn ở trạng thái `processing`.

## 5. Giao hàng thất bại

GHN liên hệ giao lại tối đa 3 lần. Sau 3 lần không liên lạc được, đơn tự động chuyển về kho và bị huỷ.
