// Cấu hình hệ thống
const BACKEND_URL = window.location.hostname
  ? `${window.location.protocol}//${window.location.hostname}:8000`
  : "http://127.0.0.1:8000";

// Các phần tử giao diện (DOM)
const chatWidgetFab = document.getElementById("chat-widget-fab");
const chatWidgetWindow = document.getElementById("chat-widget-window");
const minimizeWidgetBtn = document.getElementById("minimize-widget-btn");
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const clearChatTopBtn = document.getElementById("clear-chat-top-btn");
const productModal = document.getElementById("product-modal");
const closeModalBtn = document.getElementById("close-modal-btn");
const modalProductTitle = document.getElementById("modal-product-title");
const modalProductImg = document.getElementById("modal-product-img");
const modalProductCompany = document.getElementById("modal-product-company");
const modalProductPrice = document.getElementById("modal-product-price");
const modalProductDesc = document.getElementById("modal-product-desc");

// Cache lưu sản phẩm để hiển thị chi tiết
let loadedProducts = {};
let chatHistory = [];

// Khởi tạo ứng dụng
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
});

// Đăng ký các sự kiện
function setupEventListeners() {
  // Đóng/Mở khung chat
  chatWidgetFab.addEventListener("click", () => {
    chatWidgetWindow.classList.add("active");
    chatWidgetFab.style.display = "none";
    userInput.focus();
  });

  // Hiệu ứng Vệt ma thuật (Mouse Trail)
  let lastTrailTime = 0;
  chatWidgetWindow.addEventListener("mousemove", (e) => {
    const now = Date.now();
    if (now - lastTrailTime < 60) return; // Giới hạn tốc độ sinh hiệu ứng
    lastTrailTime = now;

    const trail = document.createElement("div");
    trail.className = "mouse-trail";
    
    // Tọa độ chuột trong khung chat
    const rect = chatWidgetWindow.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    trail.style.left = `${x}px`;
    trail.style.top = `${y}px`;
    
    // Icon ngẫu nhiên
    const icons = ["💧", "✨", "🫧", "🍃", "🌿"];
    trail.textContent = icons[Math.floor(Math.random() * icons.length)];
    
    chatWidgetWindow.appendChild(trail);
    
    // Xóa đi sau khi kết thúc animation (1s)
    setTimeout(() => trail.remove(), 1000);
  });

  minimizeWidgetBtn.addEventListener("click", () => {
    chatWidgetWindow.classList.remove("active");
    chatWidgetFab.style.display = "flex";
  });

  chatWidgetFab.querySelector(".fab-icon-close").addEventListener("click", (e) => {
    e.stopPropagation();
    chatWidgetWindow.classList.remove("active");
    chatWidgetFab.style.display = "flex";
  });

  // Xử lý form gửi tin nhắn
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleUserMessageSend();
  });

  setupWelcomeSuggestions();

  clearChatTopBtn.addEventListener("click", resetChat);

  closeModalBtn.addEventListener("click", () => {
    productModal.classList.remove("active");
  });

  productModal.addEventListener("click", (e) => {
    if (e.target === productModal) {
      productModal.classList.remove("active");
    }
  });
}

// Setup suggestions in default welcome chat bubble
function setupWelcomeSuggestions() {
  document.querySelectorAll(".welcome-suggest-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const query = btn.getAttribute("data-query");
      sendQuickQuery(query);
    });
  });
}

// Send quick suggestions directly
function sendQuickQuery(query) {
  userInput.value = query;
  handleUserMessageSend();
}

// Xóa lịch sử chat
function resetChat() {
  chatHistory = []; // Reset history memory
  chatMessages.innerHTML = `
        <div class="message system-msg">
            <div class="msg-avatar" style="background: none; border: none; padding: 0;">
                <img src="wood_water_robot.png" style="width: 36px; height: 36px; border-radius: 50%; object-fit: cover; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            </div>
            <div class="msg-bubble">
                <p>Hệ thống đã được làm mới. Tôi sẵn sàng hỗ trợ bạn tra cứu và tư vấn thông tin sản phẩm thiết bị công nghiệp mới.</p>
                <p>Bạn cần tôi hỗ trợ thiết bị nào?</p>
                <div class="welcome-suggestions">
                    <button class="welcome-suggest-btn" data-query="Có bao nhiêu sản phẩm của hãng Balluff?">Thống kê sản phẩm Balluff</button>
                    <button class="welcome-suggest-btn" data-query="Cho tôi xem thông số của Bộ điều khiển nhiệt độ OMRON E5CC-QX3ASM-001">Thông số E5CC-QX3ASM-001</button>
                </div>
            </div>
        </div>
    `;
  setupWelcomeSuggestions();
  scrollToBottom();
}

// Hiển thị thông tin hệ thống
function showSystemInfo() {
  alert(
    "Trợ lý Ảo Đại Dương Automation - Version 1.0 (RAG)\n\n" +
      "- Backend: Python FastAPI\n" +
      "- Search Engine: Hybrid Search (Vector + BM25 + RRF + Cross-Encoder Reranker)\n" +
      "- Embeddings: intfloat/multilingual-e5-small (384 dimensions)\n" +
      "- Reranker: BAAI/bge-reranker-v2-m3\n" +
      "- LLM: Gemini 2.5 Flash\n" +
      "- Database: 5,056 Industrial Equipment products"
  );
}

// Xử lý gửi tin nhắn lên server
async function handleUserMessageSend() {
  const text = userInput.value.trim();
  if (!text) return;

  // Khóa ô nhập liệu khi đang xử lý
  setLoadingState(true);

  // Hiển thị tin nhắn user
  addMessageToUI("user", text);
  userInput.value = "";
  scrollToBottom();

  // Lưu vào lịch sử
  chatHistory.push({ role: "user", content: text });

  // Giữ lịch sử ngắn gọn (tối đa 10 tin)
  if (chatHistory.length > 10) {
    chatHistory = chatHistory.slice(-10);
  }

  // Thêm khung tin nhắn bot để stream
  const botMessageId = addBotMessageStreamingPlaceholder();
  scrollToBottom();

  try {
    const response = await fetch(`${BACKEND_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: text,
        history: chatHistory,
      }),
    });

    if (response.ok) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullResponse = "";
      let sources = [];

      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Giữ lại phần tử cuối cùng vì nó có thể là một dòng chưa hoàn chỉnh
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data.type === 'sources') {
              sources = data.sources;
              finalizeBotMessageStreaming(botMessageId, fullResponse, sources);
            } else if (data.type === 'chunk') {
              fullResponse += data.content;
              updateBotMessageStreaming(botMessageId, fullResponse);
              // Nhường luồng cho trình duyệt vẽ giao diện để tránh bị gom cục (batching)
              await new Promise(resolve => setTimeout(resolve, 10));
            } else if (data.type === 'error') {
              fullResponse += "\n\n⚠️ " + data.content;
              updateBotMessageStreaming(botMessageId, fullResponse);
            }
          } catch(e) {
            console.error("Parse JSON stream error:", e, line);
          }
        }
      }

      // Xử lý phần text còn sót trong buffer
      if (buffer.trim()) {
        try {
          const data = JSON.parse(buffer);
          if (data.type === 'chunk') {
            fullResponse += data.content;
            updateBotMessageStreaming(botMessageId, fullResponse);
          }
        } catch(e) {}
      }

      finalizeBotMessageStreaming(botMessageId, fullResponse, sources);

      chatHistory.push({ role: "assistant", content: fullResponse });
      if (chatHistory.length > 10) {
        chatHistory = chatHistory.slice(-10);
      }
    } else {
      const errData = await response.json();
      const errorMsg = `⚠️ Đã xảy ra lỗi từ server: ${errData.detail || "Không rõ nguyên nhân."}`;
      finalizeBotMessageStreaming(botMessageId, errorMsg, []);
    }
  } catch (error) {
    const errorMsg = `⚠️ Không thể kết nối tới server backend tại ${BACKEND_URL}. Vui lòng kiểm tra xem backend Python đã chạy chưa.`;
    finalizeBotMessageStreaming(botMessageId, errorMsg, []);
  } finally {
    // Mở khóa ô nhập liệu
    setLoadingState(false);
  }

  scrollToBottom();
}

// Hàm cập nhật trạng thái UI (khóa/mở)
function setLoadingState(isLoading) {
  userInput.disabled = isLoading;
  sendBtn.disabled = isLoading;
  
  if (isLoading) {
    userInput.placeholder = "Đang xử lý...";
    sendBtn.style.opacity = "0.5";
    sendBtn.style.pointerEvents = "none";
    
    // Khóa luôn các nút gợi ý
    document.querySelectorAll(".welcome-suggest-btn").forEach(btn => {
        btn.disabled = true;
        btn.style.opacity = "0.5";
        btn.style.pointerEvents = "none";
    });
  } else {
    userInput.placeholder = "Nhập câu hỏi tại đây...";
    sendBtn.style.opacity = "1";
    sendBtn.style.pointerEvents = "auto";
    userInput.focus();
    
    // Mở khóa các nút gợi ý
    document.querySelectorAll(".welcome-suggest-btn").forEach(btn => {
        btn.disabled = false;
        btn.style.opacity = "1";
        btn.style.pointerEvents = "auto";
    });
  }
}

// Hàm hiển thị tin nhắn lên giao diện
function addMessageToUI(sender, text, sources = []) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${sender === "user" ? "user-msg" : "system-msg"}`;

  const avatarHTML =
    sender === "user"
      ? `<div class="msg-avatar"><i class="fa-solid fa-user"></i></div>`
      : `<div class="msg-avatar" style="background: none; border: none; padding: 0;"><img src="wood_water_robot.png" style="width: 36px; height: 36px; border-radius: 50%; object-fit: cover; box-shadow: 0 2px 5px rgba(0,0,0,0.1);"></div>`;

  let bubbleContent = "";
  if (sender === "user") {
    bubbleContent = `<p class="user-message-text">${escapeHTML(text)}</p>`;
  } else {
    // Parse Markdown
    bubbleContent = renderMarkdownSafe(text);
  }

  // Hiển thị danh sách sản phẩm gợi ý
  let sourcesHTML = "";
  if (sources && sources.length > 0) {
    sourcesHTML = `
            <div class="sources-container">
                <div class="sources-title">
                    <i class="fa-solid fa-layer-group"></i> Thiết bị phù hợp tìm được:
                </div>
                <div class="products-grid">
        `;

    sources.forEach((prod) => {
      // Lưu vào cache
      loadedProducts[prod.id] = prod;

      const hasImage = prod.image && prod.image.trim() !== "";
      const productId = String(prod.id);
      const safeImage = escapeHTML(safeURL(prod.image || ""));
      const safeDevice = escapeHTML(prod.device || "");
      const safeBrand = escapeHTML(prod.brand || prod.company || "");
      const safeCompany = escapeHTML(prod.company || "");
      const safePrice = escapeHTML(prod.price_formatted || "");
      const imgHTML = hasImage
        ? `<img class="product-img" src="${safeImage}" alt="${safeDevice}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                   <div class="product-img-missing" style="display:none;"><i class="fa-solid fa-image"></i></div>`
        : `<div class="product-img-missing"><i class="fa-solid fa-image"></i></div>`;

      sourcesHTML += `
                <div class="product-card">
                    <div class="product-img-wrapper">
                        ${imgHTML}
                    </div>
                    <div class="product-brand">${safeBrand}</div>
                    <div class="product-name" title="${safeDevice}">${safeDevice}</div>
                    <div class="product-footer">
                        <div class="product-price">${safePrice}</div>
                        <button class="view-details-btn" data-product-id="${escapeHTML(productId)}">Chi tiết</button>
                    </div>
                </div>
            `;
    });

    sourcesHTML += `
                </div>
            </div>
        `;
  }

  msgDiv.innerHTML = `
        ${avatarHTML}
        <div class="msg-bubble">
            <div class="msg-text-content">${bubbleContent}</div>
            ${sourcesHTML}
        </div>
    `;

  chatMessages.appendChild(msgDiv);

  msgDiv.querySelectorAll(".view-details-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      openProductModal(btn.dataset.productId);
    });
  });
}

// Add typing loading indicator
function addTypingIndicator() {
  const indicatorId = "typing-" + Date.now();
  const msgDiv = document.createElement("div");
  msgDiv.className = "message system-msg";
  msgDiv.id = indicatorId;

  msgDiv.innerHTML = `
        <div class="msg-avatar" style="background: none; border: none; padding: 0;"><img src="wood_water_robot.png" style="width: 36px; height: 36px; border-radius: 50%; object-fit: cover; box-shadow: 0 2px 5px rgba(0,0,0,0.1);"></div>
        <div class="msg-bubble">
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
    `;

  chatMessages.appendChild(msgDiv);
  return indicatorId;
}

// Remove loading indicator
function removeTypingIndicator(id) {
  const indicator = document.getElementById(id);
  if (indicator) {
    indicator.remove();
  }
}

// Tạo bot message placeholder cho streaming
function addBotMessageStreamingPlaceholder() {
  const msgId = "bot-stream-" + Date.now();
  const msgDiv = document.createElement("div");
  msgDiv.className = "message system-msg";
  msgDiv.id = msgId;

  msgDiv.innerHTML = `
    <div class="msg-avatar" style="background: none; border: none; padding: 0;"><img src="wood_water_robot.png" style="width: 36px; height: 36px; border-radius: 50%; object-fit: cover; box-shadow: 0 2px 5px rgba(0,0,0,0.1);"></div>
    <div class="msg-bubble">
      <div class="msg-text-content" id="content-${msgId}">
        <div style="display: inline-block; width: 8px; height: 8px; background: #999; border-radius: 50%; animation: pulse 1.5s ease-in-out infinite;"></div>
      </div>
    </div>
  `;

  chatMessages.appendChild(msgDiv);

  // Add pulse animation CSS nếu chưa có
  if (!document.getElementById("streaming-style")) {
    const style = document.createElement("style");
    style.id = "streaming-style";
    style.textContent = `
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
      }
    `;
    document.head.appendChild(style);
  }

  return msgId;
}


// Hoàn thành bot message + thêm sources
function finalizeBotMessageStreaming(msgId, content, sources = []) {
  const msgDiv = document.getElementById(msgId);
  if (!msgDiv) return;

  // Render markdown content
  const bubbleContent = renderMarkdownSafe(content);

  // Compile sources cards
  let sourcesHTML = "";
  if (sources && sources.length > 0) {
    sourcesHTML = `
      <div class="sources-container">
        <div class="sources-title">
          <i class="fa-solid fa-layer-group"></i> Thiết bị phù hợp tìm được:
        </div>
        <div class="products-grid">
    `;

    sources.forEach((prod) => {
      loadedProducts[prod.id] = prod;

      const hasImage = prod.image && prod.image.trim() !== "";
      const productId = String(prod.id);
      const safeImage = escapeHTML(safeURL(prod.image || ""));
      const safeDevice = escapeHTML(prod.device || "");
      const safeBrand = escapeHTML(prod.brand || prod.company || "");
      const safeCompany = escapeHTML(prod.company || "");
      const safePrice = escapeHTML(prod.price_formatted || "");
      const imgHTML = hasImage
        ? `<img class="product-img" src="${safeImage}" alt="${safeDevice}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
           <div class="product-img-missing" style="display:none;"><i class="fa-solid fa-image"></i></div>`
        : `<div class="product-img-missing"><i class="fa-solid fa-image"></i></div>`;

      sourcesHTML += `
        <div class="product-card">
          <div class="product-img-wrapper">
            ${imgHTML}
          </div>
          <div class="product-brand">${safeBrand}</div>
          <div class="product-name" title="${safeDevice}">${safeDevice}</div>
          <div class="product-footer">
            <div class="product-price">${safePrice}</div>
            <button class="view-details-btn" data-product-id="${escapeHTML(productId)}">Chi tiết</button>
          </div>
        </div>
      `;
    });

    sourcesHTML += `
        </div>
      </div>
    `;
  }

  // Update message HTML
  msgDiv.innerHTML = `
    <div class="msg-avatar" style="background: none; border: none; padding: 0;"><img src="wood_water_robot.png" style="width: 36px; height: 36px; border-radius: 50%; object-fit: cover; box-shadow: 0 2px 5px rgba(0,0,0,0.1);"></div>
    <div class="msg-bubble">
      <div class="msg-text-content">${bubbleContent}</div>
      ${sourcesHTML}
    </div>
  `;

  // Attach event listeners to view-details buttons
  msgDiv.querySelectorAll(".view-details-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      openProductModal(btn.dataset.productId);
    });
  });

  scrollToBottom();
}

function updateBotMessageStreaming(msgId, content) {
  const msgDiv = document.getElementById(msgId);
  if (!msgDiv) return;
  const contentDiv = msgDiv.querySelector('.msg-text-content');
  if (contentDiv) {
    contentDiv.innerHTML = renderMarkdownSafe(content);
  }
  scrollToBottom();
}

// Preprocess raw description string to add bullet points and line breaks for better readability
function preprocessDescription(desc) {
  if (!desc) return "";

  // 0. Làm phẳng chuỗi, xóa bỏ các dấu xuống dòng vô duyên (ví dụ bị cắt giữa "Nhiệt\nđộ:")
  let formatted = desc.replace(/[\r\n]+/g, " ");

  // 1. Thay thế các ký tự | bằng dấu xuống dòng và đầu dòng gạch ngang markdown
  formatted = formatted.replace(/\s*\|\s*/g, "\n- ");

  // Danh sách chữ cái tiếng Việt (không bao gồm số) để làm ranh giới từ
  const viChars =
    "a-zA-ZàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ";
  const uppercaseViChars =
    "A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ";

  // Danh sách từ khóa viết hoa hoàn toàn
  const uppercaseKeywords = [
    "TÊN SẢN PHẨM",
    "DÒNG SẢN PHẨM",
    "DÒNG",
    "NGUỒN CẤP",
    "TẦN SỐ",
    "PHẠM VI ĐIỆN ÁP HOẠT ĐỘNG",
    "CÔNG SUẤT TIÊU THỤ",
    "PHƯƠNG PHÁP ĐIỀU KHIỂN",
    "PHƯƠNG PHÁP CÀI ĐẶT",
    "PHƯƠNG PHÁP HIỂN THỊ",
    "LOẠI ĐẦU RA",
    "ĐẦU RA PHỤ TRỢ",
    "ĐẦU VÀO CẢM BIẾN NHIỆT ĐỘ",
    "ĐẦU VÀO CẢM BIẾN",
    "ĐẦU VÀO ANALOG",
    "THỜI GIAN TÍCH PHÂN",
    "THỜI GIAN PHÁI SINH",
    "THỜI GIAN KIỂM SOÁT",
    "PHẠM VI CÀI ĐẶT BÁO THỨC",
    "THỜI GIAN LẤY MẪU ĐẦU VÀO",
    "KÍCH THƯỚC",
    "ĐIỆN TRỞ CÁCH ĐIỆN",
    "ĐỘ BỀN ĐIỆN MÔI",
    "NHIỆT ĐỘ MÔI TRƯỜNG",
    "ĐỘ ẨM MÔI TRƯỜNG",
    "TRỌNG LƯỢNG",
    "THÔNG SỐ KỸ THUẬT",
    "THÔNG SỐ",
    "ỨNG DỤNG",
    "ĐẶC ĐIỂM",
    "TÍNH NĂNG",
    "BẢO HÀNH",
    "CHẤT LIỆU",
    "MÀU SẮC",
    "XUẤT XỨ",
    "HÃNG SẢN XUẤT",
    "CÔNG TY",
    // Bổ sung các từ khóa viết hoa khác của cảm biến quang
    "MÃ HÀNG",
    "PHƯƠNG PHÁP CẢM BIẾN",
    "NGÕ RA",
    "KHOẢNG CÁCH CẢM BIẾN",
    "CHẾ ĐỘ HOẠT ĐỘNG",
    "DÒNG TIÊU THỤ",
    "THỜI GIAN ĐÁP ỨNG",
    "THỜI GIAN ĐẶT LẠI NGUỒN",
    "KIỂM SOÁT ĐẦU RA",
    "NGUỒN SÁNG",
    "PHƯƠNG THỨC KẾT NỐI",
    "PHƯƠNG PHÁP KẾT NỐI",
    "ĐẦU RA",
    "ĐẦU VÀO",
    "THỜI GIAN ĐẶT LẠI",
  ];

  // Thêm các biến thể kèm tham số như (I), (D) nếu có
  const normalizedKeywords = uppercaseKeywords.map((kw) => {
    if (kw === "THỜI GIAN TÍCH PHÂN")
      return "THỜI GIAN TÍCH PHÂN(?:\\s*\\(I\\))?";
    if (kw === "THỜI GIAN PHÁI SINH")
      return "THỜI GIAN PHÁI SINH(?:\\s*\\(D\\))?";
    return kw;
  });

  // Sắp xếp các từ khóa dài lên trước để match ưu tiên
  normalizedKeywords.sort((a, b) => b.length - a.length);

  // Danh sách từ khóa viết thường/viết hoa thông thường đi kèm dấu hai chấm
  const generalKeywords = [
    "Tên sản phẩm",
    "Dòng sản phẩm",
    "Dòng",
    "Nguồn cấp",
    "Tần số",
    "Phạm vi điện áp hoạt động",
    "Công suất tiêu thụ",
    "Phương pháp điều khiển",
    "Phương pháp cài đặt",
    "Phương pháp hiển thị",
    "Loại đầu ra",
    "Đầu ra phụ trợ",
    "Đầu vào cảm biến nhiệt độ",
    "Đầu vào cảm biến",
    "Đầu vào analog",
    "Thời gian tích phân",
    "Thời gian phái sinh",
    "Thời gian kiểm soát",
    "Phạm vi cài đặt báo thức",
    "Thời gian lấy mẫu đầu vào",
    "Kích thước",
    "Điện trở cách điện",
    "Độ bền điện môm",
    "Độ bền điện mối",
    "Độ bền điện môi",
    "Nhiệt độ môi trường",
    "Độ ẩm môi trường",
    "Trọng lượng",
    "Thông số kỹ thuật",
    "Thông số",
    "Ứng dụng",
    "Đặc điểm",
    "Tính năng",
    "Bảo hành",
    "Chất liệu",
    "Màu sắc",
    "Xuất xứ",
    "Hãng sản xuất",
    "Công ty",
    // Bổ sung các từ khóa viết thường khác của cảm biến quang
    "Mã hàng",
    "Phương pháp cảm biến",
    "Ngõ ra",
    "Khoảng cách cảm biến",
    "Chế độ hoạt động",
    "Dòng tiêu thụ",
    "Thời gian đáp ứng",
    "Thời gian đặt lại nguồn",
    "Kiểm soát đầu ra",
    "Nguồn sáng",
    "Phương thức kết nối",
    "Phương pháp kết nối",
    "Đầu ra",
    "Đầu vào",
    "Thời gian đặt lại",
  ];
  generalKeywords.sort((a, b) => b.length - a.length);

  // Tạo pattern cho từ khóa viết hoa không có dấu hai chấm 
  // Dùng lookbehind/lookahead để tránh cắt ngang từ, tránh bắt bên trong ** đã format
  const uppercasePattern =
    "(?<![" + viChars + "0-9*])(?:" + normalizedKeywords.join("|") + ")(?![" + viChars + "])\\s*:?[ \\t]*";
    
  const generalPattern =
    "(?<![" + viChars + "0-9*])(?:" + generalKeywords.join("|") + ")(?![" + viChars + "])\\s*:[ \\t]*";

  // BƯỚC 1: Xử lý các từ khóa cố định (chính xác tuyệt đối)
  const specificRegex = new RegExp("(" + generalPattern + "|" + uppercasePattern + ")", "g");
  formatted = formatted.replace(specificRegex, (match, p1, offset) => {
    let cleanMatch = match.trim();
    let formattedMatch = cleanMatch;
    if (cleanMatch.endsWith(":")) {
      formattedMatch = `**${cleanMatch.slice(0, -1).trim()}**: `;
    } else {
      formattedMatch = `**${cleanMatch}**: `;
    }
    return (offset === 0 ? "- " : "\n- ") + formattedMatch;
  });

  // BƯỚC 2: Xử lý các Key động (Flexible Keys)
  // Quy tắc chuẩn Tiếng Việt: CHỈ ký tự ĐẦU TIÊN được phép viết Hoa, toàn bộ phần còn lại (tối đa 5 từ) BẮT BUỘC viết thường, không chứa số.
  // Điều này ngăn chặn việc bắt nhầm các từ tiếng Anh/Thương hiệu dính liền phía trước (ví dụ "CN Pattlite Kích thước sản phẩm:" sẽ chỉ bắt "Kích thước sản phẩm:")
  const lowerViChars = "a-zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ";
  const colonPattern =
    "(?<![" + viChars + "0-9*])(?:[" + uppercaseViChars + "][" + lowerViChars + "]{1,}(?:[ \\t]+[" + lowerViChars + "]{1,}){0,4}(?:\\s*\\([^)]+\\))?)[ \\t]*:[ \\t]*";

  const colonRegex = new RegExp("(" + colonPattern + ")", "g");
  formatted = formatted.replace(colonRegex, (match, p1, offset) => {
    let cleanMatch = match.trim();
    let formattedMatch = `**${cleanMatch.slice(0, -1).trim()}**: `;
    return (offset === 0 && !formatted.includes("\n")) ? "- " + formattedMatch : "\n- " + formattedMatch;
  });

  // 3. Đảm bảo toàn bộ chuỗi được gạch đầu dòng đồng bộ nếu chưa có
  if (!formatted.trim().startsWith("-") && !formatted.trim().startsWith("*")) {
    formatted = "- " + formatted.trim();
  }

  // Dọn dẹp khoảng trắng thừa và sửa lỗi trùng lặp dấu gạch đầu dòng (ví dụ: - - **TÊN SẢN PHẨM**)
  formatted = formatted
    .split("\n")
    .map((line) => {
      let cleanLine = line.trim();
      // Xóa dấu gạch đầu dòng lặp lại dư thừa
      cleanLine = cleanLine.replace(/^[-*]\s+[-*]\s+/, "- ");
      // Nếu dòng không rỗng và không có dấu gạch đầu dòng, tự động thêm
      if (
        cleanLine.length > 0 &&
        !cleanLine.startsWith("-") &&
        !cleanLine.startsWith("*")
      ) {
        cleanLine = "- " + cleanLine;
      }
      return cleanLine;
    })
    .filter((line) => {
      const trimmed = line.trim();
      return trimmed.length > 0 && trimmed !== "-" && trimmed !== "*";
    })
    .join("\n");

  return formatted;
}

// Open modal for details view
window.openProductModal = function (productId) {
  const prod = loadedProducts[productId];
  if (!prod) return;

  modalProductTitle.textContent = prod.device;
  // Hiển thị brand (hãng sản xuất) và company (nhà phân phối) riêng biệt
  const brandName = prod.brand || prod.company || "";
  const distributorName = prod.company || "";
  if (brandName !== distributorName && distributorName) {
    modalProductCompany.innerHTML = `<strong>${escapeHTML(brandName)}</strong> <span style="opacity:0.7;font-size:0.85em">(PP: ${escapeHTML(distributorName)})</span>`;
  } else {
    modalProductCompany.textContent = brandName;
  }
  modalProductPrice.textContent = prod.price_formatted;

  const imageURL = safeURL(prod.image || "");
  if (imageURL) {
    modalProductImg.src = imageURL;
    modalProductImg.style.display = "block";
  } else {
    modalProductImg.src = "";
    modalProductImg.style.display = "none";
  }

  // Hiển thị nội dung mô tả sản phẩm (đã được tự động tách dòng và in đậm)
  let markdownText = preprocessDescription(prod.description || "");
  
  // Fallback: Nếu không có description nhưng có specs thì mới lấy specs ra hiển thị
  if (!markdownText && prod.specs && Object.keys(prod.specs).length > 0) {
    for (const [key, value] of Object.entries(prod.specs)) {
      markdownText += `- **${key.trim()}**: ${value}\n`;
    }
  }

  modalProductDesc.innerHTML = renderMarkdownSafe(markdownText);

  productModal.classList.add("active");
};

// Utils
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHTML(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderMarkdownSafe(text) {
  // Cấu hình marked.js để hỗ trợ xuống dòng tự động (tốt cho giao diện chat)
  marked.setOptions({
    breaks: true,
    gfm: true
  });
  
  const rawHTML = marked.parse(String(text || ""));
  return sanitizeHTML(rawHTML);
}

function sanitizeHTML(html) {
  const template = document.createElement("template");
  template.innerHTML = html;

  template.content
    .querySelectorAll("script, style, iframe, object, embed")
    .forEach((node) => node.remove());
  template.content.querySelectorAll("*").forEach((node) => {
    [...node.attributes].forEach((attr) => {
      const name = attr.name.toLowerCase();
      const value = attr.value.trim().toLowerCase();
      if (name.startsWith("on") || value.startsWith("javascript:")) {
        node.removeAttribute(attr.name);
      }
    });
  });

  return template.innerHTML;
}

function safeURL(url) {
  const value = String(url || "").trim();
  if (!value) return "";

  try {
    const parsed = new URL(value, window.location.href);
    if (["http:", "https:"].includes(parsed.protocol)) {
      return value;
    }
  } catch (error) {
    return "";
  }

  return "";
}
