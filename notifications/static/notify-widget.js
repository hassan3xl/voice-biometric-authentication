/**
 * QStack Notification Widget SDK
 * Drop-in real-time notification client.
 * Handles Socket.IO connections, subscriptions, automatic badge updates, and toasts.
 */
(function () {
  // 1. Load Socket.IO client library dynamically if not present
  if (typeof io === "undefined") {
    const script = document.createElement("script");
    script.src = "https://cdn.socket.io/4.7.5/socket.io.min.js";
    script.async = true;
    script.onload = () => initQStackNotification();
    document.head.appendChild(script);
  } else {
    initQStackNotification();
  }

  function initQStackNotification() {
    // Find the script tag config attributes
    const currentScript =
      document.getElementById("qstack-notify-script") ||
      document.currentScript ||
      document.querySelector('script[src*="notify-widget.js"]');
    if (!currentScript) {
      console.warn("[QStack] Configuration script tag not found.");
      return;
    }

    const serverUrl = currentScript.getAttribute("data-server");
    const channel = currentScript.getAttribute("data-channel") || "default";
    const recipient = currentScript.getAttribute("data-recipient");
    const badgeSelector = currentScript.getAttribute("data-badge-selector");
    const listSelector = currentScript.getAttribute("data-list-selector");
    const enableToast = currentScript.getAttribute("data-toast") !== "false";

    // Establish Socket.IO connection
    const socket = io(serverUrl, {
      transports: ["websocket", "polling"],
    });

    socket.on("connect", () => {
      console.log(
        "[QStack] Connected to notification server. Socket ID:",
        socket.id,
      );
      socket.emit("subscribe", { channel: channel });
      console.log(`[QStack] Subscribed to room/channel: ${channel}`);
    });

    socket.on("notification", (data) => {
      // Filter by recipient if configured (case-insensitive and trimmed)
      const payload = data.payload || {};
      const payloadRecipient = (payload.recipient || "").trim().toLowerCase();
      const currentRecipient = (recipient || "").trim().toLowerCase();

      if (
        currentRecipient &&
        payloadRecipient &&
        payloadRecipient !== currentRecipient &&
        payloadRecipient !== "all"
      ) {
        console.log(
          `[QStack] Live notification filtered out (for ${payloadRecipient}, but this client is ${currentRecipient}).`
        );
        return;
      }

      console.log("[QStack] Live notification received:", data);

      // 1. Dispatch custom DOM event for external libraries (like Alpine.js or HTMX)
      const event = new CustomEvent("qstack:notification", {
        detail: data,
        bubbles: true,
        cancelable: true,
      });
      document.dispatchEvent(event);

      // 2. Automatically update notification badge element(s) if selector provided
      if (badgeSelector) {
        const badges = document.querySelectorAll(badgeSelector);
        badges.forEach((badge) => {
          let count = parseInt(badge.textContent || "0", 10);
          if (isNaN(count)) count = 0;
          badge.textContent = count + 1;
          badge.style.display = "flex"; // Ensure badge is visible
        });
      }

      // 3. Append to list/dropdown if selector provided
      if (listSelector) {
        const lists = document.querySelectorAll(listSelector);
        lists.forEach((list) => {
          // Remove "No new notifications" placeholder if it exists
          const placeholder =
            list.querySelector(".qstack-no-notifications") ||
            list.querySelector(".text-center");
          if (
            placeholder &&
            placeholder.textContent.toLowerCase().includes("no new")
          ) {
            placeholder.remove();
          }

          // Create new list item (customizable & beautiful)
          const item = document.createElement("div");
          item.className = "qstack-notification-item";
          item.style.padding = "12px 16px";
          item.style.borderBottom = "1px solid rgba(255,255,255,0.08)";
          item.style.transition = "all 0.2s ease";

          const titleEl = document.createElement("h4");
          titleEl.className = "text-sm font-semibold text-textLight";
          titleEl.style.margin = "0 0 2px 0";
          titleEl.style.color = "#f8fafc";
          titleEl.textContent = data.title;

          const bodyEl = document.createElement("p");
          bodyEl.className = "text-xs text-gray-400";
          bodyEl.style.margin = "0";
          bodyEl.style.color = "#9ca3af";
          bodyEl.textContent = data.body;

          const timeEl = document.createElement("span");
          timeEl.className = "text-[10px] text-gold/60";
          timeEl.style.marginTop = "4px";
          timeEl.style.display = "block";
          timeEl.style.color = "rgba(212, 175, 55, 0.6)";
          timeEl.textContent = new Date(
            data.created_at || Date.now(),
          ).toLocaleTimeString();

          item.appendChild(titleEl);
          item.appendChild(bodyEl);
          item.appendChild(timeEl);

          // Insert at top of list
          list.insertBefore(item, list.firstChild);
        });
      }

      // 4. Show real-time glassmorphic toast notification
      if (enableToast) {
        showPremiumToast(data);
      }
    });

    socket.on("disconnect", () => {
      console.log("[QStack] Disconnected from notification server.");
    });
  }

  function showPremiumToast(data) {
    // Create container if not exists
    let container = document.getElementById("qstack-toast-container");
    if (!container) {
      container = document.createElement("div");
      container.id = "qstack-toast-container";
      container.style.position = "fixed";
      container.style.bottom = "24px";
      container.style.right = "24px";
      container.style.zIndex = "99999";
      container.style.display = "flex";
      container.style.flexDirection = "column";
      container.style.gap = "12px";
      document.body.appendChild(container);
    }

    // Create Glassmorphism Toast
    const toast = document.createElement("div");
    toast.style.background = "rgba(30, 41, 59, 0.8)";
    toast.style.backdropFilter = "blur(12px)";
    toast.style.webkitBackdropFilter = "blur(12px)";
    toast.style.border = "1px solid rgba(255, 255, 255, 0.1)";
    toast.style.color = "#f8fafc";
    toast.style.padding = "16px 20px";
    toast.style.borderRadius = "12px";
    toast.style.boxShadow =
      "0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.4)";
    toast.style.width = "320px";
    toast.style.display = "flex";
    toast.style.flexDirection = "column";
    toast.style.gap = "4px";
    toast.style.transform = "translateY(24px)";
    toast.style.opacity = "0";
    toast.style.transition = "all 0.4s cubic-bezier(0.16, 1, 0.3, 1)";
    toast.style.fontFamily = "system-ui, -apple-system, sans-serif";

    // Toast Header
    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.justifyContent = "space-between";
    header.style.alignItems = "center";

    const title = document.createElement("strong");
    title.style.fontSize = "14px";
    title.style.fontWeight = "600";
    title.style.color = "#d4af37"; // Gold
    title.textContent = data.title;

    const closeBtn = document.createElement("button");
    closeBtn.innerHTML = "&times;";
    closeBtn.style.background = "none";
    closeBtn.style.border = "none";
    closeBtn.style.color = "#94a3b8";
    closeBtn.style.fontSize = "18px";
    closeBtn.style.cursor = "pointer";
    closeBtn.style.padding = "0";
    closeBtn.style.lineHeight = "1";
    closeBtn.onclick = () => {
      toast.style.transform = "translateX(120px)";
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 400);
    };

    header.appendChild(title);
    header.appendChild(closeBtn);

    // Toast Body
    const body = document.createElement("p");
    body.style.margin = "0";
    body.style.fontSize = "13px";
    body.style.color = "#e2e8f0";
    body.style.lineHeight = "1.4";
    body.textContent = data.body;

    toast.appendChild(header);
    toast.appendChild(body);
    container.appendChild(toast);

    // Animate in
    setTimeout(() => {
      toast.style.transform = "translateY(0)";
      toast.style.opacity = "1";
    }, 50);

    // Auto-dismiss
    setTimeout(() => {
      if (toast.parentNode) {
        toast.style.transform = "translateX(120px)";
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 400);
      }
    }, 6000);
  }
})();
