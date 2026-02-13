const { useEffect, useRef, useState } = React;

const API_URL = "/chat";
const CHATS_API = "/chats";
const USER_ID_KEY = "user_id";
const WELCOME_MESSAGE =
    "Welcome to Mnemos. I keep track of what matters to you and carry context across conversations.";

async function parseApiResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
        return await response.json();
    }
    const text = await response.text();
    return { detail: text || `Request failed with status ${response.status}` };
}

function LoginScreen({ mode, setMode, onSubmit, isLoading, error }) {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");

    const submit = (e) => {
        e.preventDefault();
        onSubmit({ username, password });
    };

    return (
        <div className="min-h-screen bg-[#161A30] text-[#F0ECE5] flex items-center justify-center px-4">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(148,137,121,0.2),transparent_45%),radial-gradient(circle_at_85%_10%,rgba(223,208,184,0.08),transparent_35%)]" />
            <div className="relative w-full max-w-md bg-[#393E46]/92 border border-[#948979]/45 rounded-3xl p-7 shadow-[0_20px_80px_rgba(0,0,0,0.45)] backdrop-blur">
                <h1 className="text-2xl font-bold tracking-tight text-[#F0ECE5]">Mnemos</h1>
                <p className="text-[#948979] text-sm mt-1 mb-5">
                    {mode === "login" ? "Sign in to continue" : "Create your account"}
                </p>
                <div className="grid grid-cols-2 gap-2 mb-5 bg-[#161A30] border border-[#948979]/40 p-1 rounded-xl">
                    <button
                        type="button"
                        onClick={() => setMode("login")}
                        className={`py-2 rounded-lg text-sm font-semibold transition-colors ${
                            mode === "login" ? "bg-[#948979] text-[#161A30]" : "text-[#948979] hover:text-[#F0ECE5]"
                        }`}
                    >
                        Login
                    </button>
                    <button
                        type="button"
                        onClick={() => setMode("signup")}
                        className={`py-2 rounded-lg text-sm font-semibold transition-colors ${
                            mode === "signup" ? "bg-[#948979] text-[#161A30]" : "text-[#948979] hover:text-[#F0ECE5]"
                        }`}
                    >
                        Signup
                    </button>
                </div>
                <form onSubmit={submit} className="space-y-3">
                    <input
                        type="text"
                        required
                        placeholder="Username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        className="w-full bg-[#161A30] border border-[#948979]/45 text-[#F0ECE5] placeholder-[#948979] rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-[#948979]"
                    />
                    <input
                        type="password"
                        required
                        placeholder="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full bg-[#161A30] border border-[#948979]/45 text-[#F0ECE5] placeholder-[#948979] rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-[#948979]"
                    />
                    {error && (
                        <div className="text-sm text-red-200 bg-red-950/40 border border-red-900 rounded-xl px-3 py-2">
                            {error}
                        </div>
                    )}
                    <button
                        type="submit"
                        disabled={isLoading}
                        className="w-full bg-[#948979] hover:bg-[#F0ECE5] disabled:bg-[#393E46] disabled:text-[#948979] text-[#161A30] font-semibold py-3 rounded-xl transition-colors"
                    >
                        {isLoading ? "Please wait..." : mode === "login" ? "Login" : "Signup"}
                    </button>
                </form>
            </div>
        </div>
    );
}

function TypewriterText({ text, speed = 18, active = true }) {
    const [display, setDisplay] = useState(active ? "" : text);

    useEffect(() => {
        if (!active) {
            setDisplay(text);
            return;
        }

        let i = 0;
        const timer = setInterval(() => {
            i += 1;
            setDisplay(text.slice(0, i));
            if (i >= text.length) clearInterval(timer);
        }, speed);

        return () => clearInterval(timer);
    }, [text, speed, active]);

    const done = display.length >= text.length;
    return (
        <span>
            {display}
            <span className={`ml-0.5 inline-block text-[#948979] ${done ? "opacity-0" : "opacity-100 animate-pulse"}`}>|</span>
        </span>
    );
}

function App() {
    const [userId, setUserId] = useState(localStorage.getItem(USER_ID_KEY) || "");
    const [authMode, setAuthMode] = useState("login");
    const [authError, setAuthError] = useState("");
    const [authLoading, setAuthLoading] = useState(false);

    const [chats, setChats] = useState([]);
    const [currentChatId, setCurrentChatId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [isWaitingResponse, setIsWaitingResponse] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [showAnimatedWelcome, setShowAnimatedWelcome] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [deleteConfirmChatId, setDeleteConfirmChatId] = useState(null);
    const [copiedMessageKey, setCopiedMessageKey] = useState("");
    const messagesEndRef = useRef(null);
    const textareaRef = useRef(null);
    const copyResetTimerRef = useRef(null);

    const formatDateTime = (iso) => {
        try {
            return new Date(iso).toLocaleString([], {
                year: "numeric",
                month: "short",
                day: "2-digit",
                hour: "2-digit",
                minute: "2-digit",
            });
        } catch {
            return iso || "";
        }
    };
    const userInitial = ((userId || "").trim().charAt(0) || "U").toUpperCase();

    useEffect(() => {
        const behavior = isStreaming ? "auto" : "smooth";
        messagesEndRef.current?.scrollIntoView({ behavior });
    }, [messages, isLoading, isStreaming]);

    useEffect(() => {
        if (!userId) return;
        loadChats();
    }, [userId]);

    useEffect(() => {
        if (!userId) return;
        const welcomeKey = `welcome_seen_${userId}`;
        const seen = localStorage.getItem(welcomeKey);
        if (!seen) {
            setShowAnimatedWelcome(true);
            localStorage.setItem(welcomeKey, "1");
            return;
        }
        setShowAnimatedWelcome(false);
    }, [userId]);

    useEffect(() => {
        if (currentChatId) {
            loadChatMessages(currentChatId);
        } else {
            setMessages([]);
        }
    }, [currentChatId]);

    useEffect(() => {
        autoResizeTextarea();
    }, [inputMessage]);

    useEffect(() => {
        return () => {
            if (copyResetTimerRef.current) {
                clearTimeout(copyResetTimerRef.current);
            }
        };
    }, []);

    const getHeaders = (withJson = false, includeUser = false) => {
        const headers = {};
        if (withJson) headers["Content-Type"] = "application/json";
        if (includeUser && userId) headers["X-User-ID"] = userId;
        return headers;
    };

    const autoResizeTextarea = () => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = "auto";
        const maxHeight = 180;
        el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
        el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
    };

    const resetTextareaHeight = () => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = "48px";
        el.style.overflowY = "hidden";
    };

    const logout = () => {
        localStorage.removeItem(USER_ID_KEY);
        setUserId("");
        setChats([]);
        setCurrentChatId(null);
        setMessages([]);
        setAuthError("");
        setCopiedMessageKey("");
    };

    const copyMessageToClipboard = async (text, messageKey) => {
        const content = (text || "").toString();
        if (!content) return;

        let copied = false;
        try {
            if (navigator?.clipboard?.writeText) {
                await navigator.clipboard.writeText(content);
                copied = true;
            }
        } catch {
            copied = false;
        }

        if (!copied) {
            const ta = document.createElement("textarea");
            ta.value = content;
            ta.setAttribute("readonly", "");
            ta.style.position = "fixed";
            ta.style.top = "-9999px";
            document.body.appendChild(ta);
            ta.select();
            copied = document.execCommand("copy");
            document.body.removeChild(ta);
        }

        if (copied) {
            setCopiedMessageKey(messageKey);
            if (copyResetTimerRef.current) clearTimeout(copyResetTimerRef.current);
            copyResetTimerRef.current = setTimeout(() => setCopiedMessageKey(""), 1200);
        }
    };

    const loadChats = async () => {
        try {
            const response = await fetch(CHATS_API, { headers: getHeaders(false, true) });
            const data = await parseApiResponse(response);
            if (!response.ok) throw new Error(data.detail || "Failed to load chats");
            setChats(data);
            if (data.length > 0 && !currentChatId) setCurrentChatId(data[0].id);
        } catch (error) {
            console.error("Error loading chats:", error);
        }
    };

    const loadChatMessages = async (chatId) => {
        try {
            const response = await fetch(`${CHATS_API}/${chatId}`, { headers: getHeaders(false, true) });
            const data = await parseApiResponse(response);
            if (!response.ok) throw new Error(data.detail || "Failed to load messages");
            setMessages(data.messages || []);
        } catch (error) {
            console.error("Error loading messages:", error);
        }
    };

    const createNewChat = async () => {
        try {
            const response = await fetch(`${CHATS_API}/new`, { method: "POST", headers: getHeaders(false, true) });
            const data = await parseApiResponse(response);
            if (!response.ok) throw new Error(data.detail || "Failed to create chat");
            setCurrentChatId(data.id);
            setMessages([]);
            await loadChats();
        } catch (error) {
            console.error("Error creating chat:", error);
        }
    };

    const deleteChat = async (chatId) => {
        try {
            const response = await fetch(`${CHATS_API}/${chatId}`, { method: "DELETE", headers: getHeaders(false, true) });
            const data = await parseApiResponse(response);
            if (!response.ok) throw new Error(data.detail || "Failed to delete chat");
            if (currentChatId === chatId) {
                setCurrentChatId(null);
                setMessages([]);
            }
            await loadChats();
        } catch (error) {
            console.error("Error deleting chat:", error);
        }
    };

    const streamAssistantReply = (fullText) => {
        const messageId = `assistant-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
        setMessages((prev) => [
            ...prev,
            { id: messageId, role: "assistant", content: "", timestamp: new Date().toISOString() }
        ]);
        setIsStreaming(true);

        return new Promise((resolve) => {
            let i = 0;
            let rafId = 0;
            const tick = () => {
                const step = Math.max(2, Math.ceil(fullText.length / 180));
                i = Math.min(fullText.length, i + step);
                const partial = fullText.slice(0, i);
                setMessages((prev) =>
                    prev.map((m) => (m.id === messageId ? { ...m, content: partial } : m))
                );
                if (i < fullText.length) {
                    rafId = requestAnimationFrame(tick);
                } else {
                    setIsStreaming(false);
                    resolve();
                }
            };
            rafId = requestAnimationFrame(tick);
        });
    };

    const sendMessage = async () => {
        if (!inputMessage.trim() || isLoading) return;
        const userMessage = inputMessage.trim();
        setInputMessage("");
        resetTextareaHeight();
        setIsLoading(true);
        setIsWaitingResponse(true);

        setMessages((prev) => [
            ...prev,
            { role: "user", content: userMessage, timestamp: new Date().toISOString() }
        ]);

        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: getHeaders(true, true),
                body: JSON.stringify({ message: userMessage, chat_id: currentChatId })
            });
            const data = await parseApiResponse(response);
            if (!response.ok) throw new Error(data.error || data.detail || "Failed to get response");

            setIsWaitingResponse(false);
            await streamAssistantReply(data.reply || "");
            if (data.chat_id !== currentChatId) setCurrentChatId(data.chat_id);
            await loadChats();
        } catch (error) {
            console.error("Error sending message:", error);
            setIsWaitingResponse(false);
            await streamAssistantReply(`Error: ${error.message}`);
        } finally {
            setIsLoading(false);
            setIsWaitingResponse(false);
            setIsStreaming(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const submitLogin = async ({ username, password }) => {
        setAuthError("");
        setAuthLoading(true);
        try {
            const endpoint = authMode === "signup" ? "/auth/signup" : "/auth/login";
            const response = await fetch(endpoint, {
                method: "POST",
                headers: getHeaders(true),
                body: JSON.stringify({ username, password })
            });
            const data = await parseApiResponse(response);
            if (!response.ok) throw new Error(data.detail || `${authMode === "signup" ? "Signup" : "Login"} failed`);
            localStorage.setItem(USER_ID_KEY, data.user_id);
            setUserId(data.user_id);
            setAuthMode("login");
        } catch (error) {
            setAuthError(error.message);
        } finally {
            setAuthLoading(false);
        }
    };

    if (!userId) {
        return <LoginScreen mode={authMode} setMode={setAuthMode} onSubmit={submitLogin} isLoading={authLoading} error={authError} />;
    }

    return (
        <div className="flex h-screen bg-[#161A30] text-[#F0ECE5]">
            <div className="fixed inset-0 bg-[radial-gradient(circle_at_0%_0%,rgba(223,208,184,0.08),transparent_38%),radial-gradient(circle_at_100%_100%,rgba(148,137,121,0.2),transparent_42%)] pointer-events-none" />

            <div className={`${sidebarOpen ? "w-72" : "w-0"} relative transition-all duration-500 ease-in-out overflow-hidden bg-[#393E46]/95 border-r border-[#948979]/45 flex flex-col`}>
                <div className="p-4 border-b border-[#948979]/35">
                    <button
                        onClick={createNewChat}
                        className="w-full bg-[#F0ECE5] hover:bg-[#DFD0B8] active:scale-[0.99] text-[#161A30] font-semibold py-2.5 px-4 rounded-xl transition-all duration-200 ease-out flex items-center justify-center gap-2"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        New Chat
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
                    {chats.length === 0 ? (
                        <div className="text-[#948979] text-sm text-center mt-4">No chats yet. Start a new conversation.</div>
                    ) : (
                        chats.map((chat) => (
                            <div
                                key={chat.id}
                                className={`group p-3 mb-2 rounded-xl cursor-pointer transition-all duration-250 ease-out transform-gpu ${
                                    currentChatId === chat.id
                                        ? "bg-[#F0ECE5] text-[#161A30] shadow-md ring-1 ring-[#F0ECE5]/50 scale-[1.01]"
                                        : "bg-[#161A30]/80 hover:bg-[#161A30] hover:-translate-y-[1px] text-[#F0ECE5] border border-[#948979]/35"
                                }`}
                                onClick={() => setCurrentChatId(chat.id)}
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex-1 min-w-0">
                                        <div className="font-medium truncate">{chat.title}</div>
                                        <div className={`text-xs mt-1 ${currentChatId === chat.id ? "text-[#393E46]" : "text-[#948979]"}`}>
                                            Created: {formatDateTime(chat.created_at)}
                                        </div>
                                    </div>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setDeleteConfirmChatId(chat.id);
                                        }}
                                        className={`ml-2 transition-all duration-200 ease-out ${
                                            currentChatId === chat.id
                                                ? "text-[#393E46] hover:text-red-700"
                                                : "text-[#948979] hover:text-red-400"
                                        } ${
                                            "opacity-0 translate-y-1 group-hover:opacity-100 group-hover:translate-y-0"
                                        }`}
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                        </svg>
                                    </button>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>

            <div className="flex-1 flex flex-col relative">
                <div className="bg-[#393E46]/92 border-b border-[#948979]/35 p-4 flex items-center justify-between backdrop-blur">
                    <div className="flex items-center gap-4">
                        <button onClick={() => setSidebarOpen(!sidebarOpen)} className="text-[#948979] hover:text-[#F0ECE5] transition-colors duration-200">
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                        <h1 className="text-xl font-semibold tracking-tight text-[#F0ECE5]">Mnemos</h1>
                        <span className="text-xs text-[#948979]">User: {userId}</span>
                    </div>
                    <button onClick={logout} className="text-sm bg-[#161A30] hover:bg-[#393E46] active:scale-[0.99] border border-[#948979]/45 text-[#F0ECE5] px-3 py-1.5 rounded-lg transition-all duration-200">
                        Logout
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
                    {messages.length === 0 ? (
                        <div className="flex items-center justify-center h-full">
                            <div className="max-w-2xl w-full rounded-2xl border border-[#948979]/45 bg-[#393E46]/88 px-6 py-5 shadow-[0_12px_40px_rgba(0,0,0,0.28)]">
                                <p className="text-xs uppercase tracking-[0.18em] text-[#948979] mb-2">Mnemos</p>
                                <p className="text-lg leading-relaxed text-[#F0ECE5]">
                                    <TypewriterText text={WELCOME_MESSAGE} speed={16} active={showAnimatedWelcome} />
                                </p>
                                <p className="text-sm text-[#948979] mt-3">Start by sharing anything you want me to remember.</p>
                            </div>
                        </div>
                    ) : (
                        <div className="max-w-4xl mx-auto space-y-5">
                            {messages.map((msg, idx) => {
                                const messageKey = msg.id || `${msg.role}-${idx}`;
                                const isCopied = copiedMessageKey === messageKey;
                                const isUser = msg.role === "user";
                                return (
                                <div key={messageKey} className={`flex fade-up ${isUser ? "justify-end" : "justify-start"}`}>
                                    <div className="group max-w-[85%]">
                                        <div className={`flex items-center gap-2 ${isUser ? "justify-end" : "justify-start"}`}>
                                            {!isUser && (
                                                <div className="h-8 w-8 shrink-0 rounded-full border border-[#948979]/45 bg-[#393E46] text-[#F0ECE5] text-sm font-semibold flex items-center justify-center">
                                                    M
                                                </div>
                                            )}
                                            <div className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}>
                                                <div
                                                    className={`rounded-2xl px-5 py-3 shadow transition-all duration-200 ${
                                                        isUser
                                                            ? "bg-[#F0ECE5] text-[#161A30]"
                                                            : "bg-[#31304D] text-[#F0ECE5] border border-[#948979]/35"
                                                    }`}
                                                >
                                                    <div className="whitespace-pre-wrap break-words leading-relaxed">{msg.content}</div>
                                                </div>
                                                <button
                                                    type="button"
                                                    aria-label="Copy message"
                                                    onClick={() => copyMessageToClipboard(msg.content, messageKey)}
                                                    className="mt-1 inline-flex h-8 w-8 items-center justify-center text-[#948979] opacity-0 transition-all duration-200 delay-0 group-hover:opacity-100 group-hover:delay-150 hover:text-[#F0ECE5] focus:opacity-100 focus:delay-0 focus:outline-none"
                                                >
                                                    {isCopied ? (
                                                        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
                                                            <path strokeLinecap="round" strokeLinejoin="round" d="M20 6L9 17l-5-5" />
                                                        </svg>
                                                    ) : (
                                                        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
                                                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                                                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                                                        </svg>
                                                    )}
                                                </button>
                                            </div>
                                            {isUser && (
                                                <div className="h-8 w-8 shrink-0 rounded-full border border-[#948979]/45 bg-[#393E46] text-[#F0ECE5] text-sm font-semibold flex items-center justify-center">
                                                    {userInitial}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            )})}
                            {isWaitingResponse && (
                                <div className="flex justify-start">
                                    <div className="bg-[#31304D] border border-[#948979]/35 rounded-2xl px-4 py-2 text-[#948979] text-sm">
                                        Thinking...
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                <div className="bg-[#393E46]/92 border-t border-[#948979]/35 p-4 backdrop-blur">
                    <div className="max-w-4xl mx-auto flex items-end gap-3">
                        <textarea
                            ref={textareaRef}
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            onKeyDown={handleKeyPress}
                            rows={1}
                            placeholder="Type your message..."
                            disabled={isLoading}
                            className="flex-1 min-h-[48px] max-h-[180px] resize-none bg-[#161A30] border border-[#948979]/45 text-[#F0ECE5] placeholder-[#948979] rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-[#948979] transition-all duration-200 disabled:opacity-50 custom-scrollbar"
                        />
                        <button
                            onClick={sendMessage}
                            disabled={isLoading || !inputMessage.trim()}
                            className="bg-[#948979] hover:bg-[#F0ECE5] active:scale-[0.99] disabled:bg-[#393E46] disabled:text-[#948979] disabled:cursor-not-allowed text-[#161A30] font-semibold px-6 py-3 rounded-xl transition-all duration-200"
                        >
                            {isLoading ? "Sending..." : "Send"}
                        </button>
                    </div>
                </div>
            </div>

            {deleteConfirmChatId && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm fade-in">
                    <div className="w-full max-w-sm rounded-2xl border border-[#948979]/45 bg-[#393E46] p-5 shadow-2xl soft-pop">
                        <h3 className="text-lg font-semibold text-[#F0ECE5]">Delete Chat?</h3>
                        <p className="mt-2 text-sm text-[#948979]">
                            This will permanently remove this chat from your sidebar.
                        </p>
                        <div className="mt-5 flex justify-end gap-2">
                            <button
                                onClick={() => setDeleteConfirmChatId(null)}
                                className="rounded-lg border border-[#948979]/40 bg-[#161A30] px-4 py-2 text-sm text-[#F0ECE5] transition-colors hover:bg-[#393E46]"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={async () => {
                                    const chatId = deleteConfirmChatId;
                                    setDeleteConfirmChatId(null);
                                    await deleteChat(chatId);
                                }}
                                className="rounded-lg bg-red-500/90 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-red-500"
                            >
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

const rootElement = document.getElementById("root");
if (ReactDOM.createRoot) {
    const root = ReactDOM.createRoot(rootElement);
    root.render(<App />);
} else {
    ReactDOM.render(<App />, rootElement);
}
