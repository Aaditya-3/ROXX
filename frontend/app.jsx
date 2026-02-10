const { useEffect, useRef, useState } = React;

const API_URL = "/chat";
const CHATS_API = "/chats";
const USER_ID_KEY = "user_id";

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
        <div className="min-h-screen bg-slate-900 text-slate-100 flex items-center justify-center px-4">
            <div className="w-full max-w-md bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl">
                <h1 className="text-2xl font-bold mb-2 text-white">AI Chat with Memory</h1>
                <p className="text-slate-400 text-sm mb-5">
                    {mode === "login" ? "Login to continue" : "Create an account first"}
                </p>
                <div className="grid grid-cols-2 gap-2 mb-5 bg-slate-700 p-1 rounded-lg">
                    <button
                        type="button"
                        onClick={() => setMode("login")}
                        className={`py-2 rounded-md text-sm font-semibold transition-colors ${
                            mode === "login" ? "bg-blue-600 text-white" : "text-slate-300 hover:text-white"
                        }`}
                    >
                        Login
                    </button>
                    <button
                        type="button"
                        onClick={() => setMode("signup")}
                        className={`py-2 rounded-md text-sm font-semibold transition-colors ${
                            mode === "signup" ? "bg-blue-600 text-white" : "text-slate-300 hover:text-white"
                        }`}
                    >
                        Signup
                    </button>
                </div>
                <form onSubmit={submit} className="space-y-4">
                    <input
                        type="text"
                        required
                        placeholder="Username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        className="w-full bg-slate-700 text-white placeholder-slate-400 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <input
                        type="password"
                        required
                        placeholder="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full bg-slate-700 text-white placeholder-slate-400 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    {error && (
                        <div className="text-sm text-red-300 bg-red-900/30 border border-red-800 rounded-lg px-3 py-2">
                            {error}
                        </div>
                    )}
                    <button
                        type="submit"
                        disabled={isLoading}
                        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white font-semibold py-3 rounded-lg transition-colors"
                    >
                        {isLoading ? "Please wait..." : mode === "login" ? "Login" : "Signup"}
                    </button>
                </form>
            </div>
        </div>
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
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const messagesEndRef = useRef(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isLoading]);

    useEffect(() => {
        if (!userId) return;
        loadChats();
    }, [userId]);

    useEffect(() => {
        if (currentChatId) {
            loadChatMessages(currentChatId);
        } else {
            setMessages([]);
        }
    }, [currentChatId]);

    const getHeaders = (withJson = false, includeUser = false) => {
        const headers = {};
        if (withJson) headers["Content-Type"] = "application/json";
        if (includeUser && userId) headers["X-User-ID"] = userId;
        return headers;
    };

    const logout = () => {
        localStorage.removeItem(USER_ID_KEY);
        setUserId("");
        setChats([]);
        setCurrentChatId(null);
        setMessages([]);
        setAuthError("");
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

    const sendMessage = async () => {
        if (!inputMessage.trim() || isLoading) return;
        const userMessage = inputMessage.trim();
        setInputMessage("");
        setIsLoading(true);

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

            setMessages((prev) => [
                ...prev,
                { role: "assistant", content: data.reply, timestamp: new Date().toISOString() }
            ]);
            if (data.chat_id !== currentChatId) setCurrentChatId(data.chat_id);
            await loadChats();
        } catch (error) {
            console.error("Error sending message:", error);
            setMessages((prev) => [
                ...prev,
                { role: "assistant", content: `Error: ${error.message}`, timestamp: new Date().toISOString() }
            ]);
        } finally {
            setIsLoading(false);
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
        <div className="flex h-screen bg-slate-900 text-slate-100">
            <div className={`${sidebarOpen ? "w-64" : "w-0"} transition-all duration-300 overflow-hidden bg-slate-800 border-r border-slate-700 flex flex-col`}>
                <div className="p-4 border-b border-slate-700">
                    <button
                        onClick={createNewChat}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        New Chat
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
                    {chats.length === 0 ? (
                        <div className="text-slate-400 text-sm text-center mt-4">No chats yet. Start a new conversation!</div>
                    ) : (
                        chats.map((chat) => (
                            <div
                                key={chat.id}
                                className={`group p-3 mb-2 rounded-lg cursor-pointer transition-colors ${currentChatId === chat.id ? "bg-blue-600 text-white" : "bg-slate-700 hover:bg-slate-600 text-slate-200"}`}
                                onClick={() => setCurrentChatId(chat.id)}
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex-1 min-w-0">
                                        <div className="font-medium truncate">{chat.title}</div>
                                        <div className={`text-xs mt-1 ${currentChatId === chat.id ? "text-blue-100" : "text-slate-400"}`}>
                                            {new Date(chat.updated_at).toLocaleDateString()}
                                        </div>
                                    </div>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            deleteChat(chat.id);
                                        }}
                                        className="ml-2 opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition-opacity"
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

            <div className="flex-1 flex flex-col">
                <div className="bg-slate-800 border-b border-slate-700 p-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <button onClick={() => setSidebarOpen(!sidebarOpen)} className="text-slate-300 hover:text-white transition-colors">
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                        <h1 className="text-xl font-bold text-white">AI Chat with Memory</h1>
                        <span className="text-xs text-slate-400">User: {userId}</span>
                    </div>
                    <button onClick={logout} className="text-sm bg-slate-700 hover:bg-slate-600 text-slate-100 px-3 py-1.5 rounded-md transition-colors">
                        Logout
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
                    {messages.length === 0 ? (
                        <div className="flex items-center justify-center h-full">
                            <div className="text-center text-slate-400">
                                <p className="text-lg font-medium mb-2">Start a conversation</p>
                                <p className="text-sm">Send a message to begin chatting with AI</p>
                            </div>
                        </div>
                    ) : (
                        <div className="max-w-4xl mx-auto space-y-6">
                            {messages.map((msg, idx) => (
                                <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                                    <div className={`max-w-[80%] rounded-2xl px-5 py-3 ${msg.role === "user" ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-100"}`}>
                                        <div className="whitespace-pre-wrap break-words">{msg.content}</div>
                                    </div>
                                </div>
                            ))}
                            {isLoading && (
                                <div className="flex justify-start">
                                    <div className="bg-slate-700 rounded-2xl px-5 py-3">Sending...</div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                <div className="bg-slate-800 border-t border-slate-700 p-4">
                    <div className="max-w-4xl mx-auto flex gap-3">
                        <input
                            type="text"
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            onKeyDown={handleKeyPress}
                            placeholder="Type your message..."
                            disabled={isLoading}
                            className="flex-1 bg-slate-700 text-white placeholder-slate-400 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                        />
                        <button
                            onClick={sendMessage}
                            disabled={isLoading || !inputMessage.trim()}
                            className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white font-semibold px-6 py-3 rounded-lg transition-colors"
                        >
                            {isLoading ? "Sending..." : "Send"}
                        </button>
                    </div>
                </div>
            </div>
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
