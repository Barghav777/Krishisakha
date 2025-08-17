document.addEventListener('DOMContentLoaded', initializeApp);

function initializeApp() {
  const API_URL = '/process_input';
  const themeToggle = document.getElementById('theme-toggle');
  const chatContainer = document.getElementById('chat-container');
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const voiceBtn = document.getElementById('voice-btn');
  const imageBtn = document.getElementById('image-btn');
  const imageUpload = document.getElementById('image-upload');
  const newChatBtn = document.getElementById('new-chat-btn');
  const chatHistoryContainer = document.getElementById('chat-history');
  const faqList = document.getElementById('faq-list');
  const previewContainer = document.getElementById('input-preview-container');
  const hamburgerBtn = document.getElementById('hamburger-btn');
  const faqBtnMobile = document.getElementById('faq-btn-mobile');
  const sidebar = document.getElementById('sidebar');
  const faqSidebar = document.getElementById('faq-sidebar');
  const chatWrapper = document.querySelector('.chat-wrapper');

  let currentChat = { id: Date.now(), messages: [] };
  let mediaRecorder;
  let audioChunks = [];
  let isLoading = false;
  let selectedImages = [];
  let selectedVoiceFile = null;

  function speakText(text, language = 'en') {
    if (!text || !('speechSynthesis' in window)) {
      console.warn('Text-to-speech not supported or no text provided.');
      return;
    }

    speechSynthesis.cancel(); // Stop any ongoing speech

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;

    // --- UPDATED SECTION START ---

    // Expanded language map for all supported Indian languages
    const languageMap = {
      'en': 'en-US', // English
      'hi': 'hi-IN', // Hindi
      'gu': 'gu-IN', // Gujarati
      'bn': 'bn-IN', // Bengali
      'ta': 'ta-IN', // Tamil
      'te': 'te-IN', // Telugu
      'mr': 'mr-IN', // Marathi
      'kn': 'kn-IN', // Kannada
      'ml': 'ml-IN', // Malayalam
      'pa': 'pa-IN', // Punjabi
      'ur': 'ur-IN', // Urdu
      'as': 'as-IN', // Assamese
      'or': 'or-IN'  // Odia
    };

    const targetLang = languageMap[language] || language || 'en-US';
    utterance.lang = targetLang;

    // Improved voice selection logic
    let voices = speechSynthesis.getVoices();
    
    // 1. Try to find a perfect match (e.g., 'hi-IN')
    let selectedVoice = voices.find(voice => voice.lang === targetLang);

    // 2. If no perfect match, find a voice for the base language (e.g., 'hi')
    if (!selectedVoice) {
      selectedVoice = voices.find(voice => voice.lang.startsWith(language));
    }

    if (selectedVoice) {
      utterance.voice = selectedVoice;
    } else {
      console.warn(`No voice found for language: ${targetLang}. Using browser default.`);
    }
    
    // --- UPDATED SECTION END ---

    utterance.onerror = (event) => {
      console.error('Speech synthesis error:', event.error);
    };

    speechSynthesis.speak(utterance);
  }

  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'dark') {
    themeToggle.checked = true;
    document.body.classList.add('dark-mode');
  }
  themeToggle.addEventListener('change', () => {
    document.body.classList.toggle('dark-mode', themeToggle.checked);
    localStorage.setItem('theme', themeToggle.checked ? 'dark' : 'light');
  });

  if ('speechSynthesis' in window) {
    speechSynthesis.onvoiceschanged = () => {};
  }

  document.addEventListener('submit', (e) => e.preventDefault());
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const target = e.target;
      if (target.tagName === 'TEXTAREA' && target.id === 'user-input' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }
  });

  sendBtn.addEventListener('click', sendMessage);
  userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  
  voiceBtn.addEventListener('click', toggleVoiceRecording);
  imageBtn.addEventListener('click', () => imageUpload.click());
  imageUpload.addEventListener('change', handleImagePreview);
  newChatBtn.addEventListener('click', startNewChat);

  hamburgerBtn.addEventListener('click', (e) => { sidebar.classList.toggle('visible'); e.stopPropagation(); });
  faqBtnMobile.addEventListener('click', (e) => { faqSidebar.classList.toggle('visible'); e.stopPropagation(); });
  chatWrapper.addEventListener('click', () => {
    if (sidebar.classList.contains('visible')) sidebar.classList.remove('visible');
    if (faqSidebar.classList.contains('visible')) faqSidebar.classList.remove('visible');
  });

  userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = `${userInput.scrollHeight}px`;
  });

  if (faqList) {
    faqList.querySelectorAll('.faq-item').forEach(item => {
      const questionText = item.querySelector('.faq-question')?.textContent || '';
      const answerP = item.querySelector('.faq-answer');
      if (answerP && !answerP.querySelector('.ask-ai-btn')) {
        const askAIButton = document.createElement('button');
        askAIButton.className = 'ask-ai-btn';
        askAIButton.textContent = 'Ask AI for more details';
        askAIButton.onclick = (e) => {
          e.stopPropagation();
          userInput.value = questionText;
          sendMessage();
        };
        answerP.appendChild(document.createElement('br'));
        answerP.appendChild(askAIButton);
      }
    });
  }

  const chats = JSON.parse(localStorage.getItem('krishi-chats') || '{}');
  const chatIds = Object.keys(chats).map(Number);
  if (chatIds.length > 0) {
    const mostRecentId = Math.max(...chatIds);
    loadChat(mostRecentId);
  } else {
    currentChat = { id: Date.now(), messages: [] };
    addMessage({ from: 'bot', english_response: 'Welcome to KrishiSakha! How can I help you today?' });
  }
  renderChatHistory();

  function sendMessage() {
    const query = userInput.value.trim();
    const hasAnyInput = !!query || selectedImages.length > 0 || !!selectedVoiceFile;
    if (!hasAnyInput || isLoading) return;
    
    isLoading = true;
    const messagePayload = { from: 'user', id: `msg-${Date.now()}` };
    const apiPayload = {};

    if (query) { messagePayload.text = query; apiPayload.text = query; }
    if (selectedImages.length > 0) { messagePayload.imageURL = URL.createObjectURL(selectedImages[0]); apiPayload.images = [...selectedImages]; }
    if (selectedVoiceFile) { messagePayload.audioURL = URL.createObjectURL(selectedVoiceFile); apiPayload.voice = selectedVoiceFile; }
    
    addMessage(messagePayload);
    userInput.value = '';
    userInput.style.height = 'auto';
    previewContainer.innerHTML = '';
    
    getBotResponse(apiPayload).finally(() => { isLoading = false; });
    
    selectedImages = [];
    selectedVoiceFile = null;
    imageUpload.value = '';
  }

  function handleImagePreview(event) {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;
    selectedImages.push(...files);
    previewContainer.innerHTML = '';
    selectedImages.forEach((file, idx) => {
      const wrapper = document.createElement('div');
      wrapper.className = 'input-preview-item';
      const img = document.createElement('img');
      img.className = 'input-preview-img';
      img.src = URL.createObjectURL(file);
      const removeBtn = document.createElement('button');
      removeBtn.className = 'remove-preview-btn';
      removeBtn.textContent = '×';
      removeBtn.onclick = () => {
        selectedImages.splice(idx, 1);
        handleImagePreview({ target: { files: [] } });
      };
      wrapper.appendChild(img);
      wrapper.appendChild(removeBtn);
      previewContainer.appendChild(wrapper);
    });
  }

  function toggleVoiceRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          mediaRecorder = new MediaRecorder(stream);
          audioChunks = [];
          mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
          mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            selectedVoiceFile = new File([blob], 'recording.webm', { type: 'audio/webm' });
            showVoicePreview(URL.createObjectURL(blob));
            stream.getTracks().forEach(t => t.stop());
          };
          mediaRecorder.start();
          voiceBtn.textContent = '⏹️';
        })
        .catch(err => alert('Microphone access denied.'));
    } else if (mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      voiceBtn.textContent = '🎤';
    }
  }

  function showVoicePreview(audioURL) {
    const wrapper = document.createElement('div');
    wrapper.className = 'input-preview-item voice-preview';
    wrapper.style.cssText = 'display: flex; align-items: center; background-color: var(--primary-green); border-radius: 15px; padding: 8px 12px; color: white;';
    const audioIcon = document.createElement('span'); audioIcon.textContent = '🎤'; audioIcon.style.marginRight = '8px';
    const label = document.createElement('span'); label.textContent = 'Voice Recording (queued)'; label.style.fontSize = '14px';
    const removeBtn = document.createElement('button'); removeBtn.className = 'remove-preview-btn'; removeBtn.textContent = '×';
    removeBtn.style.marginLeft = '10px';
    removeBtn.onclick = () => { selectedVoiceFile = null; wrapper.remove(); };
    wrapper.appendChild(audioIcon); wrapper.appendChild(label); wrapper.appendChild(removeBtn);
    previewContainer.appendChild(wrapper);
  }

  async function getBotResponse(input) {
    toggleLoading(true);
    const loadingBubble = createMessageBubble({ from: 'bot', loading: true });
    chatContainer.prepend(loadingBubble);
    const formData = new FormData();
    if (input.text) formData.append('text', input.text);
    if (input.voice) formData.append('voice', input.voice, 'recording.webm');
    if (input.images && input.images.length) {
      input.images.forEach((f) => formData.append('image', f, f.name));
    }

    try {
      const response = await fetch(API_URL, { method: 'POST', body: formData, mode: 'cors', cache: 'no-store', signal: AbortSignal.timeout(30000) });
      const contentType = response.headers.get('content-type') || '';
      if (!response.ok) {
        const raw = contentType.includes('application/json') ? await response.json() : await response.text();
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }
      const data = contentType.includes('application/json') ? await response.json() : {};
      addMessage({ from: 'bot', original_response: data?.bot_response?.original_response || "No response.", english_response: data?.bot_response?.english_response || "No response.", detected_lang_code: data?.bot_response?.detected_lang_code || "en" });
    } catch (error) {
        let errorMessage = `Error: ${error.message}`;
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            errorMessage = 'Connection failed. Please ensure the server is running.';
        } else if (error.name === 'TimeoutError') {
            errorMessage = 'Request timed out. The server may be busy.';
        }
        addMessage({ from: 'bot', english_response: errorMessage, original_response: errorMessage });
    } finally {
      loadingBubble.remove();
      toggleLoading(false);
    }
  }

  function addMessage(msg) {
    currentChat.messages.push(msg);
    renderChat();
    saveChatToHistory();
  }

  function renderChat() {
    chatContainer.innerHTML = '';
    currentChat.messages.slice().reverse().forEach(msg => {
      chatContainer.appendChild(createMessageBubble(msg));
    });
  }

  function createMessageBubble(msg) {
    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${msg.from === 'user' ? 'user-bubble' : 'bot-bubble'}`;
    if (msg.id) bubble.id = msg.id;

    if (msg.audioURL) {
      const audio = new Audio(msg.audioURL);
      const wrapper = document.createElement('div');
      wrapper.className = 'voice-message';
      const playBtn = document.createElement('button'); playBtn.className = 'voice-play-btn'; playBtn.textContent = '▶️';
      const progress = document.createElement('input'); progress.type = 'range'; progress.min = 0; progress.value = 0; progress.step = 0.01; progress.className = 'voice-progress';
      const timeLabel = document.createElement('span'); timeLabel.className = 'voice-time'; timeLabel.textContent = '0:00';
      playBtn.onclick = () => { if (audio.paused) { audio.play(); playBtn.textContent = '⛔'; } else { audio.pause(); playBtn.textContent = '▶️'; } };
      audio.addEventListener('timeupdate', () => { progress.value = audio.currentTime; progress.max = audio.duration || 0; timeLabel.textContent = formatTime(audio.currentTime); });
      progress.addEventListener('input', () => { audio.currentTime = progress.value; });
      audio.addEventListener('ended', () => { playBtn.textContent = '▶️'; });
      wrapper.append(playBtn, progress, timeLabel);
      bubble.appendChild(wrapper);
    }
    if (msg.imageURL) {
      const img = document.createElement('img');
      img.className = 'bubble-image';
      img.src = msg.imageURL;
      bubble.appendChild(img);
    }
    if (msg.loading) {
      bubble.innerHTML = `<div class="loading-indicator"><div></div><div></div><div></div></div>`;
      return bubble;
    }
    if (msg.from === 'bot' && msg.original_response && msg.english_response) {
      const orig = document.createElement('div'); orig.className = 'original-lang'; orig.textContent = msg.original_response;
      const eng = document.createElement('div'); eng.className = 'english-lang'; eng.textContent = msg.english_response;
      const origSpeakBtn = document.createElement('button'); origSpeakBtn.className = 'speak-btn'; origSpeakBtn.textContent = '🔊 Speak Original';
      origSpeakBtn.onclick = () => speakText(msg.original_response, msg.detected_language || 'hi');
      const engSpeakBtn = document.createElement('button'); engSpeakBtn.className = 'speak-btn'; engSpeakBtn.textContent = '🔊 Speak English';
      engSpeakBtn.onclick = () => speakText(msg.english_response, 'en');
      bubble.append(orig, origSpeakBtn, eng, engSpeakBtn);
      return bubble;
    }
    const textContent = msg.text || (msg.from === 'bot' ? msg.english_response : '');
    if (textContent) {
      const p = document.createElement('p'); p.textContent = textContent;
      bubble.appendChild(p);
      if (msg.from === 'bot') {
        const speakBtn = document.createElement('button'); speakBtn.className = 'speak-btn'; speakBtn.textContent = '🔊 Speak';
        speakBtn.onclick = () => speakText(textContent, msg.detected_language || 'en');
        bubble.appendChild(speakBtn);
      }
    }
    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
        return `${mins}:${secs}`;
    }
    return bubble;
  }

  function saveChatToHistory() {
    const chats = JSON.parse(localStorage.getItem('krishi-chats') || '{}');
    if (currentChat.messages.length > (currentChat.messages[0]?.from === 'bot' ? 1 : 0)) {
      chats[currentChat.id] = currentChat;
    }
    localStorage.setItem('krishi-chats', JSON.stringify(chats));
    renderChatHistory();
  }

  function renderChatHistory() {
    const chats = JSON.parse(localStorage.getItem('krishi-chats') || '{}');
    chatHistoryContainer.innerHTML = '';
    Object.keys(chats).sort((a, b) => b - a).forEach(id => {
      const chat = chats[id];
      if (chat.messages.length > 0) {
        const item = document.createElement('div');
        item.className = 'chat-history-item';
        item.dataset.chatId = id;
        const firstUserMsg = chat.messages.find(m => m.from === 'user');
        const titleSpan = document.createElement('span');
        titleSpan.textContent = (firstUserMsg?.text || 'Chat').substring(0, 25) + '...';
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-chat-btn';
        deleteBtn.textContent = '🗑️';
        deleteBtn.onclick = (e) => { e.stopPropagation(); deleteChat(id); };
        item.onclick = () => loadChat(id);
        item.appendChild(titleSpan);
        item.appendChild(deleteBtn);
        chatHistoryContainer.appendChild(item);
      }
    });
  }

  function deleteChat(id) {
    if (!confirm('Delete this chat? This cannot be undone.')) return;
    const chats = JSON.parse(localStorage.getItem('krishi-chats') || '{}');
    delete chats[id];
    localStorage.setItem('krishi-chats', JSON.stringify(chats));
    if (String(currentChat.id) === String(id)) {
      const remainingIds = Object.keys(chats).map(Number);
      if (remainingIds.length) {
        loadChat(Math.max(...remainingIds));
      } else {
        startNewChat(true);
      }
    } else {
      renderChatHistory();
    }
  }

  function loadChat(chatId) {
    const chats = JSON.parse(localStorage.getItem('krishi-chats') || '{}');
    currentChat = chats[chatId];
    renderChat();
    renderChatHistory();
  }

  function startNewChat(skipSave = false) {
    if (!skipSave && currentChat.messages.length > 1) saveChatToHistory();
    currentChat = { id: Date.now(), messages: [] };
    addMessage({ from: 'bot', english_response: 'Welcome to KrishiSakha! How can I help you today?' });
    renderChatHistory();
  }

  function toggleLoading(state) {
    isLoading = state;
    sendBtn.disabled = state;
    voiceBtn.disabled = state;
    imageBtn.disabled = state;
  }
}