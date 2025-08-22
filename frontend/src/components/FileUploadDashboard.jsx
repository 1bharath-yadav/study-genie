import { useState, useRef, useEffect } from 'react';
import './FileUploadDashboard.css';

const FileUploadDashboard = () => {
  const [files, setFiles] = useState([]);
  const [timer, setTimer] = useState({ 
    hours: 0, 
    minutes: 0, 
    seconds: 0, 
    isRunning: false 
  });
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
  const [streak, setStreak] = useState(() => {
    const lastVisit = localStorage.getItem('lastVisit');
    const savedStreak = parseInt(localStorage.getItem('streak')) || 0;
    const today = new Date().toDateString();
    
    if (lastVisit !== today) {
      localStorage.setItem('lastVisit', today);
      localStorage.setItem('streak', String(savedStreak + 1));
      return savedStreak + 1;
    }
    return savedStreak;
  });
  const [showStreakAnimation, setShowStreakAnimation] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    { id: 1, text: "Hello! Upload a Any File I'll help you analyze it.", sender: 'bot' }
  ]);
  const [inputText, setInputText] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [selectedMenu, setSelectedMenu] = useState('upload');
  const fileInputRef = useRef(null);

  // Define menu items
  const menuItems = [
    { id: 'upload', label: 'Upload Files', icon: '📄' },
    { id: 'summarize', label: 'Summarize', icon: '📝' },
    { id: 'quiz', label: 'Generate Quiz', icon: '❓' },
    { id: 'notes', label: 'Create Notes', icon: '📔' },
    { id: 'flashcards', label: 'Flashcards', icon: '🗂' }
  ];

  // File handling functions
  const handleFileUpload = (event) => {
    const uploadedFiles = Array.from(event.target.files);
    setIsUploading(true);
    
    setTimeout(() => {
      const newFiles = uploadedFiles.map(file => ({
        id: Date.now() + Math.random(),
        name: file.name,
        type: file.type,
        size: file.size,
        uploadDate: new Date().toLocaleDateString()
      }));
      
      setFiles(prevFiles => [...prevFiles, ...newFiles]);
      setIsUploading(false);
      
      if (uploadedFiles.length > 0) {
        const fileList = uploadedFiles.map(f => f.name).join(', ');
        setChatMessages(prev => [...prev, {
          id: Date.now(),
          text: `I've uploaded ${uploadedFiles.length} file(s): ${fileList}. How can I help you with these files?`,
          sender: 'bot'
        }]);
      }
    }, 1500);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    const validFiles = droppedFiles.filter(file => 
      file.type === 'application/pdf' || 
      file.type === 'text/plain' ||
      file.name.endsWith('.txt') ||
      file.name.endsWith('.pdf')
    );
    
    if (validFiles.length > 0) {
      handleFileUpload({ target: { files: validFiles } });
    }
  };

  const removeFile = (id) => {
    setFiles(prevFiles => prevFiles.filter(file => file.id !== id));
  };

  // Chat functions
  const handleSendMessage = () => {
    if (inputText.trim() === '') return;
    
    setChatMessages(prev => [...prev, {
      id: Date.now(),
      text: inputText,
      sender: 'user'
    }]);
    
    setInputText('');
    
    setTimeout(() => {
      setChatMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: "I've processed your request based on the uploaded documents. Is there anything specific you'd like to know?",
        sender: 'bot'
      }]);
    }, 1000);
  };

  // Effects
  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
  }, [theme]);

  useEffect(() => {
    const lastVisit = localStorage.getItem('lastVisit');
    const today = new Date().toDateString();
    
    if (lastVisit !== today) {
      setShowStreakAnimation(true);
      setTimeout(() => setShowStreakAnimation(false), 2000);
    }
  }, []);

  useEffect(() => {
    let intervalId;
    if (timer.isRunning) {
      intervalId = setInterval(() => {
        setTimer(prevTimer => {
          let newSeconds = prevTimer.seconds + 1;
          let newMinutes = prevTimer.minutes;
          let newHours = prevTimer.hours;

          if (newSeconds === 60) {
            newSeconds = 0;
            newMinutes += 1;
          }
          if (newMinutes === 60) {
            newMinutes = 0;
            newHours += 1;
          }

          return { 
            ...prevTimer,
            hours: newHours, 
            minutes: newMinutes, 
            seconds: newSeconds 
          };
        });
      }, 1000);
    }

    return () => clearInterval(intervalId);
  }, [timer.isRunning]);

  // Render component
  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div className="header-title">
          <h1>Study Genie</h1>
        </div>
        
        {/* Horizontal Menu Bar */}
        <div className="menu-bar">
          {menuItems.map(item => (
            <div
              key={item.id}
              className={`menu-item ${selectedMenu === item.id ? 'selected' : ''}`}
              onClick={() => setSelectedMenu(item.id)}
            >
              <span className="menu-icon">{item.icon}</span>
              <span className="menu-label">{item.label}</span>
            </div>
          ))}
        </div>

        <div className="header-controls">
          <div className="timer-controls">
            <div className="timer" onClick={() => setTimer(prev => ({ ...prev, isRunning: !prev.isRunning }))}>
              <span className="timer-icon">⏱️</span>
              <span className="timer-text">
                {String(timer.hours).padStart(2, '0')}:
                {String(timer.minutes).padStart(2, '0')}:
                {String(timer.seconds).padStart(2, '0')}
              </span>
              <span className="timer-status">{timer.isRunning ? '⏸' : '▶️'}</span>
            </div>
            <button 
              className="timer-reset"
              onClick={() => setTimer({ hours: 0, minutes: 0, seconds: 0, isRunning: false })}
            >
              🔄
            </button>
          </div>
          <div className="streak">
            <span className="streak-icon">🔥</span>
            <span className="streak-count">{streak}</span>
          </div>
          <button 
            className="theme-toggle"
            onClick={() => {
              const newTheme = theme === 'light' ? 'dark' : 'light';
              setTheme(newTheme);
              document.body.setAttribute('data-theme', newTheme);
            }}
          >
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
        </div>
      </div>

      <div className="dashboard-content">
        {/* Main Content Area */}
        <div className="main-content">
          <div className="file-upload-section">
            <div 
              className="upload-area"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="upload-placeholder">
                <div className="upload-icon">📄</div>
                <h3>Drop files here or click to browse</h3>
                <p>Supports PDF and text files</p>
              </div>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                multiple
                accept=".pdf,.txt,text/plain,application/pdf"
                style={{ display: 'none' }}
              />
            </div>
            
            {isUploading && (
              <div className="upload-progress">
                <div className="progress-bar">
                  <div className="progress-fill"></div>
                </div>
                <p>Processing files...</p>
              </div>
            )}
            
            {files.length > 0 && (
              <div className="uploaded-files">
                <h3>Uploaded Files</h3>
                <div className="file-list">
                  {files.map(file => (
                    <div key={file.id} className="file-item">
                      <div className="file-icon">
                        {file.type === 'application/pdf' ? '📕' : '📝'}
                      </div>
                      <div className="file-info">
                        <div className="file-name">{file.name}</div>
                        <div className="file-details">
                          {Math.round(file.size / 1024)} KB • {file.uploadDate}
                        </div>
                      </div>
                      <button 
                        className="remove-file-btn"
                        onClick={() => removeFile(file.id)}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="chat-section">
            <div className="chat-messages">
              {chatMessages.map(message => (
                <div key={message.id} className={`message ${message.sender}`}>
                  <div className="message-content">
                    {message.text}
                  </div>
                </div>
              ))}
            </div>
            
            <div className="chat-input">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Ask about your documents..."
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              />
              <button onClick={handleSendMessage} disabled={inputText.trim() === ''}>
                Send
              </button>
            </div>
          </div>
        </div>
      </div>

      {showStreakAnimation && (
        <div className="streak-animation">
          +1 Day Streak! 🎉
        </div>
      )}
    </div>
  );
};

export default FileUploadDashboard;
