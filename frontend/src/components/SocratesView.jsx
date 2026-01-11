import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api';
import LedgerPanel from './LedgerPanel';
import GlossaryPanel from './GlossaryPanel';
import './SocratesView.css';

/**
 * SocratesView - Interactive Socratic dialogue mode
 *
 * Implements the Socratic method through structured questioning,
 * tracking commitments, definitions, and contradictions in a ledger.
 */
function SocratesView({ onClose }) {
  // Session state
  const [session, setSession] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // UI state
  const [userInput, setUserInput] = useState('');
  const [showLedger, setShowLedger] = useState(true);
  const [showGlossary, setShowGlossary] = useState(false);
  const [selectedFallacy, setSelectedFallacy] = useState(null);

  // Conversation history for display
  const [messages, setMessages] = useState([]);

  // Current Socrates output (latest response)
  const [currentOutput, setCurrentOutput] = useState(null);

  // Stop recommendation state
  const [stopRecommended, setStopRecommended] = useState(false);
  const [stopCriteria, setStopCriteria] = useState([]);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input after loading completes
  useEffect(() => {
    if (!isLoading && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isLoading]);

  /**
   * Start a new Socratic session
   */
  const startSession = async (initialInquiry = null) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await api.createModeSession('socrates', initialInquiry);
      setSession(result);
      setSessionId(result.session_id);
      setMessages([]);
      setCurrentOutput(null);
      setStopRecommended(false);
      setStopCriteria([]);

      // If no initial inquiry, Socrates will ask for one
      if (!initialInquiry) {
        setMessages([{
          role: 'assistant',
          content: "Welcome to Socratic inquiry. What question or belief would you like to examine together?",
          type: 'greeting'
        }]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Send a message and get Socrates' response
   */
  const sendMessage = async () => {
    if (!userInput.trim() || isLoading) return;

    const message = userInput.trim();
    setUserInput('');
    setIsLoading(true);
    setError(null);

    // Add user message to display
    setMessages(prev => [...prev, {
      role: 'user',
      content: message
    }]);

    try {
      const result = await api.modeTurn('socrates', sessionId, message);

      // Update session with new state
      setSession(result.session);

      // Extract the response
      const output = result.output;
      setCurrentOutput(output);

      // Check for parse errors
      if (result.parse_error) {
        setError(`Parse warning: Response may be incomplete. ${result.error || ''}`);
      }

      // Add Socrates' response to display
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: output.next_question || output.acknowledgment || 'I see.',
        output: output,
        type: output.question_type || 'response'
      }]);

      // Check for stop recommendation
      if (result.stop_recommended) {
        setStopRecommended(true);
        setStopCriteria(result.stop_criteria || []);
      }

    } catch (err) {
      setError(err.message);
      // Remove the optimistic user message on error
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Handle user confirming they want to stop
   */
  const handleStop = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await api.stopModeSession('socrates', sessionId);

      // Add summary to messages
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: result.summary || 'Session concluded.',
        type: 'summary',
        isSummary: true
      }]);

      // Update session state
      setSession(prev => ({
        ...prev,
        status: 'completed'
      }));

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Handle user wanting to continue despite stop recommendation
   */
  const handleContinue = () => {
    setStopRecommended(false);
    setStopCriteria([]);
  };

  /**
   * Handle key press in input
   */
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  /**
   * Render a message
   */
  const renderMessage = (msg, index) => {
    const isUser = msg.role === 'user';
    const isSummary = msg.isSummary;

    return (
      <div
        key={index}
        className={`socrates-message ${isUser ? 'user' : 'assistant'} ${isSummary ? 'summary' : ''}`}
      >
        <div className="message-header">
          <span className="role-label">
            {isUser ? 'You' : 'Socrates'}
          </span>
          {msg.type && msg.type !== 'response' && (
            <span className={`question-type ${msg.type}`}>
              {msg.type.replace('_', ' ')}
            </span>
          )}
        </div>
        <div className="message-content">
          <ReactMarkdown>{msg.content}</ReactMarkdown>
        </div>
        {msg.output?.observation && (
          <div className="observation">
            <span className="observation-label">Observation:</span>
            <ReactMarkdown>{msg.output.observation}</ReactMarkdown>
          </div>
        )}
        {msg.output?.fallacy_alert && (
          <div className="fallacy-alert" onClick={() => setSelectedFallacy(msg.output.fallacy_alert.fallacy_id)}>
            <span className="alert-icon">!</span>
            <span className="fallacy-name">{msg.output.fallacy_alert.fallacy_name}</span>
            <span className="fallacy-note">{msg.output.fallacy_alert.note}</span>
          </div>
        )}
      </div>
    );
  };

  // Initial state - no session yet
  if (!sessionId) {
    return (
      <div className="socrates-view">
        <div className="socrates-header">
          <h2>Socrates Mode</h2>
          <p className="description">
            Engage in structured Socratic dialogue to examine beliefs,
            uncover assumptions, and refine understanding through careful questioning.
          </p>
        </div>

        <div className="socrates-start">
          <div className="start-options">
            <button
              className="start-btn primary"
              onClick={() => startSession()}
              disabled={isLoading}
            >
              {isLoading ? 'Starting...' : 'Begin Inquiry'}
            </button>
          </div>

          <div className="start-hint">
            <p>Socrates will guide you through a structured examination of your beliefs using the elenctic method.</p>
            <ul>
              <li>State a belief or thesis you want to examine</li>
              <li>Answer questions honestly and thoughtfully</li>
              <li>Watch as commitments and contradictions emerge in the ledger</li>
            </ul>
          </div>
        </div>

        {error && (
          <div className="error-banner">
            {error}
            <button onClick={() => setError(null)}>Dismiss</button>
          </div>
        )}

        {onClose && (
          <button className="close-btn" onClick={onClose}>
            Close
          </button>
        )}
      </div>
    );
  }

  // Active session view
  return (
    <div className="socrates-view active">
      <div className="socrates-header">
        <div className="header-left">
          <h2>Socrates Mode</h2>
          <div className="session-info">
            <span className="turn-count">Turn {session?.turn_count || 0} / {session?.max_turns || 12}</span>
            {session?.ledger?.inquiry && (
              <span className="inquiry" title={session.ledger.inquiry}>
                {session.ledger.inquiry.substring(0, 50)}
                {session.ledger.inquiry.length > 50 ? '...' : ''}
              </span>
            )}
          </div>
        </div>
        <div className="header-actions">
          <button
            className={`toggle-btn ${showLedger ? 'active' : ''}`}
            onClick={() => setShowLedger(!showLedger)}
          >
            Ledger
          </button>
          <button
            className={`toggle-btn ${showGlossary ? 'active' : ''}`}
            onClick={() => setShowGlossary(!showGlossary)}
          >
            Glossary
          </button>
          {onClose && (
            <button className="close-btn" onClick={onClose}>
              Close
            </button>
          )}
        </div>
      </div>

      <div className="socrates-main">
        <div className="conversation-area">
          <div className="messages-container">
            {messages.map((msg, i) => renderMessage(msg, i))}
            {isLoading && (
              <div className="socrates-message assistant loading">
                <div className="message-header">
                  <span className="role-label">Socrates</span>
                </div>
                <div className="message-content">
                  <span className="thinking-dots">Thinking...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {stopRecommended && session?.status !== 'completed' && (
            <div className="stop-recommendation">
              <div className="stop-header">
                <span className="stop-icon">i</span>
                <span>Socrates suggests concluding the inquiry</span>
              </div>
              {stopCriteria.length > 0 && (
                <ul className="stop-criteria">
                  {stopCriteria.map((c, i) => <li key={i}>{c}</li>)}
                </ul>
              )}
              <div className="stop-actions">
                <button className="stop-btn" onClick={handleStop}>
                  End & Summarize
                </button>
                <button className="continue-btn" onClick={handleContinue}>
                  Continue Inquiry
                </button>
              </div>
            </div>
          )}

          {session?.status !== 'completed' && (
            <div className="input-area">
              <textarea
                ref={inputRef}
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Enter your response..."
                disabled={isLoading}
                rows={3}
              />
              <button
                className="send-btn"
                onClick={sendMessage}
                disabled={isLoading || !userInput.trim()}
              >
                {isLoading ? 'Thinking...' : 'Send'}
              </button>
            </div>
          )}

          {session?.status === 'completed' && (
            <div className="session-completed">
              <p>This inquiry has concluded.</p>
              <button
                className="new-session-btn"
                onClick={() => {
                  setSessionId(null);
                  setSession(null);
                  setMessages([]);
                }}
              >
                Start New Inquiry
              </button>
            </div>
          )}
        </div>

        {showLedger && (
          <LedgerPanel
            ledger={session?.ledger}
            onClose={() => setShowLedger(false)}
          />
        )}

        {showGlossary && (
          <GlossaryPanel
            selectedId={selectedFallacy}
            onSelect={setSelectedFallacy}
            onClose={() => setShowGlossary(false)}
          />
        )}
      </div>

      {error && (
        <div className="error-banner">
          {error}
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
    </div>
  );
}

export default SocratesView;
