import { useState, useEffect } from 'react';
import { api } from '../api';
import SocratesView from './SocratesView';
import './AnalyzeTab.css';

/**
 * AnalyzeTab - Entry point for Logic Modes
 *
 * Provides access to different thinking/analysis modes:
 * - Socrates Mode (Socratic dialogue)
 * - Future: Lawyer Mode, Red Team, etc.
 */
function AnalyzeTab({ onClose }) {
  const [modes, setModes] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeMode, setActiveMode] = useState(null);

  // Load available modes on mount
  useEffect(() => {
    loadModes();
  }, []);

  const loadModes = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await api.listModes();
      setModes(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Get icon for mode category
   */
  const getModeIcon = (category) => {
    const icons = {
      'critical-thinking': '?',
      'analysis': '!',
      'creative': '*',
      'debate': '~',
    };
    return icons[category] || '>';
  };

  /**
   * Get description for mode kind
   */
  const getKindLabel = (kind) => {
    const labels = {
      'interactive': 'Interactive Dialogue',
      'analysis': 'Analysis Tool',
      'composite': 'Multi-Agent',
    };
    return labels[kind] || kind;
  };

  // If a mode is active, render its view in the overlay
  if (activeMode === 'socrates') {
    return (
      <div className="analyze-overlay">
        <SocratesView onClose={() => setActiveMode(null)} />
      </div>
    );
  }

  // Mode selection view
  return (
    <div className="analyze-overlay" onClick={(e) => {
      // Close when clicking overlay background
      if (e.target === e.currentTarget && onClose) {
        onClose();
      }
    }}>
    <div className="analyze-tab">
      <div className="analyze-header">
        <h2>Analyze</h2>
        <p className="description">
          Select a thinking mode to guide your analysis with structured reasoning protocols.
        </p>
        {onClose && (
          <button className="close-btn" onClick={onClose}>
            Close
          </button>
        )}
      </div>

      {isLoading && (
        <div className="loading-state">
          <span>Loading modes...</span>
        </div>
      )}

      {error && (
        <div className="error-state">
          <p>Failed to load modes: {error}</p>
          <button onClick={loadModes}>Retry</button>
        </div>
      )}

      {!isLoading && !error && (
        <div className="modes-grid">
          {modes.map((mode) => (
            <div
              key={mode.id}
              className={`mode-card ${mode.id}`}
              onClick={() => setActiveMode(mode.id)}
            >
              <div className="mode-icon">{getModeIcon(mode.category)}</div>
              <div className="mode-info">
                <h3 className="mode-name">{mode.display_name}</h3>
                <p className="mode-description">{mode.description}</p>
                <div className="mode-meta">
                  <span className="mode-kind">{getKindLabel(mode.kind)}</span>
                  <span className="mode-category">{mode.category}</span>
                </div>
              </div>
            </div>
          ))}

          {modes.length === 0 && (
            <div className="empty-state">
              <p>No analysis modes available yet.</p>
            </div>
          )}
        </div>
      )}

      <div className="analyze-footer">
        <p className="hint">
          Each mode uses a different protocol to structure your thinking and uncover blind spots.
        </p>
      </div>
    </div>
    </div>
  );
}

export default AnalyzeTab;
