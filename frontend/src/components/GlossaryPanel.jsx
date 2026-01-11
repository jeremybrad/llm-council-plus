import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api';
import './GlossaryPanel.css';

/**
 * GlossaryPanel - Educational reference for logical fallacies
 *
 * Shows common fallacies with definitions, examples, and how to fix them.
 * Can be linked from fallacy alerts in Socratic dialogue.
 */
function GlossaryPanel({ selectedId, onSelect, onClose }) {
  const [glossary, setGlossary] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Load glossary on mount
  useEffect(() => {
    loadGlossary();
  }, []);

  // Expand selected fallacy when it changes
  useEffect(() => {
    if (selectedId) {
      setExpandedId(selectedId);
    }
  }, [selectedId]);

  const loadGlossary = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await api.getGlossary();
      setGlossary(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id);
    if (onSelect) {
      onSelect(expandedId === id ? null : id);
    }
  };

  // Filter glossary by search query
  const filteredGlossary = glossary.filter(item => {
    if (!searchQuery.trim()) return true;
    const query = searchQuery.toLowerCase();
    return (
      item.name.toLowerCase().includes(query) ||
      item.definition.toLowerCase().includes(query) ||
      item.example?.toLowerCase().includes(query)
    );
  });

  if (isLoading) {
    return (
      <div className="glossary-panel">
        <div className="panel-header">
          <h3>Logical Fallacies</h3>
          {onClose && <button className="close-btn" onClick={onClose}>×</button>}
        </div>
        <div className="loading-state">
          <span>Loading glossary...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glossary-panel">
        <div className="panel-header">
          <h3>Logical Fallacies</h3>
          {onClose && <button className="close-btn" onClick={onClose}>×</button>}
        </div>
        <div className="error-state">
          <p>Failed to load glossary: {error}</p>
          <button onClick={loadGlossary}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="glossary-panel">
      <div className="panel-header">
        <h3>Logical Fallacies</h3>
        {onClose && <button className="close-btn" onClick={onClose}>×</button>}
      </div>

      <div className="search-bar">
        <input
          type="text"
          placeholder="Search fallacies..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        {searchQuery && (
          <button className="clear-search" onClick={() => setSearchQuery('')}>
            ×
          </button>
        )}
      </div>

      <div className="glossary-content">
        {filteredGlossary.length === 0 ? (
          <div className="empty-state">
            <p>No fallacies match your search.</p>
          </div>
        ) : (
          filteredGlossary.map((item) => (
            <div
              key={item.id}
              className={`glossary-item ${expandedId === item.id ? 'expanded' : ''}`}
            >
              <div
                className="item-header"
                onClick={() => toggleExpand(item.id)}
              >
                <span className="item-name">{item.name}</span>
                <span className="expand-icon">{expandedId === item.id ? '−' : '+'}</span>
              </div>

              {expandedId === item.id && (
                <div className="item-details">
                  <div className="detail-section">
                    <div className="detail-label">Definition</div>
                    <div className="detail-content">
                      <ReactMarkdown>{item.definition}</ReactMarkdown>
                    </div>
                  </div>

                  {item.why_it_weakens && (
                    <div className="detail-section weakness">
                      <div className="detail-label">Why It Weakens Arguments</div>
                      <div className="detail-content">
                        <ReactMarkdown>{item.why_it_weakens}</ReactMarkdown>
                      </div>
                    </div>
                  )}

                  {item.how_to_fix && (
                    <div className="detail-section fix">
                      <div className="detail-label">How to Fix</div>
                      <div className="detail-content">
                        <ReactMarkdown>{item.how_to_fix}</ReactMarkdown>
                      </div>
                    </div>
                  )}

                  {item.example && (
                    <div className="detail-section example">
                      <div className="detail-label">Example</div>
                      <div className="detail-content example-text">
                        <ReactMarkdown>{item.example}</ReactMarkdown>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))
        )}
      </div>

      <div className="panel-footer">
        <span className="count">{filteredGlossary.length} fallacies</span>
      </div>
    </div>
  );
}

export default GlossaryPanel;
