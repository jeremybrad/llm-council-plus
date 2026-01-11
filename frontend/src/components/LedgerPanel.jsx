import { useState } from 'react';
import './LedgerPanel.css';

/**
 * LedgerPanel - Displays the accumulated Socratic ledger
 *
 * Shows definitions, commitments, assumptions, counterexamples,
 * contradictions, and open questions with their status.
 */
function LedgerPanel({ ledger, onClose }) {
  const [expandedSections, setExpandedSections] = useState({
    definitions: true,
    commitments: true,
    assumptions: true,
    counterexamples: false,
    contradictions: false,
    open_questions: true,
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  if (!ledger) {
    return (
      <div className="ledger-panel">
        <div className="panel-header">
          <h3>Commitment Ledger</h3>
          {onClose && <button className="close-btn" onClick={onClose}>×</button>}
        </div>
        <div className="empty-state">
          <p>No ledger data yet. Start the inquiry to build commitments.</p>
        </div>
      </div>
    );
  }

  /**
   * Get items filtered by status
   */
  const getActiveItems = (items) => {
    if (!items) return [];
    return items.filter(item => item.status === 'active' || !item.status);
  };

  const getSupersededItems = (items) => {
    if (!items) return [];
    return items.filter(item => item.status === 'superseded');
  };

  const getRetractedItems = (items) => {
    if (!items) return [];
    return items.filter(item => item.status === 'retracted');
  };

  /**
   * Render a section of the ledger
   */
  const renderSection = (title, key, items, renderItem) => {
    const active = getActiveItems(items);
    const superseded = getSupersededItems(items);
    const retracted = getRetractedItems(items);
    const total = (items || []).length;

    if (total === 0) return null;

    return (
      <div className="ledger-section">
        <div
          className={`section-header ${expandedSections[key] ? 'expanded' : ''}`}
          onClick={() => toggleSection(key)}
        >
          <span className="section-title">{title}</span>
          <span className="section-count">
            {active.length} active
            {superseded.length > 0 && ` / ${superseded.length} superseded`}
            {retracted.length > 0 && ` / ${retracted.length} retracted`}
          </span>
          <span className="expand-icon">{expandedSections[key] ? '−' : '+'}</span>
        </div>
        {expandedSections[key] && (
          <div className="section-content">
            {active.map((item, i) => (
              <div key={item.id || i} className="ledger-item active">
                {renderItem(item)}
              </div>
            ))}
            {superseded.length > 0 && (
              <div className="superseded-group">
                <div className="group-label">Superseded</div>
                {superseded.map((item, i) => (
                  <div key={item.id || i} className="ledger-item superseded">
                    {renderItem(item)}
                    {item.superseded_by_id && (
                      <span className="superseded-by">→ {item.superseded_by_id}</span>
                    )}
                  </div>
                ))}
              </div>
            )}
            {retracted.length > 0 && (
              <div className="retracted-group">
                <div className="group-label">Retracted</div>
                {retracted.map((item, i) => (
                  <div key={item.id || i} className="ledger-item retracted">
                    {renderItem(item)}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="ledger-panel">
      <div className="panel-header">
        <h3>Commitment Ledger</h3>
        {onClose && <button className="close-btn" onClick={onClose}>×</button>}
      </div>

      <div className="ledger-content">
        {/* Inquiry and Thesis at the top */}
        {(ledger.inquiry || ledger.thesis) && (
          <div className="ledger-meta">
            {ledger.inquiry && (
              <div className="meta-item inquiry">
                <span className="meta-label">Inquiry:</span>
                <span className="meta-value">{ledger.inquiry}</span>
              </div>
            )}
            {ledger.thesis && (
              <div className="meta-item thesis">
                <span className="meta-label">Thesis:</span>
                <span className="meta-value">{ledger.thesis}</span>
              </div>
            )}
          </div>
        )}

        {/* Definitions */}
        {renderSection('Definitions', 'definitions', ledger.definitions, (item) => (
          <>
            <div className="item-header">
              <span className="item-id">[{item.id}]</span>
              <span className="item-term">{item.term}</span>
              {item.confidence && (
                <span className="confidence" title="Confidence level">
                  {Math.round(item.confidence * 100)}%
                </span>
              )}
            </div>
            <div className="item-body">{item.definition}</div>
          </>
        ))}

        {/* Commitments */}
        {renderSection('Commitments', 'commitments', ledger.commitments, (item) => (
          <>
            <div className="item-header">
              <span className="item-id">[{item.id}]</span>
              {item.source && <span className="item-source">{item.source}</span>}
            </div>
            <div className="item-body">{item.text}</div>
          </>
        ))}

        {/* Assumptions */}
        {renderSection('Assumptions', 'assumptions', ledger.assumptions, (item) => (
          <>
            <div className="item-header">
              <span className="item-id">[{item.id}]</span>
            </div>
            <div className="item-body">{item.text}</div>
          </>
        ))}

        {/* Counterexamples */}
        {renderSection('Counterexamples', 'counterexamples', ledger.counterexamples, (item) => (
          <>
            <div className="item-header">
              <span className="item-id">[{item.id}]</span>
              {item.challenges_id && (
                <span className="challenges">challenges: {item.challenges_id}</span>
              )}
            </div>
            <div className="item-body">{item.text}</div>
          </>
        ))}

        {/* Contradictions */}
        {renderSection('Contradictions', 'contradictions', ledger.contradictions, (item) => (
          <>
            <div className="item-header">
              <span className="item-id">[{item.id}]</span>
              {item.item_ids && (
                <span className="between">between: {item.item_ids.join(', ')}</span>
              )}
            </div>
            <div className="item-body">{item.text}</div>
          </>
        ))}

        {/* Open Questions */}
        {renderSection('Open Questions', 'open_questions', ledger.open_questions, (item) => (
          <>
            <div className="item-header">
              <span className="item-id">[{item.id}]</span>
            </div>
            <div className="item-body">{item.text}</div>
          </>
        ))}

        {/* Empty state if no content */}
        {!ledger.definitions?.length &&
          !ledger.commitments?.length &&
          !ledger.assumptions?.length &&
          !ledger.counterexamples?.length &&
          !ledger.contradictions?.length &&
          !ledger.open_questions?.length && (
            <div className="empty-state">
              <p>Ledger is empty. Commitments will appear as the dialogue progresses.</p>
            </div>
          )}
      </div>
    </div>
  );
}

export default LedgerPanel;
