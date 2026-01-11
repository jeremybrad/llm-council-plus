import { useState } from 'react';
import { getModelVisuals, getShortModelName } from '../utils/modelHelpers';
import ThinkBlockRenderer from './ThinkBlockRenderer';
import StageTimer from './StageTimer';
import './RoundtableTimeline.css';

/**
 * RoundtableTimeline - Displays multi-round deliberation results
 *
 * Shows:
 * - Round 1 (Openings) - Each agent's initial response
 * - Round 2 (Critiques) - Each agent critiques others
 * - Round 3 (Revisions) - Each agent revises based on feedback
 * - Moderator Summary - Decision ledger & integrated plan
 * - Chair Final - Final synthesis
 */
export default function RoundtableTimeline({
    roundtable,
    startTime,
    endTime
}) {
    const [expandedRounds, setExpandedRounds] = useState({ 1: true, 2: false, 3: false });
    const [expandedAgents, setExpandedAgents] = useState({});

    if (!roundtable) {
        return null;
    }

    const { rounds, moderator, chair_final, council_members, status } = roundtable;
    const isComplete = status === 'completed';
    const isAborted = status === 'aborted';

    const toggleRound = (roundNum) => {
        setExpandedRounds(prev => ({
            ...prev,
            [roundNum]: !prev[roundNum]
        }));
    };

    const toggleAgent = (roundNum, agentId) => {
        const key = `${roundNum}-${agentId}`;
        setExpandedAgents(prev => ({
            ...prev,
            [key]: !prev[key]
        }));
    };

    const roundLabels = {
        1: 'Opening Statements',
        2: 'Critiques & Questions',
        3: 'Revisions'
    };

    const roundIcons = {
        1: '🎤',
        2: '🔍',
        3: '✏️'
    };

    return (
        <div className="roundtable-timeline">
            <div className="roundtable-header">
                <div className="roundtable-title">
                    <span className="roundtable-icon">🎯</span>
                    Roundtable Deliberation
                    {isAborted && <span className="status-badge aborted">Aborted</span>}
                    {isComplete && <span className="status-badge complete">Complete</span>}
                </div>
                <StageTimer startTime={startTime} endTime={endTime} label="Total" />
            </div>

            {/* Rounds Timeline */}
            <div className="rounds-container">
                {rounds && rounds.map((round, idx) => {
                    const roundNum = idx + 1;
                    const isExpanded = expandedRounds[roundNum];

                    return (
                        <div key={roundNum} className={`round-section ${isExpanded ? 'expanded' : ''}`}>
                            <button
                                className="round-header"
                                onClick={() => toggleRound(roundNum)}
                            >
                                <div className="round-title">
                                    <span className="round-icon">{roundIcons[roundNum] || '📝'}</span>
                                    <span className="round-label">
                                        Round {roundNum}: {roundLabels[roundNum] || 'Discussion'}
                                    </span>
                                    <span className="response-count">
                                        {round.responses?.length || 0} responses
                                    </span>
                                </div>
                                <span className={`expand-icon ${isExpanded ? 'expanded' : ''}`}>
                                    ▼
                                </span>
                            </button>

                            {isExpanded && (
                                <div className="round-content">
                                    {round.responses?.map((response, respIdx) => {
                                        const agentKey = `${roundNum}-${respIdx}`;
                                        const isAgentExpanded = expandedAgents[agentKey] ?? true;
                                        const visuals = getModelVisuals(response.model);
                                        const shortName = getShortModelName(response.model);
                                        const hasError = response.error;

                                        return (
                                            <div
                                                key={respIdx}
                                                className={`agent-response ${hasError ? 'has-error' : ''}`}
                                            >
                                                <button
                                                    className="agent-header"
                                                    onClick={() => toggleAgent(roundNum, respIdx)}
                                                >
                                                    <div className="agent-identity">
                                                        <span
                                                            className="agent-avatar"
                                                            style={{ backgroundColor: hasError ? '#ef4444' : visuals.color }}
                                                        >
                                                            {visuals.icon}
                                                        </span>
                                                        <div className="agent-info">
                                                            <span className="agent-name">{shortName}</span>
                                                            {response.role && (
                                                                <span className="agent-role">{response.role}</span>
                                                            )}
                                                        </div>
                                                    </div>
                                                    <span className={`expand-icon ${isAgentExpanded ? 'expanded' : ''}`}>
                                                        ▼
                                                    </span>
                                                </button>

                                                {isAgentExpanded && (
                                                    <div className="agent-content">
                                                        {hasError ? (
                                                            <div className="response-error">
                                                                <span className="error-icon">⚠️</span>
                                                                <span className="error-text">
                                                                    {response.error_message || 'Failed to respond'}
                                                                </span>
                                                            </div>
                                                        ) : (
                                                            <div className="response-text markdown-content">
                                                                <ThinkBlockRenderer
                                                                    content={
                                                                        typeof response.content === 'string'
                                                                            ? response.content
                                                                            : String(response.content || 'No response')
                                                                    }
                                                                />
                                                            </div>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Moderator Summary */}
            {moderator && (
                <div className="synthesis-section moderator-section">
                    <div className="synthesis-header">
                        <span className="synthesis-icon">📋</span>
                        <span className="synthesis-title">Moderator Summary</span>
                    </div>
                    <div className="synthesis-content markdown-content">
                        <ThinkBlockRenderer
                            content={
                                typeof moderator.content === 'string'
                                    ? moderator.content
                                    : String(moderator.content || '')
                            }
                        />
                    </div>
                </div>
            )}

            {/* Chair Final Synthesis */}
            {chair_final && (
                <div className="synthesis-section chair-section">
                    <div className="synthesis-header">
                        <span className="synthesis-icon">🎯</span>
                        <span className="synthesis-title">Chair's Final Synthesis</span>
                    </div>
                    <div className="synthesis-content markdown-content">
                        <ThinkBlockRenderer
                            content={
                                typeof chair_final.content === 'string'
                                    ? chair_final.content
                                    : String(chair_final.content || '')
                            }
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
