import React, { useState, useRef, useEffect } from 'react';
import {
    Input, Button, Tag, Typography, Space, Alert,
    Select, InputNumber, Tooltip, Table
} from 'antd';
import {
    RobotOutlined, UserOutlined,
    CheckCircleOutlined, CloseCircleOutlined, LoadingOutlined,
    FileExcelOutlined, TrophyOutlined, ClearOutlined,
    SettingOutlined, SendOutlined, InfoCircleOutlined,
    TableOutlined
} from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useAppStore } from '../stores';
import { useUserStore } from '../stores/userStore';

const { Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;

// ── 类型 ──────────────────────────────────────────────────────────────────────
interface HistoryMessage { role: 'user' | 'assistant'; content: string; }
interface ToolCallRecord {
    step: number; tool: string; args: Record<string, any>;
    result?: Record<string, any>; hasError?: boolean; pending: boolean;
}
interface AssistantTurn {
    toolCalls: ToolCallRecord[]; finalText: string | null;
    isStreaming: boolean; hasError: boolean;
}
type ChatEntry = { type: 'user'; content: string } | { type: 'assistant'; turn: AssistantTurn };

// ── 工具名称映射 ───────────────────────────────────────────────────────────────
const TOOL_LABEL: Record<string, string> = {
    get_data_overview:    '数据概览 & 描述性统计',
    run_iv_report:        'IV / PSI 分析',
    filter_features:      'L2 变量筛选',
    start_training:       '启动 Optuna 建模',
    poll_task_until_done: '等待任务完成',
    generate_report:      '生成模型报告',
    get_model_metrics:    '获取最终指标',
    auto_strategy_mining: '自动化策略挖掘',
    deploy_model:         '模型上线',
    deploy_strategy:      '策略上线',
    strategy_backtest:    '策略回溯与监控',
    list_trained_models:  '查看已训练模型',
    get_score_cutoff_table: '查看分数切分表',
    get_model_report: '查看模型详细报告',
    list_active_strategies: '查看已上线策略',
};

// ── 数据概览渲染 ───────────────────────────────────────────────────────────────
const DataOverviewResult: React.FC<{ r: any }> = ({ r }) => {
    const cols     = r.columns || [];
    const ts       = r.type_summary || {};
    const ms       = r.missing_summary || {};
    const no       = r.numeric_overall || {};
    const suspects = r.suspect_exclude || [];

    // 前5行表格列定义（全部列，靠横向滚动展示）
    const headColumns = cols.map((col: string) => ({
        title: <span style={{ fontSize: 11, fontWeight: 600, whiteSpace: 'nowrap' }}>{col}</span>,
        dataIndex: col, key: col,
        width: 120,
        render: (v: any) => (
            <span style={{ fontSize: 11, color: v === null || v === undefined ? '#bfbfbf' : undefined, whiteSpace: 'nowrap' }}>
                {v === null || v === undefined ? '—' : String(v)}
            </span>
        ),
    }));
    const headData = (r.head || []).map((row: any, i: number) => ({ ...row, _key: i }));

    return (
        <div style={{ fontSize: 12 }}>

            {/* ① 基本信息 */}
            <Space wrap style={{ marginBottom: 12 }}>
                <Tag icon={<TableOutlined />} color="blue">{r.shape?.rows?.toLocaleString()} 行</Tag>
                <Tag color="purple">{r.shape?.columns} 列</Tag>
                <Tag color="geekblue">数值列 {ts.numeric_count}</Tag>
                <Tag color="cyan">字符列 {ts.object_count}</Tag>
                {ts.datetime_count > 0 && <Tag color="lime">时间列 {ts.datetime_count}</Tag>}
            </Space>

            {/* ② 前5行（横向可滚动） */}
            <div style={{ color: '#8c8c8c', fontSize: 11, marginBottom: 4 }}>📋 前 5 行数据（共 {cols.length} 列，左右滑动查看）</div>
            <div style={{ overflowX: 'auto', marginBottom: 14, border: '1px solid #f0f0f0', borderRadius: 4 }}>
                <Table
                    dataSource={headData}
                    columns={headColumns}
                    rowKey="_key"
                    size="small"
                    pagination={false}
                    scroll={{ x: cols.length * 120 }}  /* 强制横向滚动 */
                    style={{ minWidth: 300 }}
                />
            </div>

            {/* ③ 数值列整体摘要（一行卡片，不逐列展开） */}
            {Object.keys(no).length > 0 && (
                <div style={{ marginBottom: 12 }}>
                    <div style={{ color: '#8c8c8c', fontSize: 11, marginBottom: 6 }}>📊 数值列整体摘要（{ts.numeric_count} 列汇总）</div>
                    <Space wrap>
                        <Tag color="default">均值均值 {no.mean_of_means}</Tag>
                        <Tag color="default">均标准差 {no.mean_of_stds}</Tag>
                        <Tag color="default">全局最小 {no.overall_min}</Tag>
                        <Tag color="default">全局最大 {no.overall_max}</Tag>
                    </Space>
                </div>
            )}

            {/* ④ 缺失摘要 */}
            {ms.total_missing_cols > 0 ? (
                <div style={{ marginBottom: 12 }}>
                    <div style={{ color: '#8c8c8c', fontSize: 11, marginBottom: 6 }}>
                        ⚠️ 缺失情况：{ms.total_missing_cols} 列有缺失，整体缺失率 {((ms.overall_missing_rate || 0) * 100).toFixed(2)}%
                    </div>
                    <Space wrap>
                        {(ms.top_missing || []).map((item: any) => (
                            <Tag key={item.column}
                                color={item.missing_rate > 0.3 ? 'red' : item.missing_rate > 0.1 ? 'orange' : 'default'}>
                                {item.column}: {(item.missing_rate * 100).toFixed(1)}%
                            </Tag>
                        ))}
                    </Space>
                </div>
            ) : (
                <div style={{ color: '#52c41a', fontSize: 11, marginBottom: 12 }}>✅ 无缺失值</div>
            )}

            {/* ⑤ 疑似排除列推荐 */}
            {suspects.length > 0 && (
                <div style={{ background: '#fffbe6', border: '1px solid #ffe58f', borderRadius: 6, padding: '8px 12px' }}>
                    <div style={{ color: '#874d00', fontSize: 11, fontWeight: 600, marginBottom: 6 }}>
                        💡 建议排除列（请与用户确认）
                    </div>
                    <Space wrap>
                        {suspects.map((s: any) => (
                            <Tooltip key={s.column} title={`类型: ${s.dtype} | ${s.reason}`}>
                                <Tag color="warning" style={{ cursor: 'default' }}>
                                    {s.column}
                                    <Text type="secondary" style={{ fontSize: 10, marginLeft: 4 }}>
                                        {s.reason.includes('ID') ? '🔑' : '📅'}
                                    </Text>
                                </Tag>
                            </Tooltip>
                        ))}
                    </Space>
                </div>
            )}
        </div>
    );
};

// ── 工具结果摘要 ───────────────────────────────────────────────────────────────
const ToolResult: React.FC<{ tc: ToolCallRecord }> = ({ tc }) => {
    if (tc.pending) return null;
    const r = tc.result || {};
    const s: React.CSSProperties = { fontSize: 12, color: '#595959', marginTop: 2 };
    if (tc.hasError) return <Text type="danger" style={{ fontSize: 12 }}>✗ {r.error}</Text>;

    switch (tc.tool) {
        case 'get_data_overview':
            return <DataOverviewResult r={r} />;
        case 'run_iv_report':
            return <div style={s}>共 <b>{r.total_features}</b> 特征，有效（IV≥0.01）<b style={{ color: '#1677ff' }}>{r.valid_features_iv_gt_001}</b> 个</div>;
        case 'filter_features':
            return <div style={s}>保留 <b style={{ color: '#52c41a' }}>{r.kept_count}</b> 个，剔除 <b style={{ color: '#ff4d4f' }}>{r.removed_count}</b> 个</div>;
        case 'start_training':
            return <div style={s}>任务 <Text code>{r.task_id}</Text>，{r.model_type?.toUpperCase()} × {r.n_trials} trials，{r.feature_count} 特征</div>;
        case 'poll_task_until_done':
            if (r.status === 'completed') {
                const m = r.metrics || {};
                return (
                    <Space wrap style={{ fontSize: 12, marginTop: 2 }}>
                        {m.train_auc && <Tag color="blue">Train AUC {Number(m.train_auc).toFixed(4)}</Tag>}
                        {m.valid_auc && <Tag color="green">Valid AUC {Number(m.valid_auc).toFixed(4)}</Tag>}
                        {m.oot_ks   && <Tag color="orange">OOT KS {Number(m.oot_ks).toFixed(4)}</Tag>}
                        <Tag>result_id: {r.model_result_id}</Tag>
                    </Space>
                );
            }
            return <Text type="danger" style={{ fontSize: 12 }}>失败: {r.error}</Text>;
        case 'generate_report':
            return <div style={s}>报告任务 <Text code>{r.task_id}</Text></div>;
        case 'get_model_metrics':
            return <div style={s}>{r.report_file_path ? <><FileExcelOutlined style={{ color: '#52c41a' }} /> {r.report_file_path}</> : '报告生成中...'}</div>;
        case 'auto_strategy_mining':
            return <div style={s}>挖掘任务 <Text code>{r.task_id}</Text></div>;
        case 'deploy_model':
            return <div style={s}>模型已上线，部署ID：<Text code>{r.deployment_id}</Text></div>;
        case 'deploy_strategy':
            return <div style={s}>策略已上线，共 <b style={{ color: '#52c41a' }}>{r.rules_count}</b> 条规则</div>;
        case 'strategy_backtest':
            return <div style={s}>批次 {r.batch_name}：预估通过率 <b style={{ color: '#1677ff' }}>{r.approval_rate ? (r.approval_rate * 100).toFixed(2) : 0}%</b> (总样本 {r.total_count})</div>;
        case 'list_trained_models':
            return (
                <div style={s}>
                    共 <b style={{ color: '#1677ff' }}>{r.total}</b> 个模型
                    {r.models && r.models.length > 0 && (
                        <ul style={{ paddingLeft: 20, margin: '4px 0 0 0' }}>
                            {r.models.slice(0, 3).map((m: any) => (
                                <li key={m.model_result_id}>
                                    ID:{m.model_result_id} ({m.model_type}) - AUC:{m.valid_auc}
                                    {m.is_active && <b style={{ color: '#52c41a', marginLeft: 8 }}>[当前上线]</b>}
                                </li>
                            ))}
                            {r.models.length > 3 && <li>...</li>}
                        </ul>
                    )}
                </div>
            );
        case 'get_score_cutoff_table':
            return (
                <div style={s}>
                    模型 {r.model_result_id} 的分数分布已获取。
                    <div style={{ marginTop: 4, maxHeight: 150, overflowY: 'auto', border: '1px solid #f0f0f0', padding: 4 }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                            <thead>
                                <tr style={{ background: '#f5f5f5', textAlign: 'left' }}>
                                    <th style={{ padding: '4px 0' }}>概率区间 (Proba)</th>
                                    <th>坏率</th>
                                    <th>样本占比</th>
                                    <th>累计占比</th>
                                </tr>
                            </thead>
                            <tbody>
                                {(r.cutoff_table || []).slice(0, 10).map((row: any, idx: number) => {
                                    const fmt = (v: any) => v !== undefined ? Number(v).toFixed(4) : '-';
                                    return (
                                        <tr key={idx} style={{ borderBottom: '1px solid #f0f0f0' }}>
                                            <td style={{ padding: '4px 0' }}>{fmt(row.train_min)} - {fmt(row.train_max)}</td>
                                            <td>{(row.train_bad_rate * 100).toFixed(2)}%</td>
                                            <td>{((row.train_total_prop || 0) * 100).toFixed(1)}%</td>
                                            <td>{((row.train_cum_total_prop || 0) * 100).toFixed(1)}%</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            );
        case 'get_model_report':
            return (
                <div style={s}>
                    模型 {r.model_result_id} 的报告详情已加载。
                    <div style={{ marginTop: 4, fontSize: 12 }}>
                        • 数据样本: {r.data_summary?.length || 0} 个数据集<br/>
                        • 特征数量: {r.feature_importance?.length || 0} 个<br/>
                        • PSI 稳定性: {r.psi_monthly?.length || 0} 个月数据
                    </div>
                </div>
            );
        case 'list_active_strategies':
            return (
                <div style={s}>
                    当前共有 <b style={{ color: '#1677ff' }}>{r.total}</b> 条已上线策略：
                    {r.strategies && r.strategies.length > 0 && (
                        <ul style={{ paddingLeft: 20, margin: '4px 0 0 0' }}>
                            {r.strategies.map((st: any) => (
                                <li key={st.id}>
                                    <b>{st.name}</b> (优先级 {st.priority}): {st.rules?.length || 0} 条规则
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            );
        default:
            return <div style={s}>{JSON.stringify(r).slice(0, 120)}</div>;
    }
};

// ── 工具调用卡片 ───────────────────────────────────────────────────────────────
const ToolCallCard: React.FC<{ tc: ToolCallRecord }> = ({ tc }) => (
    <div style={{
        background: '#fafafa', border: '1px solid #f0f0f0', borderRadius: 6,
        padding: '6px 10px', margin: '4px 0',
    }}>
        <Space>
            {tc.pending
                ? <LoadingOutlined style={{ color: '#1677ff' }} />
                : tc.hasError
                    ? <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
                    : <CheckCircleOutlined style={{ color: '#52c41a' }} />}
            <Tag color={tc.pending ? 'processing' : tc.hasError ? 'error' : 'success'} style={{ margin: 0 }}>
                Step {tc.step}
            </Tag>
            <Text strong style={{ fontSize: 12 }}>{TOOL_LABEL[tc.tool] || tc.tool}</Text>
        </Space>
        <ToolResult tc={tc} />
    </div>
);

// ── 助手气泡 ───────────────────────────────────────────────────────────────────
const AssistantBubble: React.FC<{ turn: AssistantTurn }> = ({ turn }) => (
    <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        <div style={{
            width: 36, height: 36, borderRadius: '50%', flexShrink: 0,
            background: 'linear-gradient(135deg, #1677ff, #722ed1)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
            <RobotOutlined style={{ color: '#fff', fontSize: 18 }} />
        </div>
        <div style={{ flex: 1, maxWidth: 'calc(100% - 46px)' }}>
            {turn.toolCalls.length > 0 && (
                <div style={{ marginBottom: turn.finalText ? 8 : 0 }}>
                    {turn.toolCalls.map((tc, i) => <ToolCallCard key={i} tc={tc} />)}
                </div>
            )}
            {turn.finalText && (
                <div className="markdown-body" style={{
                    background: '#fff', border: '1px solid #e8e8e8', borderRadius: 8, padding: '10px 14px',
                    fontSize: 14, lineHeight: '1.6'
                }}>
                    <style>{`
                        .markdown-body table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
                        .markdown-body th, .markdown-body td { border: 1px solid #dfe2e5; padding: 6px 13px; }
                        .markdown-body tr:nth-child(2n) { background-color: #f6f8fa; }
                        .markdown-body ul, .markdown-body ol { padding-left: 2em; margin-bottom: 16px; }
                        .markdown-body p { margin-bottom: 10px; }
                        .markdown-body code { background-color: rgba(27,31,35,.05); border-radius: 3px; padding: .2em .4em; font-family: SFMono-Regular,Consolas,Liberation Mono,Menlo,monospace; font-size: 85%; }
                        .markdown-body pre { background-color: #f6f8fa; border-radius: 3px; padding: 16px; overflow: auto; line-height: 1.45; }
                    `}</style>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                        {turn.finalText.includes('完成') && <TrophyOutlined style={{ color: '#52c41a', marginTop: 4 }} />}
                        <div style={{ flex: 1, overflow: 'hidden' }}>
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {turn.finalText}
                            </ReactMarkdown>
                        </div>
                    </div>
                </div>
            )}
            {turn.isStreaming && turn.toolCalls.length === 0 && !turn.finalText && (
                <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: 8, padding: '10px 14px' }}>
                    <LoadingOutlined style={{ marginRight: 8 }} />
                    <Text type="secondary">思考中...</Text>
                </div>
            )}
            {turn.hasError && !turn.isStreaming && (
                <Alert type="error" showIcon message="本轮执行中断" style={{ marginTop: 8 }} />
            )}
        </div>
    </div>
);

// ── 用户气泡 ───────────────────────────────────────────────────────────────────
const UserBubble: React.FC<{ content: string }> = ({ content }) => (
    <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexDirection: 'row-reverse' }}>
        <div style={{
            width: 36, height: 36, borderRadius: '50%', flexShrink: 0,
            background: '#e6f4ff',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
            <UserOutlined style={{ color: '#1677ff', fontSize: 18 }} />
        </div>
        <div style={{ background: '#e6f4ff', borderRadius: 8, padding: '10px 14px', maxWidth: '75%' }}>
            <Text style={{ whiteSpace: 'pre-wrap' }}>{content}</Text>
        </div>
    </div>
);

// ── 主页面 ─────────────────────────────────────────────────────────────────────
const AgentPage: React.FC = () => {
    const { currentProjectId, currentDatasetId, excludeCols } = useAppStore();
    const { token } = useUserStore();

    const [dep, setDep]             = useState('label');
    const [modelType, setModelType] = useState('xgb');
    const [nTrials, setNTrials]     = useState<number>(30);
    const [showSettings, setShowSettings] = useState(true);

    const [chatEntries, setChatEntries] = useState<ChatEntry[]>([]);
    const [historyMsgs, setHistoryMsgs] = useState<HistoryMessage[]>([]);
    const [inputText, setInputText]     = useState('');
    const [isStreaming, setIsStreaming]  = useState(false);

    const bottomRef = useRef<HTMLDivElement>(null);
    const abortRef  = useRef<AbortController | null>(null);

    // 当项目或数据集切换时，清空历史，避免旧的上下文 (包含旧ID) 被发送给 LLM
    useEffect(() => {
        setChatEntries([]);
        setHistoryMsgs([]);
    }, [currentProjectId, currentDatasetId]);

    useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [chatEntries]);

    const isReady = !!currentProjectId && !!currentDatasetId;

    const handleSend = async () => {
        const text = inputText.trim();
        if (!text || !isReady || isStreaming) return;
        setInputText('');
        setIsStreaming(true);

        setChatEntries(prev => [...prev, { type: 'user', content: text }]);
        setChatEntries(prev => [...prev, { type: 'assistant', turn: { toolCalls: [], finalText: null, isStreaming: true, hasError: false } }]);

        const tcBuffer: ToolCallRecord[] = [];
        let finalText = '';
        let stepCounter = 0;

        const updateLast = (fn: (t: AssistantTurn) => AssistantTurn) => {
            setChatEntries(prev => {
                const copy = [...prev];
                const last = copy[copy.length - 1];
                if (last.type === 'assistant') copy[copy.length - 1] = { type: 'assistant', turn: fn(last.turn) };
                return copy;
            });
        };

        abortRef.current = new AbortController();
        try {
            const resp = await fetch('/api/agent/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                body: JSON.stringify({
                    project_id: currentProjectId, dataset_id: currentDatasetId,
                    dep, model_type: modelType, n_trials: nTrials,
                    exclude_cols: excludeCols || [],
                    messages: historyMsgs, user_message: text,
                }),
                signal: abortRef.current.signal,
            });

            if (!resp.ok) { updateLast(t => ({ ...t, isStreaming: false, hasError: true, finalText: `请求失败: HTTP ${resp.status}` })); return; }

            const reader = resp.body!.getReader();
            const decoder = new TextDecoder();
            let buf = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                const lines = buf.split('\n');
                buf = lines.pop() || '';

                for (const line of lines) {
                    const t = line.trim();
                    if (!t.startsWith('data: ')) continue;
                    let frame: any;
                    try { frame = JSON.parse(t.slice(6)); } catch { continue; }

                    if (frame.type === 'tool_start') {
                        stepCounter++;
                        const tc: ToolCallRecord = { step: frame.step ?? stepCounter, tool: frame.tool, args: frame.args || {}, pending: true };
                        tcBuffer.push(tc);
                        updateLast(turn => ({ ...turn, toolCalls: [...tcBuffer] }));
                    }
                    if (frame.type === 'tool_done') {
                        const idx = tcBuffer.findIndex(tc => tc.step === frame.step);
                        if (idx >= 0) { tcBuffer[idx] = { ...tcBuffer[idx], result: frame.result, hasError: !!frame.has_error, pending: false }; }
                        updateLast(turn => ({ ...turn, toolCalls: [...tcBuffer], hasError: turn.hasError || !!frame.has_error }));
                    }
                    if (frame.type === 'llm_thought') { finalText = frame.content || ''; updateLast(t => ({ ...t, finalText })); }
                    if (frame.type === 'finished') {
                        finalText = frame.summary || frame.assistant_message || '';
                        updateLast(t => ({ ...t, finalText, isStreaming: false }));
                        setHistoryMsgs(prev => [...prev,
                            { role: 'user', content: text },
                            { role: 'assistant', content: finalText },
                        ]);
                    }
                    if (frame.type === 'error') { updateLast(t => ({ ...t, isStreaming: false, hasError: true, finalText: frame.message })); }
                }
            }
        } catch (err: any) {
            if (err.name !== 'AbortError') updateLast(t => ({ ...t, isStreaming: false, hasError: true, finalText: `连接异常: ${err.message}` }));
        } finally {
            setIsStreaming(false);
            updateLast(t => ({ ...t, isStreaming: false }));
        }
    };

    const handleStop  = () => { abortRef.current?.abort(); setIsStreaming(false); };
    const handleClear = () => { setChatEntries([]); setHistoryMsgs([]); };
    const handleKey   = (e: React.KeyboardEvent) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } };

    const QUICK_CMDS = ['请开始完整建模流程', '帮我看看数据', '做描述性统计', '帮我做 IV 分析', '执行变量筛选', '开始训练模型', '生成模型报告'];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 120px)' }}>

            {/* 顶栏 */}
            <div style={{ padding: '12px 16px', background: '#fff', borderBottom: '1px solid #f0f0f0', display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
                <RobotOutlined style={{ fontSize: 20, color: '#1677ff' }} />
                <span style={{ fontWeight: 600, fontSize: 16 }}>智能建模 Agent</span>
                <div style={{ flex: 1 }} />
                {currentProjectId ? <Tag color="blue">项目 {currentProjectId}</Tag> : <Tag>未选项目</Tag>}
                {currentDatasetId ? <Tag color="green">数据集 {currentDatasetId}</Tag> : <Tag>未上传数据</Tag>}
                <Tooltip title="参数配置">
                    <Button icon={<SettingOutlined />} size="small" type={showSettings ? 'primary' : 'default'} onClick={() => setShowSettings(v => !v)} />
                </Tooltip>
                <Tooltip title="清空对话">
                    <Button icon={<ClearOutlined />} size="small" onClick={handleClear} disabled={isStreaming} />
                </Tooltip>
            </div>

            {/* 参数配置条 */}
            {showSettings && (
                <div style={{ padding: '10px 16px', background: '#fafafa', borderBottom: '1px solid #f0f0f0', flexShrink: 0, display: 'flex', gap: 20, flexWrap: 'wrap', alignItems: 'center' }}>
                    <Space size={4}><Text type="secondary" style={{ fontSize: 12 }}>标签列</Text>
                        <Input value={dep} onChange={e => setDep(e.target.value)} style={{ width: 90 }} size="small" disabled={isStreaming} /></Space>
                    <Space size={4}><Text type="secondary" style={{ fontSize: 12 }}>模型</Text>
                        <Select value={modelType} onChange={setModelType} style={{ width: 120 }} size="small" disabled={isStreaming}>
                            <Option value="xgb">XGBoost</Option><Option value="lgb">LightGBM</Option>
                            <Option value="lr">逻辑回归</Option><Option value="rf">随机森林</Option>
                        </Select></Space>
                    <Space size={4}><Text type="secondary" style={{ fontSize: 12 }}>Optuna 次数</Text>
                        <InputNumber value={nTrials} onChange={v => setNTrials(v ?? 30)} min={5} max={300} size="small" style={{ width: 70 }} disabled={isStreaming} /></Space>
                    <Text type="secondary" style={{ fontSize: 11, marginLeft: 'auto' }}>
                        <InfoCircleOutlined /> 首轮对话时这些参数自动注入上下文
                    </Text>
                </div>
            )}

            {/* 消息列表 */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px 16px', background: '#f5f5f5' }}>
                {chatEntries.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '60px 0' }}>
                        {!isReady
                            ? <Alert type="warning" showIcon message="请先在「项目管理」选择项目，并在「数据管理」上传数据集" style={{ maxWidth: 500, margin: '0 auto' }} />
                            : (<>
                                <RobotOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
                                <div style={{ color: '#8c8c8c', marginBottom: 20 }}>你好！我是 AutoModeling 智能建模助手。开始建模前我会先帮你查看数据并确认排除列。</div>
                                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
                                    {QUICK_CMDS.map(cmd => (
                                        <Tag key={cmd} style={{ cursor: 'pointer', padding: '4px 10px', borderRadius: 12 }} onClick={() => setInputText(cmd)}>{cmd}</Tag>
                                    ))}
                                </div>
                            </>)
                        }
                    </div>
                ) : (
                    chatEntries.map((entry, i) =>
                        entry.type === 'user' ? <UserBubble key={i} content={entry.content} /> : <AssistantBubble key={i} turn={entry.turn} />
                    )
                )}
                <div ref={bottomRef} />
            </div>

            {/* 快捷指令行 */}
            {chatEntries.length > 0 && (
                <div style={{ padding: '6px 16px', background: '#fafafa', borderTop: '1px solid #f0f0f0', flexShrink: 0, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {QUICK_CMDS.slice(1).map(cmd => (
                        <Tag key={cmd} style={{ cursor: 'pointer', borderRadius: 10, fontSize: 11 }} onClick={() => { if (!isStreaming) setInputText(cmd); }}>{cmd}</Tag>
                    ))}
                </div>
            )}

            {/* 输入区 */}
            <div style={{ padding: '12px 16px', background: '#fff', borderTop: '1px solid #f0f0f0', flexShrink: 0, display: 'flex', gap: 10, alignItems: 'flex-end' }}>
                <TextArea
                    value={inputText} onChange={e => setInputText(e.target.value)} onKeyDown={handleKey}
                    placeholder={isReady ? '输入消息... (Enter 发送，Shift+Enter 换行)' : '请先选择项目和数据集'}
                    autoSize={{ minRows: 1, maxRows: 5 }} disabled={!isReady || isStreaming}
                    style={{ flex: 1, borderRadius: 8, resize: 'none' }}
                />
                {isStreaming
                    ? <Button danger onClick={handleStop} style={{ borderRadius: 8 }}>停止</Button>
                    : <Button type="primary" icon={<SendOutlined />} onClick={handleSend} disabled={!isReady || !inputText.trim()} style={{ borderRadius: 8 }}>发送</Button>
                }
            </div>
        </div>
    );
};

export default AgentPage;
