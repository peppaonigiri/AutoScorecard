import React, { useEffect, useState } from 'react';
import { Card, Table, Descriptions, Typography, Row, Col, message, Switch, Space, Tag, Button, Modal, Tabs, Progress } from 'antd';
import { CheckCircleOutlined, FileTextOutlined, DownloadOutlined } from '@ant-design/icons';
import api from '../services/api';
import { useAppStore } from '../stores';

const { Title } = Typography;

import ReactECharts from 'echarts-for-react';

const ResultPage: React.FC = () => {
    const currentProjectId = useAppStore((state) => state.currentProjectId);
    const [results, setResults] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [reportModalVisible, setReportModalVisible] = useState(false);
    const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
    const [reportData, setReportData] = useState<any>(null);
    const [reportLoading, setReportLoading] = useState(false);
    const [genProgress, setGenProgress] = useState(0);
    const [genStatus, setGenStatus] = useState<string | null>(null);
    const [genTaskId, setGenTaskId] = useState<number | null>(null);

    useEffect(() => {
        if (currentProjectId) {
            fetchResults();
        }
    }, [currentProjectId]);

    const toggleModelStatus = async (resultId: number, checked: boolean) => {
        if (!currentProjectId) return;
        try {
            await api.patch(`/projects/${currentProjectId}/models/${resultId}/status`, {
                status: checked ? 'active' : 'inactive'
            });
            message.success(`模型已${checked ? '上线激活' : '下线停用'}`);
            fetchResults();
        } catch (err: any) {
            message.error('切换状态失败');
        }
    };

    const fetchResults = async () => {
        setLoading(true);
        try {
            const res: any = await api.get(`/projects/${currentProjectId}/results`);
            setResults(res.items || []);
        } catch (err: any) {
            const detail = err.response?.data?.detail || '获取结果列表失败';
            message.error(`结果获取异常: ${detail}`);
            console.error('Fetch results error:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleViewReport = async (modelId: number) => {
        setSelectedModelId(modelId);
        setReportModalVisible(true);
        setReportData(null);
        setReportLoading(true);
        try {
            const res: any = await api.get(`/projects/${currentProjectId}/models/${modelId}/report`);
            setReportData(res);
        } catch (err: any) {
            if (err.response?.status === 404) {
                // message.info('报告尚未生成，正在请求生成...');
                // tryGenerateReport(modelId);
            } else {
                message.error('获取报告失败');
            }
        } finally {
            setReportLoading(false);
        }
    };

    const tryGenerateReport = async (modelId: number) => {
        setReportLoading(true);
        setGenProgress(0);
        setGenStatus('正在准备任务...');
        try {
            const res: any = await api.post(`/projects/${currentProjectId}/models/${modelId}/report`);
            const taskId = res.task_id;
            setGenTaskId(taskId);

            // 开始轮询进度
            const poll = setInterval(async () => {
                try {
                    const task: any = await api.get(`/tasks/${taskId}`);
                    setGenProgress(Math.floor(task.progress));
                    setGenStatus(task.progress_data?.message || '正在计算中...');

                    if (task.status === 'completed') {
                        clearInterval(poll);
                        setGenTaskId(null);
                        message.success('报告生成成功！');
                        handleViewReport(modelId); // 自动加载
                    } else if (task.status === 'failed') {
                        clearInterval(poll);
                        setGenTaskId(null);
                        setReportLoading(false);
                        message.error(`生成失败: ${task.error_msg}`);
                    }
                } catch (e) {
                    clearInterval(poll);
                    setGenTaskId(null);
                    setReportLoading(false);
                }
            }, 1000);
        } catch (err: any) {
            message.error('提交生成任务失败');
            setReportLoading(false);
        }
    };

    const downloadReport = () => {
        if (!selectedModelId) return;
        window.open(`/api/projects/${currentProjectId}/models/${selectedModelId}/report/download`, '_blank');
    };

    const getScoreChartOption = (dist: any) => {
        if (!dist || !dist.bins) return {};

        // bins 是边际，我们需要区间中心或区间标签
        const labels = dist.bins.slice(0, -1).map((b: number, i: number) => {
            return `${Math.round(b)}-${Math.round(dist.bins[i + 1])}`;
        });

        return {
            title: { text: '分数分布对比 (Good vs Bad)', left: 'center', textStyle: { fontSize: 14 } },
            tooltip: { trigger: 'axis' },
            legend: { top: 30, data: ['好样本 (0)', '坏样本 (1)'] },
            grid: { top: 70, left: '3%', right: '4%', bottom: '3%', containLabel: true },
            xAxis: { type: 'category', data: labels, axisLabel: { rotate: 45 } },
            yAxis: { type: 'value', name: '样本数' },
            series: [
                {
                    name: '好样本 (0)',
                    type: 'bar',
                    data: dist.counts_good,
                    itemStyle: { color: '#52c41a' },
                    emphasis: { focus: 'series' },
                },
                {
                    name: '坏样本 (1)',
                    type: 'bar',
                    data: dist.counts_bad,
                    itemStyle: { color: '#ff4d4f' },
                    emphasis: { focus: 'series' },
                }
            ]
        };
    };

    if (!currentProjectId) return <div>请先选择项目</div>;

    return (
        <div>
            <Title level={4}>模型结果报告</Title>

            {results.length === 0 && !loading && (
                <Card>暂无训练完成的模型，请先前往「模型训练」页面执行任务。</Card>
            )}

            {results.map((result) => {
                // 构建 metrics 列表
                const metricsData = Object.entries(result.metrics).map(([k, v]: any) => ({
                    metric: k,
                    value: typeof v === 'number' ? v.toFixed(4) : v,
                }));

                // 构建 feature importance 列表
                const impData = Object.entries(result.feature_importance)
                    .map(([feat, imp]: any) => ({ feat, imp }))
                    .sort((a, b) => b.imp - a.imp);

                return (
                    <Card
                        key={result.id}
                        title={`模型版本 #${result.id} (${result.model_type})`}
                        style={{ marginBottom: 24 }}
                        extra={
                            <Space size="middle">
                                {result.is_active && <Tag icon={<CheckCircleOutlined />} color="success">当前在线 (Active)</Tag>}
                                <Switch
                                    checked={result.is_active}
                                    checkedChildren="上线"
                                    unCheckedChildren="离线"
                                    onChange={(checked) => toggleModelStatus(result.id, checked)}
                                />
                                <Button
                                    icon={<FileTextOutlined />}
                                    onClick={() => handleViewReport(result.id)}
                                >
                                    查看报表
                                </Button>
                            </Space>
                        }
                    >
                        <Descriptions bordered column={3} size="small" style={{ marginBottom: 24 }}>
                            <Descriptions.Item label="训练时间">{new Date(result.created_at).toLocaleString()}</Descriptions.Item>
                            <Descriptions.Item label="评分卡参数">
                                {result.score_config ? `Base:${result.score_config.base_score} | PDO:${result.score_config.pdo}` : '未配置'}
                            </Descriptions.Item>
                            <Descriptions.Item label="保留特征数">{result.feature_list?.length}</Descriptions.Item>
                            <Descriptions.Item label="Optuna策略">Type {result.optuna_strategy}</Descriptions.Item>
                            <Descriptions.Item label="搜索次数">{result.n_trials}</Descriptions.Item>
                            <Descriptions.Item label="使用的超参" span={1}>
                                <pre style={{ margin: 0, fontSize: 12, maxHeight: 100, overflow: 'auto' }}>{JSON.stringify(result.params, null, 2)}</pre>
                            </Descriptions.Item>
                        </Descriptions>

                        <Row gutter={24}>
                            <Col span={8}>
                                <Card type="inner" title="核心指标 (Metrics)" size="small">
                                    <Table
                                        dataSource={metricsData}
                                        columns={[
                                            { title: '指标', dataIndex: 'metric', key: 'metric' },
                                            { title: '得分', dataIndex: 'value', key: 'value', render: val => <strong>{val}</strong> }
                                        ]}
                                        rowKey="metric"
                                        pagination={false}
                                        size="small"
                                    />
                                </Card>
                            </Col>
                            <Col span={6}>
                                <Card type="inner" title="特征重要性 (Top 10)" size="small">
                                    <Table
                                        dataSource={impData.slice(0, 10)}
                                        columns={[
                                            { title: '特征名', dataIndex: 'feat', key: 'feat' },
                                            { title: 'Imp', dataIndex: 'imp', key: 'imp', render: val => val.toFixed(3) }
                                        ]}
                                        rowKey="feat"
                                        pagination={false}
                                        size="small"
                                    />
                                </Card>
                            </Col>
                            <Col span={10}>
                                <Card type="inner" title="分数分布图" size="small">
                                    {result.score_distribution && result.score_distribution.bins ? (
                                        <ReactECharts
                                            option={getScoreChartOption(result.score_distribution)}
                                            style={{ height: 300, width: '100%' }}
                                        />
                                    ) : (
                                        <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#999' }}>
                                            该模型版本暂无分数分布数据
                                        </div>
                                    )}
                                </Card>
                            </Col>
                        </Row>
                    </Card>
                );
            })}

            <Modal
                title={`模型报告详情 #${selectedModelId}`}
                open={reportModalVisible}
                onCancel={() => setReportModalVisible(false)}
                width={1200}
                footer={[
                    <Button key="close" onClick={() => setReportModalVisible(false)}>关闭</Button>,
                    reportData && (
                        <Button key="download" type="primary" icon={<DownloadOutlined />} onClick={downloadReport}>
                            导出 Excel
                        </Button>
                    ),
                    !reportData && !reportLoading && (
                        <Button key="generate" type="primary" onClick={() => tryGenerateReport(selectedModelId!)}>
                            生成报告
                        </Button>
                    )
                ]}
            >
                {reportLoading && !reportData ? (
                    <div style={{ padding: '60px 0', textAlign: 'center' }}>
                        {genTaskId ? (
                            <div style={{ maxWidth: 400, margin: '0 auto' }}>
                                <Progress percent={genProgress} status="active" />
                                <div style={{ marginTop: 16, color: '#666' }}>{genStatus}</div>
                            </div>
                        ) : (
                            <div>数据加载中...</div>
                        )}
                    </div>
                ) : reportData ? (
                    <Tabs defaultActiveKey="summary">
                        <Tabs.TabPane tab="数据概要" key="summary">
                            <Table
                                dataSource={reportData.data_summary}
                                columns={[
                                    { title: '数据集', dataIndex: 'dataset', key: 'dataset' },
                                    { title: '样本量', dataIndex: 'count', key: 'count' },
                                    { title: '坏账率', dataIndex: 'badrate', key: 'badrate', render: val => (val * 100).toFixed(2) + '%' },
                                    { title: '最小月份', dataIndex: 'month_min', key: 'month_min' },
                                    { title: '最大月份', dataIndex: 'month_max', key: 'month_max' },
                                ]}
                                pagination={false}
                                size="small"
                            />
                        </Tabs.TabPane>
                        <Tabs.TabPane tab="模型性能" key="performance">
                            <Table
                                dataSource={reportData.performance_eval}
                                columns={[
                                    { title: '数据集', dataIndex: 'datasets', key: 'datasets', fixed: 'left' },
                                    { title: 'AUC', dataIndex: 'auc', key: 'auc', render: val => val?.toFixed(4) },
                                    { title: 'KS', dataIndex: 'ks', key: 'ks', render: (val) => <strong style={{ color: '#1890ff' }}>{val?.toFixed(4)}</strong> },
                                    { title: 'Top 1% Lift', dataIndex: 'top_1_lift', key: 'top_1_lift', render: val => val?.toFixed(2) },
                                    { title: 'Top 2% Lift', dataIndex: 'top_2_lift', key: 'top_2_lift', render: val => val?.toFixed(2) },
                                    { title: 'Top 3% Lift', dataIndex: 'top_3_lift', key: 'top_3_lift', render: val => val?.toFixed(2) },
                                    { title: 'Top 5% Lift', dataIndex: 'top_5_lift', key: 'top_5_lift', render: val => val?.toFixed(2) },
                                    { title: 'Top 10% Lift', dataIndex: 'top_10_lift', key: 'top_10_lift', render: val => val?.toFixed(2) },
                                    { title: 'Top 20% Lift', dataIndex: 'top_20_lift', key: 'top_20_lift', render: val => val?.toFixed(2) },
                                ]}
                                pagination={false}
                                size="small"
                                scroll={{ x: 1000 }}
                            />
                        </Tabs.TabPane>
                        <Tabs.TabPane tab="特征重要性" key="importance">
                            <Table
                                dataSource={reportData.feature_importance}
                                columns={[
                                    { title: '特征名', dataIndex: 'var_name', key: 'var_name', fixed: 'left' },
                                    { title: '重要性', dataIndex: 'importance', key: 'importance', render: val => val?.toFixed(4) },
                                    { title: 'Train IV', dataIndex: 'train_iv', key: 'train_iv', render: val => val?.toFixed(4) },
                                    { title: 'Valid IV', dataIndex: 'valid_iv', key: 'valid_iv', render: val => val?.toFixed(4) },
                                    { title: 'OOT IV', dataIndex: 'oot_iv', key: 'oot_iv', render: val => val?.toFixed(4) },
                                    { title: 'Train KS', dataIndex: 'train_ks', key: 'train_ks', render: val => val?.toFixed(4) },
                                    { title: 'Valid KS', dataIndex: 'valid_ks', key: 'valid_ks', render: val => val?.toFixed(4) },
                                    { title: 'OOT KS', dataIndex: 'oot_ks', key: 'oot_ks', render: val => val?.toFixed(4) },
                                ]}
                                pagination={{ pageSize: 12 }}
                                size="small"
                                scroll={{ x: 1000 }}
                            />
                        </Tabs.TabPane>
                        <Tabs.TabPane tab="分箱明细与 PSI" key="psi">
                            <Row gutter={24}>
                                <Col span={8}>
                                    <Card type="inner" title="稳定性指标 (PSI)" size="small">
                                        <Table
                                            dataSource={reportData.psi_monthly}
                                            columns={[
                                                { title: '月份', dataIndex: 'month', key: 'month' },
                                                { title: 'PSI', dataIndex: 'psi', key: 'psi', render: (val, record: any) => record.is_baseline ? <Tag>基准</Tag> : val.toFixed(4) }
                                            ]}
                                            pagination={false}
                                            size="small"
                                        />
                                        {reportData.psi_train_oot && reportData.psi_train_oot.length > 0 && (
                                            <div style={{ marginTop: 12, padding: 8, background: '#f5f5f5' }}>
                                                <strong>总体 PSI (Train vs OOT):</strong> {reportData.psi_train_oot[0].value.toFixed(4)}
                                            </div>
                                        )}
                                    </Card>
                                </Col>
                                <Col span={16}>
                                    <Card type="inner" title="分箱分布明细" size="small">
                                        <Table
                                            dataSource={reportData.bin_details}
                                            columns={[
                                                { title: '特征名', dataIndex: 'var_name', key: 'var_name', filters: Array.from(new Set(((reportData?.bin_details || []) as any[]).map((x: any) => x.var_name))).map(x => ({ text: x as string, value: x as string })), onFilter: (value, record: any) => record.var_name === value },
                                                { title: '分箱', dataIndex: 'bin', key: 'bin' },
                                                { title: '样本占比', dataIndex: 'total_pct', key: 'total_pct', render: val => (val * 100).toFixed(2) + '%' },
                                                { title: '坏账率', dataIndex: 'bad_rate', key: 'bad_rate', render: val => (val * 100).toFixed(2) + '%' },
                                                { title: 'WOE', dataIndex: 'woe', key: 'woe', render: val => val?.toFixed(3) },
                                                { title: 'IV', dataIndex: 'iv', key: 'iv', render: val => val?.toFixed(3) },
                                                { title: 'Lift', dataIndex: 'lift', key: 'lift', render: val => val?.toFixed(2) },
                                                { title: '累计 Lift', dataIndex: 'cum_lift', key: 'cum_lift', render: val => val?.toFixed(2) },
                                            ]}
                                            pagination={{ pageSize: 10 }}
                                            size="small"
                                        />
                                    </Card>
                                </Col>
                            </Row>
                        </Tabs.TabPane>
                    </Tabs>
                ) : (
                    <div style={{ padding: 50, textAlign: 'center', color: '#999' }}>
                        该模型版本尚未生成评估报告，请点击下方「生成报告」按钮。
                    </div>
                )}
            </Modal>
        </div>
    );
};

export default ResultPage;
