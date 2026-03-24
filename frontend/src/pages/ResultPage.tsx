import React, { useEffect, useState } from 'react';
import { Card, Table, Descriptions, Typography, Row, Col, message, Switch, Space, Tag } from 'antd';
import { CheckCircleOutlined } from '@ant-design/icons';
import api from '../services/api';
import { useAppStore } from '../stores';

const { Title } = Typography;

import ReactECharts from 'echarts-for-react';

const ResultPage: React.FC = () => {
    const currentProjectId = useAppStore((state) => state.currentProjectId);
    const [results, setResults] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);

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
        </div>
    );
};

export default ResultPage;
