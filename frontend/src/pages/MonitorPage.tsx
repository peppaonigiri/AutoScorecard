import React, { useEffect, useState } from 'react';
import { Card, Button, Row, Col, Statistic, Table, Empty, message, Tag, Space, Typography, Tabs } from 'antd';
const { Text } = Typography;
import { PlayCircleOutlined, DashboardOutlined, SafetyCertificateOutlined, LineChartOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import api from '../services/api';
import { useAppStore } from '../stores';


const MonitorPage: React.FC = () => {
    const currentProjectId = useAppStore((state) => state.currentProjectId);
    const [deployments, setDeployments] = useState<any[]>([]);
    const [logs, setLogs] = useState<any[]>([]);
    const [simulating, setSimulating] = useState(false);

    // 策略监控相关
    const [strategyLogs, setStrategyLogs] = useState<any[]>([]);

    useEffect(() => {
        if (currentProjectId) {
            fetchDeployments();
            fetchStrategyLogs();
        }
    }, [currentProjectId]);

    useEffect(() => {
        if (currentProjectId) {
            fetchLogs();
        }
    }, [currentProjectId]);

    const fetchDeployments = async () => {
        try {
            const res: any = await api.get(`/projects/${currentProjectId}/deployments`);
            setDeployments(res || []);
            if (res && res.length > 0) {
            }
        } catch (err) {
            message.error('获取部署列表失败');
        }
    };

    const fetchLogs = async () => {
        try {
            const res: any = await api.get(`/projects/${currentProjectId}/monitor/model_logs`);
            setLogs(res || []);
        } catch (err) {
            message.error('获取监控日志失败');
        }
    };

    const handleRunSimulation = async () => {
        setSimulating(true);
        try {
            await api.post(`/projects/${currentProjectId}/monitor/simulate_all`);
            message.success('模型与策略全量监控任务完成');
            fetchLogs();
            fetchStrategyLogs();
        } catch (err) {
            message.error('模拟任务执行失败');
        } finally {
            setSimulating(false);
        }
    };

    const fetchStrategyLogs = async () => {
        try {
            const res: any = await api.get(`/projects/${currentProjectId}/strategy_monitor/logs`);
            setStrategyLogs(res || []);
        } catch (err) {
            console.error('获取策略监控日志失败', err);
        }
    };


    // 准备 PSI 趋势图数据
    const getPsiChartOption = () => {
        const sortedLogs = [...logs].reverse();
        return {
            title: { text: '稳定性监控 (PSI Trend)', left: 'center' },
            tooltip: { trigger: 'axis' },
            xAxis: {
                type: 'category',
                data: sortedLogs.map(l => l.batch_name.split('_')[1] || l.batch_name),
                axisLabel: { rotate: 45 }
            },
            yAxis: { type: 'value', name: 'PSI' },
            series: [{
                data: sortedLogs.map(l => l.psi),
                type: 'line',
                smooth: true,
                areaStyle: {
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 0, y2: 1,
                        colorStops: [{ offset: 0, color: 'rgba(24, 144, 255, 0.3)' }, { offset: 1, color: 'rgba(24, 144, 255, 0)' }]
                    }
                },
                markLine: {
                    data: [{ yAxis: 0.1, name: '预警线', lineStyle: { color: '#faad14' } }, { yAxis: 0.25, name: '危险线', lineStyle: { color: '#f5222d' } }]
                }
            }]
        };
    };

    // 准备得分分布图数据
    const getDistChartOption = (log: any) => {
        if (!log || !log.score_dist) return {};
        return {
            title: { text: `得分分布 - ${log.batch_name}`, left: 'center' },
            tooltip: { trigger: 'axis' },
            xAxis: {
                type: 'category',
                data: (log.score_dist.bins || []).map((b: number) => b.toFixed(0)),
                name: 'Score'
            },
            yAxis: { type: 'value', name: 'Count' },
            series: [{
                data: log.score_dist.counts,
                type: 'bar',
                itemStyle: { color: '#1677ff' }
            }]
        };
    };

    // 策略命中率对比图 (柱状图)
    const getStrategyHitChartOption = (log: any) => {
        if (!log || !log.rule_stats) return {};
        return {
            title: { text: '各规则独立拦截率', left: 'center' },
            tooltip: { trigger: 'axis', formatter: '{b}: {c}%' },
            xAxis: {
                type: 'category',
                data: log.rule_stats.map((s: any) => s.name),
                axisLabel: { rotate: 20 }
            },
            yAxis: { type: 'value', name: '拦截率 %', max: 100 },
            series: [{
                data: log.rule_stats.map((s: any) => (s.node_intercept_rate * 100).toFixed(2)),
                type: 'bar',
                itemStyle: { color: '#ff4d4f' },
                label: { show: true, position: 'top', formatter: '{c}%' }
            }]
        };
    };

    // 通过率演变趋势 (折线图/面积图 - 展示漏斗效应)
    const getStrategyPassRateChartOption = (log: any) => {
        if (!log || !log.rule_stats) return {};
        return {
            title: { text: '全局通过率演变 (Survival Funnel)', left: 'center' },
            tooltip: { trigger: 'axis' },
            xAxis: {
                type: 'category',
                data: ['初始', ...log.rule_stats.map((s: any) => s.name)],
                axisLabel: { rotate: 20 }
            },
            yAxis: { type: 'value', name: '全局存活率 %', max: 100 },
            series: [{
                name: '存活比例',
                data: [100, ...log.rule_stats.map((s: any) => (s.global_pass_rate * 100).toFixed(2))],
                type: 'line',
                smooth: true,
                areaStyle: { opacity: 0.1 },
                label: { show: true, position: 'top', formatter: '{c}%' }
            }]
        };
    };

    if (!currentProjectId) return <div style={{ padding: 40 }}><Empty description="请先选择项目" /></div>;

    const latestLog = logs[0];

    return (
        <div style={{ paddingBottom: 40 }}>
            <Card style={{ marginBottom: 24, background: 'linear-gradient(135deg, #f0f5ff 0%, #ffffff 100%)' }} className="card-shadow">
                <Row gutter={24} align="middle">
                    <Col span={12}>
                        <div style={{ marginBottom: 8 }}><DashboardOutlined /> <b>联合监控调度中心</b></div>
                        <Text type="secondary">
                            同时针对[已上线模型]与[已激活策略流]进行未来进件数据模拟，同步评估稳定性指标与风险拦截效能。
                        </Text>
                    </Col>
                    <Col span={12} style={{ textAlign: 'right' }}>
                        <Space size="large">
                            {deployments.length > 0 && (() => {
                                const active = deployments.find(d => d.status === 'active') || deployments[0];
                                return (
                                    <div style={{ textAlign: 'left' }}>
                                        <div style={{ fontSize: '12px', color: '#8c8c8c' }}>当前监测模型：</div>
                                        <Tag color="blue" style={{ fontSize: '14px', padding: '4px 8px' }}>
                                            Deployment #{active.id} (Model #{active.model_result_id}) {active.status === 'active' ? '● Active' : ''}
                                        </Tag>
                                    </div>
                                );
                            })()}
                            <Button
                                type="primary"
                                size="large"
                                icon={<PlayCircleOutlined />}
                                onClick={handleRunSimulation}
                                loading={simulating}
                                style={{ height: 50, padding: '0 32px', borderRadius: 8, fontSize: 16, boxShadow: '0 4px 12px rgba(22, 119, 255, 0.3)' }}
                            >
                                运行联合模拟监控
                            </Button>
                        </Space>
                    </Col>
                </Row>
            </Card>

            <Tabs defaultActiveKey="model" className="custom-tabs">
                <Tabs.TabPane tab={<span><LineChartOutlined /> 模型稳定性监控</span>} key="model">

                    {latestLog ? (
                        <>
                            <Row gutter={24} style={{ marginBottom: 24 }}>
                                <Col span={6}>
                                    <Card variant="borderless" className="card-shadow">
                                        <Statistic
                                            title="当前 PSI (稳定性)"
                                            value={latestLog.psi}
                                            precision={4}
                                            valueStyle={{ color: latestLog.psi < 0.1 ? '#3f8600' : latestLog.psi < 0.25 ? '#cf1322' : '#f5222d' }}
                                        />
                                        <div style={{ marginTop: 8 }}>
                                            {latestLog.psi < 0.1 ? <Tag color="success">极度稳定</Tag> : <Tag color="error">稳定性存疑</Tag>}
                                        </div>
                                    </Card>
                                </Col>
                                <Col span={6}>
                                    <Card variant="borderless" className="card-shadow">
                                        <Statistic title="平均打分" value={latestLog.avg_score} precision={1} />
                                    </Card>
                                </Col>
                                <Col span={6}>
                                    <Card variant="borderless" className="card-shadow">
                                        <Statistic title="模拟样本量" value={latestLog.sample_size} />
                                    </Card>
                                </Col>
                                <Col span={6}>
                                    <Card variant="borderless" className="card-shadow">
                                        <Statistic title="最近更新" value={new Date(latestLog.created_at).toLocaleTimeString()} />
                                    </Card>
                                </Col>
                            </Row>

                            <Row gutter={24}>
                                <Col span={14}>
                                    <Card title="稳定性趋势" className="card-shadow">
                                        <ReactECharts option={getPsiChartOption()} style={{ height: 350 }} />
                                    </Card>
                                </Col>
                                <Col span={10}>
                                    <Card title="分数分布直方图" className="card-shadow">
                                        <ReactECharts option={getDistChartOption(latestLog)} style={{ height: 350 }} />
                                    </Card>
                                </Col>
                            </Row>
                        </>
                    ) : <Empty description="暂无模型监控数据" />}
                </Tabs.TabPane>

                <Tabs.TabPane tab={<span><SafetyCertificateOutlined /> 策略流运行监控</span>} key="strategy">

                    {strategyLogs.length > 0 ? (
                        <>
                            <Row gutter={24} style={{ marginBottom: 24 }}>
                                <Col span={8}>
                                    <Card variant="borderless" className="card-shadow">
                                        <Statistic
                                            title="整场通过率"
                                            value={strategyLogs[0].approval_rate * 100}
                                            precision={2}
                                            suffix="%"
                                            valueStyle={{ color: '#52c41a' }}
                                        />
                                    </Card>
                                </Col>
                                <Col span={8}>
                                    <Card variant="borderless" className="card-shadow">
                                        <Statistic title="累计拦截量" value={strategyLogs[0].hit_count} />
                                    </Card>
                                </Col>
                                <Col span={8}>
                                    <Card variant="borderless" className="card-shadow">
                                        <Statistic title="进件总量" value={strategyLogs[0].total_count} />
                                    </Card>
                                </Col>
                            </Row>

                            <Row gutter={24}>
                                <Col span={12}>
                                    <Card title="各策略规则拦截强度" className="card-shadow">
                                        <ReactECharts option={getStrategyHitChartOption(strategyLogs[0])} style={{ height: 400 }} />
                                    </Card>
                                </Col>
                                <Col span={12}>
                                    <Card title="通过率演变趋势" className="card-shadow">
                                        <ReactECharts option={getStrategyPassRateChartOption(strategyLogs[0])} style={{ height: 400 }} />
                                    </Card>
                                </Col>
                            </Row>

                            <Card title="各规则执行详情 (最近批次)" style={{ marginTop: 24 }} className="card-shadow">
                                <Table
                                    dataSource={strategyLogs[0].rule_stats}
                                    pagination={false}
                                    rowKey="id"
                                    columns={[
                                        { title: '策略名称', dataIndex: 'name', key: 'name' },
                                        {
                                            title: '模式',
                                            dataIndex: 'rule_type',
                                            key: 'type',
                                            render: (v) => v === 'approve' ? <Tag color="green">通过型</Tag> : <Tag color="volcano">拦截型</Tag>
                                        },
                                        {
                                            title: '独立拦截率',
                                            dataIndex: 'node_intercept_rate',
                                            key: 'nir',
                                            render: v => <span style={{ fontWeight: 'bold', color: v > 0.1 ? '#cf1322' : '#fa8c16' }}>{(v * 100).toFixed(2)}%</span>
                                        },
                                        {
                                            title: '全局拦截率',
                                            dataIndex: 'global_intercept_rate',
                                            key: 'gir',
                                            render: v => `${(v * 100).toFixed(2)}%`
                                        },
                                        {
                                            title: '独立通过率',
                                            dataIndex: 'node_pass_rate',
                                            key: 'npr',
                                            render: v => `${(v * 100).toFixed(2)}%`
                                        },
                                        {
                                            title: '全局通过率',
                                            dataIndex: 'global_pass_rate',
                                            key: 'gpr',
                                            render: v => <span style={{ color: '#52c41a' }}>{(v * 100).toFixed(2)}%</span>
                                        }
                                    ]}
                                />
                            </Card>
                        </>
                    ) : <Empty description="暂无策略监控数据" />}
                </Tabs.TabPane>
            </Tabs>
        </div>
    );
};

export default MonitorPage;
