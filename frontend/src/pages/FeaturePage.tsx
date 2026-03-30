import React, { useState } from 'react';
import { Card, Table, Form, InputNumber, Button, message, Tag, Row, Col, Typography, Input, Select, Divider, Alert, Space, Statistic } from 'antd';
import api from '../services/api';
import { useAppStore } from '../stores';

const { Title, Text } = Typography;
const { Option } = Select;

const FeaturePage: React.FC = () => {
    const {
        currentProjectId, currentDatasetId, datasetInfo,
        ivReport, setIvReport,
        filterResult, setFilterResult,
        setSelectedFeatures,
        excludeCols
    } = useAppStore();
    const [form] = Form.useForm();
    const [splitForm] = Form.useForm();

    const [loading, setLoading] = useState(false);
    const [filtering, setFiltering] = useState(false);

    // 页面加载或切换项目时，尝试拉取持久化的多集报告和筛选结果
    React.useEffect(() => {
        const fetchSavedResults = async () => {
            if (currentProjectId) {
                try {
                    const res: any = await api.get(`/projects/${currentProjectId}`);
                    if (res.iv_report && res.iv_report.length > 0) {
                        setIvReport(res.iv_report);
                    } else {
                        setIvReport([]); // 切换新项目时如果为空，也需要清空
                    }

                    if (res.filter_result && Object.keys(res.filter_result).length > 0) {
                        setFilterResult(res.filter_result);
                        if (res.filter_result.kept_features) {
                            setSelectedFeatures(res.filter_result.kept_features);
                        }
                    } else {
                        setFilterResult(null);
                    }
                } catch (e) {
                    console.error('拉取项目持久化特征分析结果失败:', e);
                }
            }
        };
        fetchSavedResults();
    }, [currentProjectId]);

    const fetchIvReport = async () => {
        if (!currentProjectId || !currentDatasetId) {
            message.warning('请先上传或选择数据集');
            return;
        }
        setLoading(true);
        try {
            const splitValues = splitForm.getFieldsValue();
            const payload: any = {
                dataset_id: currentDatasetId,
                dep: splitValues.dep || 'label',
                exclude_cols: Array.from(new Set(['id', 'uuid', 'user_id', 'date', ...excludeCols])),
                split_ratios: [splitValues.train_ratio, splitValues.valid_ratio],
            };
            // OOT 切分方式
            if (splitValues.oot_col) {
                payload.oot_col = splitValues.oot_col;
            }
            if (splitValues.oot_mode === 'pct' && splitValues.oot_pct) {
                payload.oot_pct = splitValues.oot_pct;
            } else if (splitValues.oot_mode === 'time' && splitValues.oot_start_time) {
                payload.oot_start_time = splitValues.oot_start_time;
            }

            const res: any = await api.post(`/projects/${currentProjectId}/feature/iv-report`, payload);
            setIvReport(res.features || []);
            message.success(`多集分析报告已生成 (${res.total_features} 个特征)`);
        } catch (err: any) {
            message.error('生成分析报告失败: ' + (err?.response?.data?.detail || err?.message || ''));
        } finally {
            setLoading(false);
        }
    };

    const onFilterSubmit = async (values: any) => {
        if (!currentProjectId || !currentDatasetId) return;
        setFiltering(true);
        try {
            const splitValues = splitForm.getFieldsValue();
            const payload: any = {
                dataset_id: currentDatasetId,
                dep: splitValues.dep || 'label',
                exclude_cols: Array.from(new Set(['id', 'uuid', 'user_id', 'date', ...excludeCols])),
                thresholds: values,
                split_ratios: [splitValues.train_ratio, splitValues.valid_ratio],
            };
            if (splitValues.oot_col) payload.oot_col = splitValues.oot_col;
            if (splitValues.oot_mode === 'pct' && splitValues.oot_pct) {
                payload.oot_pct = splitValues.oot_pct;
            } else if (splitValues.oot_mode === 'time' && splitValues.oot_start_time) {
                payload.oot_start_time = splitValues.oot_start_time;
            }

            const res: any = await api.post(`/projects/${currentProjectId}/feature/filter`, payload);
            setFilterResult(res);
            setSelectedFeatures(res.kept_features || []);
            message.success('变量筛选完成');
        } catch (err: any) {
            message.error('筛选失败: ' + (err?.response?.data?.detail || ''));
        } finally {
            setFiltering(false);
        }
    };

    const ivColumns = [
        { title: '特征名', dataIndex: 'var_names', key: 'var_names', fixed: 'left' as const, width: 180 },
        {
            title: 'Train IV',
            dataIndex: 'iv_train',
            key: 'iv_train',
            sorter: (a: any, b: any) => (a.iv_train || 0) - (b.iv_train || 0),
            render: (val: number) => {
                if (val == null) return '-';
                const color = val > 0.1 ? '#52c41a' : val > 0.02 ? '#1677ff' : '#f5222d';
                return <Text strong style={{ color }}>{val.toFixed(4)}</Text>;
            }
        },
        { title: 'Valid IV', dataIndex: 'iv_valid', key: 'iv_valid', render: (val: number) => val?.toFixed(4) || '-' },
        { title: 'OOT IV', dataIndex: 'iv_oot', key: 'iv_oot', render: (val: number) => val?.toFixed(4) || '-' },
        {
            title: 'PSI (T↔V)',
            dataIndex: 'psi_tv',
            key: 'psi_tv',
            render: (val: number) => <Tag color={(val || 0) > 0.1 ? 'red' : 'green'}>{val?.toFixed(4) || '0.0000'}</Tag>
        },
        {
            title: 'PSI (T↔O)',
            dataIndex: 'psi_to',
            key: 'psi_to',
            render: (val: number) => <Tag color={(val || 0) > 0.15 ? 'red' : 'green'}>{val?.toFixed(4) || '0.0000'}</Tag>
        },
    ];

    return (
        <div style={{ paddingBottom: 40 }}>
            <Title level={4}>变量分析与筛选</Title>

            <Card style={{ marginBottom: 24 }} title="第一步：样本划分与多集稳定性分析">
                <Form
                    form={splitForm}
                    layout="vertical"
                    initialValues={{
                        dep: 'label',
                        train_ratio: 0.8,
                        valid_ratio: 0.2,
                        oot_col: '',
                        oot_mode: 'pct',
                        oot_pct: 0.2,
                        oot_start_time: ''
                    }}
                >
                    <Row gutter={16}>
                        <Col span={4}>
                            <Form.Item name="dep" label="目标变量">
                                <Input placeholder="默认 label" />
                            </Form.Item>
                        </Col>
                        <Col span={8}>
                            <Form.Item label="划分比例 (Train / Valid)">
                                <Space>
                                    <Form.Item name="train_ratio" noStyle><InputNumber min={0} max={1} step={0.1} style={{ width: 65 }} /></Form.Item>
                                    <Form.Item name="valid_ratio" noStyle><InputNumber min={0} max={1} step={0.1} style={{ width: 65 }} /></Form.Item>
                                </Space>
                            </Form.Item>
                        </Col>
                        <Col span={5}>
                            <Form.Item name="oot_col" label="OOT 时间列 (可选)">
                                <Input placeholder="如 backPointTime" />
                            </Form.Item>
                        </Col>
                        <Col span={3}>
                            <Form.Item name="oot_mode" label="OOT 切分方式">
                                <Select>
                                    <Option value="pct">按百分比</Option>
                                    <Option value="time">按时间点</Option>
                                </Select>
                            </Form.Item>
                        </Col>
                        <Col span={4}>
                            <Form.Item noStyle shouldUpdate={(prev, cur) => prev.oot_mode !== cur.oot_mode}>
                                {() => {
                                    const mode = splitForm.getFieldValue('oot_mode');
                                    if (mode === 'pct') {
                                        return (
                                            <Form.Item name="oot_pct" label="OOT 占比">
                                                <Select>
                                                    <Option value={0.1}>最后 10%</Option>
                                                    <Option value={0.15}>最后 15%</Option>
                                                    <Option value={0.2}>最后 20%</Option>
                                                    <Option value={0.25}>最后 25%</Option>
                                                    <Option value={0.3}>最后 30%</Option>
                                                </Select>
                                            </Form.Item>
                                        );
                                    }
                                    return (
                                        <Form.Item name="oot_start_time" label="OOT 起始时间">
                                            <Input placeholder="如 2024-01-01" />
                                        </Form.Item>
                                    );
                                }}
                            </Form.Item>
                        </Col>
                    </Row>
                    <Button type="primary" onClick={fetchIvReport} loading={loading} size="large">
                        生成多集 IV & PSI 报告
                    </Button>
                </Form>

                {ivReport.length > 0 && (
                    <Table
                        dataSource={ivReport}
                        columns={ivColumns}
                        rowKey="var_names"
                        style={{ marginTop: 24 }}
                        scroll={{ x: 900 }}
                        pagination={{ pageSize: 15 }}
                    />
                )}
            </Card>

            <Card title="第二步：执行多变量自动筛选">
                <Form
                    form={form}
                    layout="inline"
                    onFinish={onFilterSubmit}
                    initialValues={{
                        missing: 0.8,
                        freq: 0.95,
                        iv: 0.02,
                        corr: 0.9,
                        psi: 0.1
                    }}
                >
                    <Form.Item name="iv" label="最小 Train IV 阈值">
                        <InputNumber min={0} max={1} step={0.01} />
                    </Form.Item>
                    <Form.Item name="psi" label="最大 PSI 阈值">
                        <InputNumber min={0} max={1} step={0.01} />
                    </Form.Item>
                    <Form.Item name="corr" label="最大相关性">
                        <InputNumber min={0} max={1} step={0.05} />
                    </Form.Item>
                    <Form.Item>
                        <Button type="primary" htmlType="submit" loading={filtering} size="large">
                            执行自动筛选
                        </Button>
                    </Form.Item>
                </Form>

                {filterResult && (
                    <div style={{ marginTop: 24 }}>
                        <Alert
                            message="变量筛选摘要"
                            description={
                                <Row gutter={16}>
                                    <Col span={5}><Statistic title="总输入" value={filterResult.summary.total_input} /></Col>
                                    <Col span={5}><Statistic title="最终保留" value={filterResult.summary.total_kept} valueStyle={{ color: '#3f8600' }} /></Col>
                                    <Col span={5}><Statistic title="IV 剔除" value={filterResult.summary.drop_by_iv} valueStyle={{ color: '#cf1322' }} /></Col>
                                    <Col span={5}><Statistic title="PSI 剔除" value={filterResult.summary.drop_by_psi} valueStyle={{ color: '#cf1322' }} /></Col>
                                    <Col span={4}><Statistic title="相关性剔除" value={filterResult.summary.drop_by_corr} valueStyle={{ color: '#cf1322' }} /></Col>
                                </Row>
                            }
                            type="info"
                            showIcon
                            style={{ marginBottom: 24 }}
                        />

                        <Divider orientation={"left" as any}>剔除详情清单 (全量)</Divider>
                        <Row gutter={[16, 16]}>
                            {datasetInfo?.l1_results && (
                                <Col span={24}>
                                    <Card size="small" title="基础质量剔除 (Data Prep 阶段 - 已由系统默认执行)" extra={<Tag color="orange">预筛选 L1</Tag>}>
                                        <div style={{ maxHeight: 120, overflowY: 'auto' }}>
                                            <Space wrap>
                                                {(datasetInfo.l1_results.removed_info?.single_value || datasetInfo.l1_results.removed_info?.freq || []).length > 0 &&
                                                    <Tag color="volcano">单一值比例高: {(datasetInfo.l1_results.removed_info.single_value || datasetInfo.l1_results.removed_info.freq).join(', ')}</Tag>}
                                                {(datasetInfo.l1_results.removed_info?.null_rate || datasetInfo.l1_results.removed_info?.missing || []).length > 0 &&
                                                    <Tag color="magenta">缺失率过高: {(datasetInfo.l1_results.removed_info.null_rate || datasetInfo.l1_results.removed_info.missing).join(', ')}</Tag>}
                                                {(datasetInfo.l1_results.removed_info?.zero_std || []).length > 0 &&
                                                    <Tag color="red">零标准差: {datasetInfo.l1_results.removed_info.zero_std.join(', ')}</Tag>}
                                                {!(datasetInfo.l1_results.removed_info?.single_value?.length || datasetInfo.l1_results.removed_info?.freq?.length || datasetInfo.l1_results.removed_info?.null_rate?.length || datasetInfo.l1_results.removed_info?.missing?.length || datasetInfo.l1_results.removed_info?.zero_std?.length) && '无统计剔除指标'}
                                            </Space>
                                        </div>
                                    </Card>
                                </Col>
                            )}
                            <Col span={24}>
                                <Card size="small" title="业务效果剔除 (IV / PSI / 相关性)" extra={<Tag color="blue">模型筛选 L2</Tag>}>
                                    <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                                        <Text type="secondary">IV 贡献低 (<Text type="danger">{filterResult.summary.drop_by_iv}</Text>): </Text> {filterResult.dropped_features.iv?.join(', ') || '无'}<br />
                                        <Text type="secondary">稳定性差 PSI (<Text type="danger">{filterResult.summary.drop_by_psi}</Text>): </Text> {filterResult.dropped_features.psi?.join(', ') || '无'}<br />
                                        <Text type="secondary">共线性排除 (<Text type="danger">{filterResult.summary.drop_by_corr}</Text>): </Text> {filterResult.dropped_features.corr?.join(', ') || '无'}
                                    </div>
                                </Card>
                            </Col>
                        </Row>
                    </div>
                )}
            </Card>
        </div>
    );
};

export default FeaturePage;
