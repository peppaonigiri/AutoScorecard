import React, { useState, useEffect } from 'react';
import { Card, Form, Input, Select, Button, Space, Table, Typography, Statistic, Row, Col, Divider, message, Popconfirm, Modal, Descriptions, Switch, Badge, Alert, InputNumber, Progress } from 'antd';
import { PlusOutlined, DeleteOutlined, AreaChartOutlined, SaveOutlined, EyeOutlined, HolderOutlined } from '@ant-design/icons';
import { DndContext, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { restrictToVerticalAxis } from '@dnd-kit/modifiers';
import { arrayMove, SortableContext, useSortable, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { useAppStore } from '../stores';
import api from '../services/api';

const { Text } = Typography;
const { Option } = Select;

interface RowProps extends React.HTMLAttributes<HTMLTableRowElement> {
    'data-row-key': string;
}

const SortableRow = ({ children, ...props }: RowProps) => {
    const {
        attributes,
        listeners,
        setNodeRef,
        setActivatorNodeRef,
        transform,
        transition,
        isDragging,
    } = useSortable({
        id: props['data-row-key'],
    });

    const style: React.CSSProperties = {
        ...props.style,
        transform: CSS.Translate.toString(transform),
        transition,
        ...(isDragging ? { position: 'relative', zIndex: 9999 } : {}),
    };

    return (
        <tr {...props} ref={setNodeRef} style={style} {...attributes}>
            {React.Children.map(children, (child) => {
                if ((child as React.ReactElement).key === 'sort') {
                    return React.cloneElement(child as any, {
                        children: (
                            <HolderOutlined
                                ref={setActivatorNodeRef}
                                style={{ cursor: 'grab', touchAction: 'none' }}
                                {...listeners}
                            />
                        ),
                    });
                }
                return child;
            })}
        </tr>
    );
};

const StrategyPage: React.FC = () => {
    const currentProjectId = useAppStore((state: any) => state.currentProjectId);
    const [form] = Form.useForm();
    const [datasets, setDatasets] = useState<any[]>([]);
    const [columns, setColumns] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);
    const [analysisResults, setAnalysisResults] = useState<any>(null);
    const [savedStrategies, setSavedStrategies] = useState<any[]>([]);
    const [deployments, setDeployments] = useState<any[]>([]);
    const [isSaveModalVisible, setIsSaveModalVisible] = useState(false);
    const [viewModalVisible, setViewModalVisible] = useState(false);
    const [viewItem, setViewItem] = useState<any>(null);
    const [explorerVisible, setExplorerVisible] = useState(false);
    const [explorerLoading, setExplorerLoading] = useState(false);
    const [binningData, setBinningData] = useState<any[]>([]);
    const [explorerForm] = Form.useForm();
    const [miningVisible, setMiningVisible] = useState(false);
    const [miningLoading, setMiningLoading] = useState(false);
    const [miningProgress, setMiningProgress] = useState(0);
    const [recommendations, setRecommendations] = useState<any[]>([]);
    const [miningForm] = Form.useForm();

    useEffect(() => {
        if (currentProjectId) {
            fetchDatasets();
            fetchSavedStrategies();
            fetchDeployments();
        }
    }, [currentProjectId]);

    const fetchDeployments = async () => {
        try {
            const res: any = await api.get(`/projects/${currentProjectId}/deployments`);
            setDeployments(res || []);
        } catch (err) {
            console.error('获取部署模型失败', err);
        }
    };

    const fetchDatasets = async () => {
        try {
            const res: any = await api.get(`/projects/${currentProjectId}/datasets`);
            setDatasets(res || []);
        } catch (err) {
            message.error('获取数据集失败');
        }
    };

    const fetchSavedStrategies = async () => {
        try {
            const res: any = await api.get(`/strategies/projects/${currentProjectId}`);
            setSavedStrategies(res || []);
        } catch (err) {
            message.error('获取策略列表失败');
        }
    };

    const handleDatasetChange = (datasetId: number) => {
        const ds = datasets.find(d => d.id === datasetId);
        if (ds && ds.columns_info) {
            setColumns(Object.keys(ds.columns_info));
        }
    };

    const onAnalyze = async (values: any) => {
        setLoading(true);
        try {
            const res = await api.post('/strategies/analyze', {
                project_id: currentProjectId,
                dataset_id: values.dataset_id,
                rules: values.rules || [],
                combine_logic: values.combine_logic || 'and',
                rule_type: values.rule_type || 'reject'
            });
            setAnalysisResults(res);
            message.success('策略分析完成');
        } catch (err: any) {
            message.error(err.response?.data?.detail || '分析失败');
        } finally {
            setLoading(false);
        }
    };

    const handleSaveStrategy = async (nameValues: any) => {
        const values = form.getFieldsValue();
        try {
            await api.post('/strategies', {
                project_id: currentProjectId,
                name: nameValues.name,
                description: nameValues.description,
                rules: values.rules,
                combine_logic: values.combine_logic || 'and',
                rule_type: values.rule_type || 'reject',
                metrics: analysisResults || {}
            });
            message.success('策略保存成功');
            setIsSaveModalVisible(false);
            fetchSavedStrategies();
        } catch (err) {
            message.error('保存失败');
        }
    };

    const onDragEnd = async ({ active, over }: any) => {
        if (active.id !== over?.id) {
            const activeIndex = savedStrategies.findIndex((i: any) => i.id === active.id);
            const overIndex = savedStrategies.findIndex((i: any) => i.id === over?.id);
            const newArr = arrayMove(savedStrategies, activeIndex, overIndex);
            setSavedStrategies(newArr);

            // 同步到后端
            try {
                const items = newArr.map((s: any, idx: number) => ({ id: s.id, priority: idx }));
                await api.post('/strategies/reorder', { items });
                message.success('排序已更新');
            } catch (err) {
                message.error('保存排序失败');
                fetchSavedStrategies();
            }
        }
    };

    const toggleStatus = async (id: number, currentStatus: string) => {
        const nextStatus = currentStatus === 'active' ? 'draft' : 'active';
        try {
            await api.patch(`/strategies/${id}/status`, { status: nextStatus });
            message.success(`策略已${nextStatus === 'active' ? '上线' : '下线'}`);
            fetchSavedStrategies();
        } catch (err) {
            message.error('操作失败');
        }
    };

    const handleExploreBinning = async (values: any) => {
        setExplorerLoading(true);
        try {
            const res: any = await api.post(`/datasets/${values.dataset_id}/binning-explorer`, values);
            setBinningData(res || []);
            message.success('分箱计算完成');
        } catch (err: any) {
            message.error(err.response?.data?.detail || '计算失败');
        } finally {
            setExplorerLoading(false);
        }
    };

    const binningColumns = [
        { title: '区间', dataIndex: 'bin', key: 'bin' },
        { title: '样本数', dataIndex: 'total', key: 'total' },
        { title: '样本占比', dataIndex: 'total_pct', key: 'total_pct', render: (v: number) => (v * 100).toFixed(2) + '%' },
        { title: '坏人占比', dataIndex: 'bad_pct', key: 'bad_pct', render: (v: number) => (v * 100).toFixed(2) + '%' },
        { title: '坏账率', dataIndex: 'bad_rate', key: 'bad_rate', render: (v: number) => (v * 100).toFixed(2) + '%' },
        { title: 'WOE', dataIndex: 'woe', key: 'woe', render: (v: number) => v?.toFixed(4) },
        { title: 'IV', dataIndex: 'iv', key: 'iv', render: (v: number) => v?.toFixed(4) },
        { title: 'Lift', dataIndex: 'lift', key: 'lift', render: (v: number) => <Text strong type={v > 1 ? 'danger' : 'success'}>{v?.toFixed(2)}</Text> },
        { title: '累计 Lift', dataIndex: 'cum_lift', key: 'cum_lift', render: (v: number) => v?.toFixed(2) },
    ];

    const strategyColumns = [
        {
            key: 'sort',
            width: 40,
        },
        { title: '策略名称', dataIndex: 'name', key: 'name' },
        {
            title: '状态',
            dataIndex: 'status',
            key: 'status',
            render: (status: string, record: any) => (
                <Space>
                    <Badge status={status === 'active' ? 'processing' : 'default'} text={status === 'active' ? '已上线' : '草稿'} />
                    <Switch
                        size="small"
                        checked={status === 'active'}
                        onChange={() => toggleStatus(record.id, status)}
                    />
                </Space>
            )
        },
        {
            title: '核心数据',
            key: 'metrics',
            render: (_: any, record: any) => (
                <span>通过率: {(record.metrics?.approval_rate * 100).toFixed(2)}% | Lift: {record.metrics?.lift}</span>
            )
        },
        { title: '规则数', key: 'rules', render: (_: any, record: any) => record.rules?.length || 0 },
        { title: '创建时间', dataIndex: 'created_at', key: 'created_at', render: (t: string) => t.substring(0, 16) },
        {
            title: '操作',
            key: 'action',
            render: (_: any, record: any) => (
                <Space>
                    <Button
                        type="link"
                        icon={<EyeOutlined />}
                        onClick={() => {
                            setViewItem(record);
                            setViewModalVisible(true);
                        }}
                    />
                    <Popconfirm title="确定删除?" onConfirm={async () => {
                        await api.delete(`/strategies/${record.id}`);
                        fetchSavedStrategies();
                    }}>
                        <Button type="link" danger icon={<DeleteOutlined />} />
                    </Popconfirm>
                </Space>
            ),
        },
    ];

    const applyAutoRule = (ruleStr: string) => {
        // 解析简单的 "var > 10" 或 "var <= 5" 格式
        // 复杂规则 "a > 1 and b <= 2" 目前暂不兼容表单格式
        const rules = form.getFieldValue('rules') || [];
        const parts = ruleStr.split(' and ');
        const newRules = [...rules];

        parts.forEach(p => {
            const match = p.match(/(.+?)\s*([><=!]+)\s*(.+)/);
            if (match) {
                newRules.push({
                    field: match[1].trim(),
                    op: match[2].trim(),
                    val: match[3].trim()
                });
            }
        });

        form.setFieldsValue({ rules: newRules.filter(r => r.field) });
        message.success('已应用规则到列表');
    };

    const handleStartMining = async (values: any) => {
        setMiningLoading(true);
        setMiningProgress(0);
        setRecommendations([]);
        try {
            const res: any = await api.post('/strategies/auto-mining', {
                ...values,
                dataset_id: values.dataset_id
            });
            pollMiningTask(res.task_id);
        } catch (err: any) {
            message.error('启动挖掘任务失败');
            setMiningLoading(false);
        }
    };

    const pollMiningTask = async (taskId: number) => {
        const timer = setInterval(async () => {
            try {
                const res: any = await api.get(`/tasks/${taskId}`);
                setMiningProgress(res.progress);

                // 发送心跳信号
                try {
                    await api.post(`/tasks/${taskId}/heartbeat`);
                } catch (hErr) {
                    console.log('Heartbeat failed', hErr);
                }

                if (res.status === 'completed') {
                    clearInterval(timer);
                    setRecommendations(res.result || []);
                    setMiningLoading(false);
                } else if (res.status === 'failed') {
                    clearInterval(timer);
                    message.error('挖掘任务失败: ' + res.error_msg);
                    setMiningLoading(false);
                }
            } catch (err) {
                clearInterval(timer);
                setMiningLoading(false);
            }
        }, 2000);
    };

    const miningColumns = [
        { title: '推荐规则', dataIndex: 'rule', key: 'rule', width: 300, render: (t: string) => <Text code>{t}</Text> },
        { title: '坏账率(Train)', dataIndex: 'train_bad_rate', key: 'train_bad_rate', render: (v: number) => (v * 100).toFixed(2) + '%' },
        { title: 'Lift(Train)', dataIndex: 'train_lift', key: 'train_lift', render: (v: number) => <Text strong type="danger">{v.toFixed(2)}</Text> },
        { title: '命中率', dataIndex: 'train_hit_rate', key: 'train_hit_rate', render: (v: number) => (v * 100).toFixed(2) + '%' },
        { title: 'PSI', dataIndex: 'psi', key: 'psi', render: (v: number) => v.toFixed(4) },
        {
            title: '操作',
            key: 'action',
            render: (_: any, record: any) => (
                <Button size="small" type="primary" ghost onClick={() => applyAutoRule(record.rule)}>应用</Button>
            )
        }
    ];

    const sensors = useSensors(
        useSensor(PointerSensor, {
            activationConstraint: {
                distance: 1,
            },
        }),
    );

    return (
        <div style={{ padding: '24px' }}>
            <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                <Button icon={<AreaChartOutlined />} onClick={() => setExplorerVisible(true)}>
                    特征分箱探索
                </Button>
                <Button type="primary" icon={<AreaChartOutlined />} onClick={() => setMiningVisible(true)}>
                    自动化策略挖掘 (Auto Mining)
                </Button>
            </div>
            <Row gutter={24}>
                <Col span={14}>
                    <Card
                        title={<span><PlusOutlined /> 制定策略规则</span>}
                        className="card-shadow"
                        bordered={false}
                    >
                        <Form
                            form={form}
                            layout="vertical"
                            onFinish={onAnalyze}
                            initialValues={{ combine_logic: 'and', rules: [{}], rule_type: 'reject' }}
                        >
                            <Row gutter={16}>
                                <Col span={8}>
                                    <Form.Item name="dataset_id" label="基于数据集" rules={[{ required: true }]}>
                                        <Select placeholder="选择数据集回测分析" onChange={handleDatasetChange}>
                                            {datasets.map(ds => <Option key={ds.id} value={ds.id}>{ds.name}</Option>)}
                                        </Select>
                                    </Form.Item>
                                </Col>
                                <Col span={8}>
                                    <Form.Item name="rule_type" label="规则模式" rules={[{ required: true }]}>
                                        <Select onChange={() => { /* 仅用于强制表单重绘以更新描述 */ }}>
                                            <Option value="reject">拦截模式 (命中即拒绝)</Option>
                                            <Option value="approve">通过模式 (命中才通过)</Option>
                                        </Select>
                                    </Form.Item>
                                </Col>
                                <Col span={8}>
                                    <Form.Item
                                        noStyle
                                        shouldUpdate={(prev, curr) => prev.rule_type !== curr.rule_type}
                                    >
                                        {({ getFieldValue }) => {
                                            const type = getFieldValue('rule_type') || 'reject';
                                            return (
                                                <Form.Item name="combine_logic" label="组合逻辑" rules={[{ required: true }]}>
                                                    <Select>
                                                        <Option value="and">AND (满足全部规则{type === 'reject' ? '才拦截' : '才通过'})</Option>
                                                        <Option value="or">OR (满足任一规则{type === 'reject' ? '即拦截' : '即通过'})</Option>
                                                    </Select>
                                                </Form.Item>
                                            );
                                        }}
                                    </Form.Item>
                                </Col>
                            </Row>

                            <Divider>自定义规则项</Divider>

                            <Form.List name="rules">
                                {(fields, { add, remove }) => (
                                    <>
                                        {fields.map(({ key, name, ...restField }) => (
                                            <Space key={key} style={{ display: 'flex', marginBottom: 8 }} align="baseline">
                                                <Form.Item {...restField} name={[name, 'field']} rules={[{ required: true, message: '选择变量' }]}>
                                                    <Select placeholder="变量名" style={{ width: 220 }}>
                                                        <Select.OptGroup label="数据特征 (Dataset Features)">
                                                            {columns.map(c => <Option key={c} value={c}>{c}</Option>)}
                                                        </Select.OptGroup>
                                                        {deployments.length > 0 && (
                                                            <Select.OptGroup label="模型分数 (Model Scores)">
                                                                {deployments.map(d => (
                                                                    <Option
                                                                        key={`model_${d.model_result_id}`}
                                                                        value={`_model_result_${d.model_result_id}`}
                                                                    >
                                                                        [模型] {d.model_result?.model_type || 'Model'} #{d.model_result_id}
                                                                    </Option>
                                                                ))}
                                                            </Select.OptGroup>
                                                        )}
                                                    </Select>
                                                </Form.Item>
                                                <Form.Item {...restField} name={[name, 'op']} rules={[{ required: true, message: '操作符' }]}>
                                                    <Select placeholder="操作符" style={{ width: 100 }}>
                                                        <Option value=">">&gt;</Option>
                                                        <Option value="<">&lt;</Option>
                                                        <Option value=">=">&gt;=</Option>
                                                        <Option value="<=">&lt;=</Option>
                                                        <Option value="==">==</Option>
                                                        <Option value="!=">!=</Option>
                                                    </Select>
                                                </Form.Item>
                                                <Form.Item {...restField} name={[name, 'val']} rules={[{ required: true, message: '阈值' }]}>
                                                    <Input placeholder="输入拦截阈值" style={{ width: 150 }} />
                                                </Form.Item>
                                                <Button type="link" danger onClick={() => remove(name)} icon={<DeleteOutlined />} />
                                            </Space>
                                        ))}
                                        <Form.Item
                                            noStyle
                                            shouldUpdate={(prev, curr) => prev.rule_type !== curr.rule_type}
                                        >
                                            {({ getFieldValue }) => (
                                                <Form.Item>
                                                    <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                                                        添加{getFieldValue('rule_type') === 'reject' ? '拦截' : '通过'}子规则
                                                    </Button>
                                                </Form.Item>
                                            )}
                                        </Form.Item>
                                    </>
                                )}
                            </Form.List>

                            <Form.Item>
                                <Space>
                                    <Button type="primary" htmlType="submit" loading={loading} icon={<AreaChartOutlined />}>
                                        开始回测分析
                                    </Button>
                                    <Button
                                        disabled={!analysisResults}
                                        onClick={() => setIsSaveModalVisible(true)}
                                        icon={<SaveOutlined />}
                                    >
                                        保存该方案
                                    </Button>
                                </Space>
                            </Form.Item>
                        </Form>
                    </Card>
                </Col>

                <Col span={10}>
                    <Card title="策略回测指标" bordered={false} className="card-shadow" loading={loading}>
                        {analysisResults ? (
                            <div>
                                <Row gutter={16}>
                                    <Col span={12}>
                                        <Statistic title="通过率" value={(analysisResults.approval_rate * 100)} precision={2} suffix="%" valueStyle={{ color: '#3f8600' }} />
                                    </Col>
                                    <Col span={12}>
                                        <Statistic title="坏人捕获率" value={(analysisResults.bad_capture_rate * 100)} precision={2} suffix="%" />
                                    </Col>
                                </Row>
                                <Divider />
                                <Row gutter={16}>
                                    <Col span={12}>
                                        <Statistic title="Lift (提升度)" value={analysisResults.lift} precision={2} valueStyle={{ color: (analysisResults.lift > 1 ? '#cf1322' : '#3f8600') }} />
                                    </Col>
                                    <Col span={12}>
                                        <Statistic title="拦截坏件数" value={analysisResults.hit_bad} />
                                    </Col>
                                </Row>
                                <div style={{ marginTop: '24px', padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
                                    <Form.Item noStyle shouldUpdate>
                                        {({ getFieldValue }) => {
                                            const type = getFieldValue('rule_type') || 'reject';
                                            return (
                                                <Text type="secondary">
                                                    本策略共分析了 {analysisResults.total_count} 条样本。
                                                    {type === 'reject' ?
                                                        `共拦截了 ${analysisResults.hit_count} 位用户。拦截用户中的坏账浓度是总体的 ${analysisResults.lift} 倍。` :
                                                        `共通过了 ${analysisResults.pass_count} 位用户。通过件的坏账率（${(analysisResults.pass_bad_rate * 100).toFixed(2)}%）远低于总体。`
                                                    }
                                                </Text>
                                            );
                                        }}
                                    </Form.Item>
                                </div>
                            </div>
                        ) : (
                            <div style={{ padding: '40px', textAlign: 'center' }}>
                                <AreaChartOutlined style={{ fontSize: '48px', color: '#ccc' }} />
                                <p style={{ marginTop: '16px', color: '#999' }}>请在左侧配置规则并开始分析</p>
                            </div>
                        )}
                    </Card>

                    <Card title="历史保存方案" variant="borderless" style={{ marginTop: '24px' }} className="card-shadow">
                        <DndContext sensors={sensors} modifiers={[restrictToVerticalAxis]} onDragEnd={onDragEnd}>
                            <SortableContext
                                items={savedStrategies.map((i) => i.id)}
                                strategy={verticalListSortingStrategy}
                            >
                                <Table
                                    components={{
                                        body: {
                                            row: SortableRow,
                                        },
                                    }}
                                    dataSource={savedStrategies}
                                    columns={strategyColumns}
                                    rowKey="id"
                                    pagination={{ pageSize: 10 }}
                                    size="small"
                                />
                            </SortableContext>
                        </DndContext>
                    </Card>
                </Col>
            </Row>

            <Modal
                title="保存策略方案"
                open={isSaveModalVisible}
                onCancel={() => setIsSaveModalVisible(false)}
                footer={null}
            >
                <Form layout="vertical" onFinish={handleSaveStrategy}>
                    <Form.Item name="name" label="方案名称" rules={[{ required: true }]}>
                        <Input placeholder="例如: 高龄低分拦截策略" />
                    </Form.Item>
                    <Form.Item name="description" label="方案描述">
                        <Input.TextArea rows={3} placeholder="备注该策略的业务背景..." />
                    </Form.Item>
                    <Form.Item>
                        <Button type="primary" htmlType="submit" block>确定保存</Button>
                    </Form.Item>
                </Form>
            </Modal>

            <Modal
                title="策略方案详情"
                open={viewModalVisible}
                onCancel={() => setViewModalVisible(false)}
                footer={null}
                width={700}
            >
                {viewItem && (
                    <div>
                        <Descriptions title="基本信息" bordered column={2} size="small">
                            <Descriptions.Item label="方案名称">{viewItem.name}</Descriptions.Item>
                            <Descriptions.Item label="创建时间">{new Date(viewItem.created_at).toLocaleString()}</Descriptions.Item>
                            <Descriptions.Item label="方案描述" span={2}>{viewItem.description || '无描述'}</Descriptions.Item>
                        </Descriptions>

                        <Divider orientation={'left' as any}>回测核心性能 (Save Context)</Divider>
                        <Row gutter={16}>
                            <Col span={8}>
                                <Statistic title="通过率" value={viewItem.metrics?.approval_rate * 100} precision={2} suffix="%" />
                            </Col>
                            <Col span={8}>
                                <Statistic title="坏人捕获率" value={viewItem.metrics?.bad_capture_rate * 100} precision={2} suffix="%" />
                            </Col>
                            <Col span={8}>
                                <Statistic title="提升度 (Lift)" value={viewItem.metrics?.lift} precision={2} />
                            </Col>
                        </Row>

                        <Divider orientation={'left' as any}>拦截规则列表</Divider>
                        <Table
                            dataSource={viewItem.rules}
                            rowKey={(_, i) => `rule_${i}`}
                            pagination={false}
                            size="small"
                            columns={[
                                {
                                    title: '变量字段',
                                    dataIndex: 'field',
                                    key: 'field',
                                    render: (f: string) => {
                                        if (f.startsWith('_model_result_')) {
                                            const mid = f.replace('_model_result_', '');
                                            const dep = deployments.find(d => String(d.model_result_id) === String(mid));
                                            const modelName = dep ? `${dep.model_result?.model_type || 'Model'}` : '';
                                            return <Text strong type="danger">{`[模型分数] ${modelName} #${mid}`}</Text>;
                                        }
                                        return f;
                                    }
                                },
                                { title: '操作符', dataIndex: 'op', key: 'op' },
                                { title: '判定阈值', dataIndex: 'val', key: 'val' },
                            ]}
                        />
                    </div>
                )}
            </Modal>

            <Modal
                title="特征分箱探索器 (Feature Binning Explorer)"
                open={explorerVisible}
                onCancel={() => setExplorerVisible(false)}
                width={1100}
                footer={null}
                zIndex={99999}
                centered
                getContainer={() => document.body}
                styles={{ mask: { zIndex: 99999 }, wrapper: { zIndex: 99999 } }}
            >
                <Alert message="提示" description="此工具用于快速探索单个变量的分箱分布，辅助制定规则阈值。" type="info" showIcon style={{ marginBottom: 20 }} />
                <Form
                    form={explorerForm}
                    layout="inline"
                    onFinish={handleExploreBinning}
                    initialValues={{ method: 'decision_tree', n_bins: 10, min_samples_leaf: 0.05, label_col: 'label' }}
                    style={{ marginBottom: 24, background: '#fafafa', padding: 16, borderRadius: 8 }}
                >
                    <Form.Item name="dataset_id" label="数据集" rules={[{ required: true }]}>
                        <Select style={{ width: 180 }} placeholder="选择数据集" onChange={(id) => {
                            const ds = datasets.find(d => d.id === id);
                            if (ds) setColumns(Object.keys(ds.columns_info));
                        }}>
                            {datasets.map(ds => <Option key={ds.id} value={ds.id}>{ds.name}</Option>)}
                        </Select>
                    </Form.Item>
                    <Form.Item name="variable" label="探索变量" rules={[{ required: true }]}>
                        <Select style={{ width: 180 }} showSearch placeholder="搜索变量">
                            {columns.map(c => <Option key={c} value={c}>{c}</Option>)}
                        </Select>
                    </Form.Item>
                    <Form.Item name="method" label="分箱方法">
                        <Select style={{ width: 140 }}>
                            <Option value="decision_tree">决策树 (最优)</Option>
                            <Option value="quantile">等频 (Quantile)</Option>
                            <Option value="chi">卡方 (Chi-square)</Option>
                        </Select>
                    </Form.Item>
                    <Form.Item
                        noStyle
                        shouldUpdate={(prev, curr) => prev.method !== curr.method}
                    >
                        {({ getFieldValue }) => (
                            getFieldValue('method') === 'decision_tree' ? (
                                <Space>
                                    <Form.Item name="min_samples_leaf" label="最小样本占比">
                                        <InputNumber step={0.01} min={0.01} max={0.5} style={{ width: 80 }} />
                                    </Form.Item>
                                    <Form.Item name="max_leaf_nodes" label="最大箱数">
                                        <InputNumber min={2} max={20} style={{ width: 60 }} />
                                    </Form.Item>
                                </Space>
                            ) : (
                                <Form.Item name="n_bins" label="目标箱数">
                                    <InputNumber min={2} max={20} style={{ width: 60 }} />
                                </Form.Item>
                            )
                        )}
                    </Form.Item>
                    <Form.Item>
                        <Button type="primary" htmlType="submit" loading={explorerLoading}>开始计算</Button>
                    </Form.Item>
                </Form>

                <Table
                    dataSource={binningData}
                    columns={binningColumns}
                    rowKey="bin"
                    loading={explorerLoading}
                    pagination={false}
                    size="small"
                    bordered
                    summary={(pageData: any) => {
                        let totalIv = 0;
                        pageData.forEach((item: any) => { totalIv += (item.iv || 0); });
                        return (
                            <Table.Summary.Row>
                                <Table.Summary.Cell index={0} colSpan={6}><strong>合计 (Sum)</strong></Table.Summary.Cell>
                                <Table.Summary.Cell index={1}><strong>{totalIv.toFixed(4)}</strong></Table.Summary.Cell>
                                <Table.Summary.Cell index={2} colSpan={2} />
                            </Table.Summary.Row>
                        );
                    }}
                />
            </Modal>

            <Modal
                title="自动化策略推荐 (Strategy Mining)"
                open={miningVisible}
                onCancel={() => setMiningVisible(false)}
                width={1200}
                footer={null}
                zIndex={99999}
                centered
                getContainer={() => document.body}
                styles={{ mask: { zIndex: 99999 }, wrapper: { zIndex: 99999 } }}
            >
                <Form
                    form={miningForm}
                    layout="inline"
                    onFinish={handleStartMining}
                    initialValues={{ tree_type: 'exrf', max_depth: 3, n_estimators: 100, min_samples_leaf: 100 }}
                    style={{ marginBottom: 24, background: '#fafafa', padding: 16, borderRadius: 8 }}
                >
                    <Form.Item name="dataset_id" label="数据集" rules={[{ required: true }]}>
                        <Select style={{ width: 160 }} placeholder="选择数据集">
                            {datasets.map(ds => <Option key={ds.id} value={ds.id}>{ds.name}</Option>)}
                        </Select>
                    </Form.Item>
                    <Form.Item name="tree_type" label="挖掘模型">
                        <Select style={{ width: 120 }}>
                            <Option value="exrf">极端随机森林</Option>
                            <Option value="rf">随机森林</Option>
                            <Option value="dt">单一决策树</Option>
                        </Select>
                    </Form.Item>
                    <Form.Item name="max_depth" label="最大深度">
                        <InputNumber min={2} max={10} style={{ width: 60 }} />
                    </Form.Item>
                    <Form.Item name="min_samples_leaf" label="最小叶子样本">
                        <InputNumber min={10} max={2000} step={50} style={{ width: 80 }} />
                    </Form.Item>
                    <Form.Item>
                        <Button type="primary" htmlType="submit" loading={miningLoading}>启动挖掘</Button>
                    </Form.Item>
                </Form>

                {miningLoading && (
                    <div style={{ textAlign: 'center', padding: '20px 0' }}>
                        <Statistic title="挖掘进度" value={miningProgress} precision={1} suffix="%" />
                        <Progress percent={miningProgress} status="active" strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }} />
                        <p style={{ marginTop: 10, color: '#999' }}>正在通过机器学习算法挖掘高风险规则组合，请稍候...</p>
                    </div>
                )}

                {!miningLoading && recommendations.length > 0 && (
                    <Table
                        dataSource={recommendations}
                        columns={miningColumns}
                        rowKey="rule"
                        size="small"
                        pagination={{ pageSize: 8 }}
                    />
                )}
            </Modal>
        </div>
    );
};

export default StrategyPage;
