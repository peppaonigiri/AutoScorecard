import React, { useState, useEffect } from 'react';
import { Card, Upload, Button, message, Table, Tabs, Descriptions, Statistic, Row, Col, Tag, Typography, InputNumber, Form, Space, Alert, Divider, Select, Popconfirm } from 'antd';
import { UploadOutlined, CheckCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import api from '../services/api';
import { useAppStore } from '../stores';

const { TabPane } = Tabs;
const { Option } = Select;
const { Text } = Typography;

const DatasetPage: React.FC = () => {
    const {
        currentProjectId, currentDatasetId, setCurrentDatasetId,
        datasetInfo, setDatasetInfo,
        datasetStats, setDatasetStats,
        previewData, setPreviewData,
        excludeCols, setExcludeCols
    } = useAppStore();

    const [uploading, setUploading] = useState(false);
    const [l1Loading, setL1Loading] = useState(false);
    const [imputing, setImputing] = useState(false);
    const [fillValue, setFillValue] = useState<number>(-999);
    const [datasets, setDatasets] = useState<any[]>([]);
    const [l1Form] = Form.useForm();

    // 初始加载及项目切换时拉取数据集列表
    useEffect(() => {
        if (currentProjectId) {
            fetchDatasetList();
        }
    }, [currentProjectId]);

    // 当 currentDatasetId 已有值但 stats 为空时（例如切回页面），自动加载详情
    useEffect(() => {
        if (currentDatasetId && !datasetStats) {
            fetchStats(currentDatasetId);
        }
        if (currentDatasetId && !previewData) {
            fetchPreview(currentDatasetId);
        }
    }, [currentDatasetId]);

    const fetchDatasetList = async () => {
        try {
            const res: any = await api.get(`/projects/${currentProjectId}/datasets`);
            setDatasets(res);
            // 如果当前没有选中数据集，但列表里有，则默认选中第一个
            if (!currentDatasetId && res.length > 0) {
                selectDataset(res[res.length - 1]);
            }
        } catch (err) {
            console.error('获取数据集列表失败', err);
        }
    };

    const selectDataset = (ds: any) => {
        setDatasetInfo(ds);
        setCurrentDatasetId(ds.id);
        if (ds.stats_cache) {
            setDatasetStats(ds.stats_cache);
        } else {
            fetchStats(ds.id);
        }
        fetchPreview(ds.id);
    };

    // 获取项目的持久化配置（如全局排除列）
    useEffect(() => {
        if (currentProjectId) {
            api.get(`/projects/${currentProjectId}`).then((res: any) => {
                if (res.exclude_cols) {
                    setExcludeCols(res.exclude_cols);
                } else {
                    setExcludeCols([]);
                }
            }).catch(e => console.error('加载项目配置失败', e));
        }
    }, [currentProjectId]);

    const handleExcludeColsChange = async (cols: string[]) => {
        setExcludeCols(cols);
        if (currentProjectId) {
            try {
                await api.put(`/projects/${currentProjectId}/exclude-cols`, { exclude_cols: cols });
            } catch (e) {
                console.error('持久化保存排除列失败', e);
                message.error('保存全局排除列失败');
            }
        }
    };

    const handleUpload = async (options: any) => {
        const { file, onSuccess, onError } = options;
        const formData = new FormData();
        formData.append('file', file);

        setUploading(true);
        try {
            const res: any = await api.post(`/projects/${currentProjectId}/datasets`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            message.success('上传成功');
            selectDataset(res); // 立即选中新上传的
            fetchDatasetList(); // 随后刷新下拉列表
            onSuccess?.(res);
        } catch (error) {
            message.error('上传失败');
            onError?.(error);
        } finally {
            setUploading(false);
        }
    };

    const fetchStats = async (datasetId: number) => {
        try {
            const res = await api.get(`/datasets/${datasetId}/stats`);
            setDatasetStats(res);
        } catch (error) {
            message.error('获取统计信息失败');
        }
    };

    const fetchPreview = async (datasetId: number) => {
        try {
            const res = await api.get(`/datasets/${datasetId}/preview`);
            setPreviewData(res);
        } catch (error) {
            message.error('获取数据预览失败');
        }
    };

    if (!currentProjectId) {
        return <div>请先在左侧选择项目</div>;
    }

    const handleDeleteDataset = async (dsId: number, dsName: string) => {
        try {
            await api.delete(`/datasets/${dsId}`);
            message.success(`数据集 "${dsName}" 已删除`);
            // 如果删除的是当前选中的，清空选中状态
            if (dsId === currentDatasetId) {
                setDatasetInfo(null);
                setCurrentDatasetId(null);
                setDatasetStats(null);
                setPreviewData(null);
            }
            fetchDatasetList();
        } catch (err: any) {
            const detail = err.response?.data?.detail || '删除失败';
            message.error(detail);
        }
    };

    // 构建预览表格的 columns
    const previewColumns = previewData?.columns?.map((col: string) => ({
        title: col,
        dataIndex: col,
        key: col,
        width: 120,
        ellipsis: true,
    })) || [];

    // 构建统计信息的 columns
    const statsColumns = [
        { title: '变量名', dataIndex: 'name', key: 'name' },
        { title: '类型', dataIndex: 'dtype', key: 'dtype' },
        {
            title: '缺失率',
            dataIndex: 'missing_rate',
            key: 'missing_rate',
            render: (val: number) => val != null ? `${(val * 100).toFixed(2)}%` : '-'
        },
        {
            title: '均值',
            dataIndex: 'mean',
            key: 'mean',
            render: (val: number) => val != null ? val.toFixed(4) : '-'
        },
        {
            title: '标准差',
            dataIndex: 'std',
            key: 'std',
            render: (val: number) => val != null ? val.toFixed(4) : '-'
        },
    ];

    const statsDataSource = datasetStats?.columns?.map((col: string) => ({
        name: col,
        dtype: datasetStats.dtypes?.[col],
        missing_rate: datasetStats.missing_rates?.[col],
        mean: datasetStats.numeric_stats?.[col]?.mean,
        std: datasetStats.numeric_stats?.[col]?.std,
    })) || [];

    // L1 初筛独立触发
    const runL1Screen = async () => {
        if (!currentDatasetId) return;
        setL1Loading(true);
        try {
            const vals = l1Form.getFieldsValue();
            // 调用后端重新计算（通过一个简易的 filter 接口，仅做 L1）
            const res: any = await api.post(`/projects/${currentProjectId}/feature/filter`, {
                dataset_id: currentDatasetId,
                dep: 'label',
                exclude_cols: Array.from(new Set(['id', 'uuid', 'user_id', 'date', ...excludeCols])),
                thresholds: {
                    missing: vals.missing_threshold,
                    freq: vals.single_value_threshold,
                    iv: 0,
                    corr: 1.0,
                    psi: 1.0
                },
                skip_l1: false
            });
            // 更新 datasetInfo 中的 l1_results 来源
            setDatasetInfo({
                ...datasetInfo,
                l1_results: {
                    removed_info: res.dropped_features,
                    kept: res.kept_features
                }
            });
            message.success(`L1 初筛完成：保留 ${res.summary.total_kept} 个变量，剔除 ${res.summary.total_dropped} 个`);
        } catch (err) {
            message.error('L1 初筛失败');
        } finally {
            setL1Loading(false);
        }
    };

    const handleImpute = async () => {
        if (!currentDatasetId) return;
        setImputing(true);
        try {
            const res: any = await api.post(`/datasets/${currentDatasetId}/impute`, {
                fill_value: fillValue,
                exclude_cols: Array.from(new Set(['id', 'uuid', 'user_id', 'date', ...excludeCols]))
            });
            message.success(res.message || '缺失值填充成功，已生成新数据集');

            // 重新拉取数据集列表并动态选取
            const dsRes: any = await api.get(`/projects/${currentProjectId}/datasets`);
            setDatasets(dsRes);

            // 如果后端返回了新生成的数据集分支 ID，自动切过去
            if (res.new_dataset_id) {
                const newDs = dsRes.find((d: any) => d.id === res.new_dataset_id);
                if (newDs) {
                    selectDataset(newDs);
                }
            } else {
                fetchStats(currentDatasetId);
                fetchPreview(currentDatasetId);
            }
        } catch (err: any) {
            const detail = err.response?.data?.detail || '填充失败';
            message.error(detail);
        } finally {
            setImputing(false);
        }
    };

    const l1Info = datasetInfo?.l1_results;

    return (
        <div>
            <Card title="项目数据集管理" style={{ marginBottom: 24 }}>
                <Row gutter={16} align="middle">
                    <Col span={12}>
                        <div style={{ marginBottom: 8 }}>已有数据集：</div>
                        <Select
                            style={{ width: '100%' }}
                            placeholder="请选择或上传数据集"
                            value={currentDatasetId || undefined}
                            onChange={(id) => {
                                const ds = datasets.find(d => d.id === id);
                                if (ds) selectDataset(ds);
                            }}
                        >
                            {datasets.map(ds => (
                                <Option key={ds.id} value={ds.id}>
                                    {ds.name} (行:{ds.n_rows}) - {new Date(ds.created_at).toLocaleString()}
                                    {datasets.length > 1 && (
                                        <Popconfirm
                                            title={`确认删除数据集 "${ds.name}" 吗？物理文件也将被清除。`}
                                            onConfirm={(e) => { e?.stopPropagation(); handleDeleteDataset(ds.id, ds.name); }}
                                            onCancel={(e) => e?.stopPropagation()}
                                            okText="确认删除"
                                            cancelText="取消"
                                        >
                                            <DeleteOutlined
                                                style={{ color: '#ff4d4f', marginLeft: 8, float: 'right' }}
                                                onClick={(e) => e.stopPropagation()}
                                            />
                                        </Popconfirm>
                                    )}
                                </Option>
                            ))}
                        </Select>
                    </Col>
                    <Col span={12} style={{ paddingTop: 24 }}>
                        <Space>
                            <Upload
                                customRequest={handleUpload}
                                showUploadList={false}
                                accept=".csv,.parquet"
                            >
                                <Button type="primary" icon={<UploadOutlined />} loading={uploading}>
                                    上传新数据集
                                </Button>
                            </Upload>
                            {datasetInfo && (
                                <Button onClick={() => { setDatasetInfo(null); setCurrentDatasetId(null); setDatasetStats(null); }}>
                                    取消当前选择
                                </Button>
                            )}
                        </Space>
                    </Col>
                </Row>

                {datasetInfo && (
                    <>
                        <Divider />
                        <Descriptions column={3}>
                            <Descriptions.Item label="当前选中名称">{datasetInfo.name}</Descriptions.Item>
                            <Descriptions.Item label="数据规模">{datasetInfo.n_rows} × {datasetInfo.n_cols}</Descriptions.Item>
                            <Descriptions.Item label="上传时间">{new Date(datasetInfo.created_at).toLocaleString()}</Descriptions.Item>
                        </Descriptions>
                        <div style={{ marginTop: 20 }}>
                            <div style={{ marginBottom: 8 }}>
                                <Alert
                                    message={<Text strong>配置全局排除列 (不参与后续所有统计与入模)</Text>}
                                    description="在此处选择的字段将被标记为非特征，系统会自动将其从 IV 计算、变量筛选及模型训练中剔除。"
                                    type="warning"
                                    showIcon
                                />
                            </div>
                            <Select
                                mode="multiple"
                                style={{ width: '100%' }}
                                placeholder="点此选择或搜索需要排除的字段（如 ID、Date 等）"
                                value={excludeCols}
                                onChange={handleExcludeColsChange}
                                allowClear
                                options={datasetStats?.columns?.map((c: string) => ({ label: c, value: c }))}
                            />
                        </div>
                    </>
                )}
            </Card>

            {/* ===== L1 初筛操作面板 ===== */}
            {datasetInfo && (
                <Card title="第一层筛选：基础变量质量检查" style={{ marginBottom: 24 }}>
                    <Form
                        form={l1Form}
                        layout="inline"
                        initialValues={{
                            single_value_threshold: 0.95,
                            missing_threshold: 0.95
                        }}
                    >
                        <Form.Item name="single_value_threshold" label="单一值比例阈值">
                            <InputNumber min={0} max={1} step={0.05} style={{ width: 100 }} />
                        </Form.Item>
                        <Form.Item name="missing_threshold" label="缺失率阈值">
                            <InputNumber min={0} max={1} step={0.05} style={{ width: 100 }} />
                        </Form.Item>
                        <Form.Item>
                            <Button type="primary" onClick={runL1Screen} loading={l1Loading}>
                                执行 L1 初筛
                            </Button>
                        </Form.Item>
                    </Form>

                    {l1Info && l1Info.kept && (
                        <Alert
                            style={{ marginTop: 16 }}
                            type="info"
                            showIcon
                            icon={<CheckCircleOutlined />}
                            message={
                                <span>
                                    L1 初筛统计结果：保留 <Text strong style={{ color: '#52c41a' }}>{l1Info.kept?.length || 0}</Text> 个变量，
                                    剔除 <Text strong style={{ color: '#f5222d' }}>
                                        {(l1Info.removed_info?.missing?.length || 0) +
                                            (l1Info.removed_info?.freq?.length || 0) +
                                            (l1Info.removed_info?.zero_std?.length || 0)}
                                    </Text> 个
                                </span>
                            }
                            description={
                                <div style={{ marginTop: 12 }}>
                                    <Row gutter={[8, 8]}>
                                        {l1Info.removed_info?.freq?.length > 0 && (
                                            <Col span={24}>
                                                <div style={{ marginBottom: 4 }}><Tag color="orange">单一值过高</Tag></div>
                                                <Space wrap>
                                                    {l1Info.removed_info.freq.map((v: string) => <Tag key={v}>{v}</Tag>)}
                                                </Space>
                                            </Col>
                                        )}
                                        {l1Info.removed_info?.missing?.length > 0 && (
                                            <Col span={24}>
                                                <div style={{ marginBottom: 4, marginTop: 8 }}><Tag color="magenta">缺失率过高</Tag></div>
                                                <Space wrap>
                                                    {l1Info.removed_info.missing.map((v: string) => <Tag key={v}>{v}</Tag>)}
                                                </Space>
                                            </Col>
                                        )}
                                        {l1Info.removed_info?.zero_std?.length > 0 && (
                                            <Col span={24}>
                                                <div style={{ marginBottom: 4, marginTop: 8 }}><Tag color="red">零标准差</Tag></div>
                                                <Space wrap>
                                                    {l1Info.removed_info.zero_std.map((v: string) => <Tag key={v}>{v}</Tag>)}
                                                </Space>
                                            </Col>
                                        )}
                                        {!(l1Info.removed_info?.freq?.length || l1Info.removed_info?.missing?.length || l1Info.removed_info?.zero_std?.length) && (
                                            <Col span={24}><Text type="secondary">暂无变量因统计异常被剔除</Text></Col>
                                        )}
                                    </Row>
                                </div>
                            }
                        />
                    )}
                </Card>
            )}

            {/* ===== 缺失值填充操作面板 ===== */}
            {datasetInfo && (
                <Card title="数据清洗：缺失值填充" style={{ marginBottom: 24 }}>
                    <div style={{ marginBottom: 16 }}>
                        <Text type="secondary">
                            在 L1 筛选完成后，您可以对剩余变量进行缺失值补全。本功能会将除排除列外的所有 **数值型字段** 中的空值（NaN）用给定的常数进行填补（默认 <Text strong type="danger">-999</Text>）。
                            <br />
                            <Text type="warning">注：操作将不会覆盖当前文件，而是会自动生成一个以 "_已填充" 为后缀的全新数据集以保障安全对照。</Text>
                        </Text>
                    </div>
                    <Space>
                        <span style={{ marginLeft: 4 }}>填充值：</span>
                        <InputNumber
                            value={fillValue}
                            onChange={(val) => setFillValue(val as number)}
                            style={{ width: 120 }}
                        />
                        <Button type="primary" onClick={handleImpute} loading={imputing}>
                            保存并执行填充
                        </Button>
                    </Space>
                </Card>
            )}

            {datasetStats && (
                <Card title="数据总览" style={{ marginBottom: 24 }}>
                    <Row gutter={16}>
                        <Col span={6}>
                            <Statistic title="总行数" value={datasetStats.n_rows} />
                        </Col>
                        <Col span={6}>
                            <Statistic title="总列数" value={datasetStats.n_cols} />
                        </Col>
                    </Row>
                </Card>
            )}

            {(datasetStats || previewData) && (
                <Card>
                    <Tabs defaultActiveKey="1">
                        <TabPane tab="字段统计" key="1">
                            <Table
                                dataSource={statsDataSource}
                                columns={statsColumns}
                                rowKey="name"
                                pagination={{ pageSize: 15 }}
                            />
                        </TabPane>
                        <TabPane tab="数据预览 (前100行)" key="2">
                            <Table
                                dataSource={previewData?.data || []}
                                columns={previewColumns}
                                rowKey={(_record: any, index: any) => index as number}
                                scroll={{ x: 'max-content', y: 400 }}
                                pagination={false}
                            />
                        </TabPane>
                    </Tabs>
                </Card>
            )}
        </div>
    );
};

export default DatasetPage;
