import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Form, Input, Select, InputNumber, Button, Steps, Progress, message, Result, Typography, Alert, Divider, Radio } from 'antd';
import { PlayCircleOutlined } from '@ant-design/icons';
import api from '../services/api';
import { useAppStore } from '../stores';

const { Option } = Select;
const { Title, Text } = Typography;

const STRATEGIES = [
    {
        group: 'KS 稳健性 (KS Stability Focus)',
        options: [
            { value: 1, label: 'Type 1 - OOT KS 惩罚平衡', desc: '以 OOT KS 为目标，同时对 Train/Valid/OOT 之间的偏离进行轻度惩罚，追求整体表现的均衡。' },
            { value: 3, label: 'Type 3 - 严格 KS 对齐控', desc: '要求 Train/Valid/OOT 三集 KS 差值必须小于设定的【差异阈值】，否则给予极度惩罚，是追求绝对稳健的最佳选择。' },
            { value: 10, label: 'Type 10 - 跨集均值优先', desc: '取验证集与 OOT 的 KS 平均值作为目标，降低单集波动对模型的影响。' },
            { value: 1349, label: 'Type 1349 - 极致收敛策略', desc: '比 Type 3 拥有更严苛的收敛偏好，适用于数据量巨大且必须极致对齐的金融风控场景。' },
        ],
    },
    {
        group: 'AUC 与概率性能 (AUC & Distribution)',
        options: [
            { value: 20, label: 'Type 20 - AUC 对齐优化', desc: '在确保 Train/Valid AUC 差异受控的前提下，最大化验证集的 AUC 表现。' },
            { value: 26, label: 'Type 26 - 三集 AUC 绝对对齐', desc: '将 AUC 指标在三集上的极致分布对齐作为首要目标，防止泛化能力衰减。' },
            { value: 51, label: 'Type 51 - KL 散度约束', desc: '在保证 KS 表现的同时，通过 KL 散度约束不同样本集下的概率分布一致性。' },
        ],
    },
    {
        group: '业务增益类 (Business Value Lift)',
        options: [
            { value: 201, label: 'Type 201 - Top 5% Lift 引导', desc: '综合考虑 KS 与前 5% 高分样本的 Lift 提升度，适合目标为“精挑细选头部人群”的业务场景。' },
        ],
    },
];

const ModelingPage: React.FC = () => {
    const { currentProjectId, currentDatasetId, selectedFeatures, excludeCols } = useAppStore();
    const navigate = useNavigate();
    const [form] = Form.useForm();
    const strategyType = Form.useWatch('strategy_type', form);
    const scoreMode = Form.useWatch(['score_config', 'mode'], form);

    const [taskId, setTaskId] = useState<number | null>(null);
    const [taskStatus, setTaskStatus] = useState<string>('');
    const [progress, setProgress] = useState<number>(0);
    const [taskResult, setTaskResult] = useState<any>(null);
    const [errorMsg, setErrorMsg] = useState<string>('');

    const timerRef = useRef<any>(null);

    const startTraining = async (values: any) => {
        if (!currentProjectId || !currentDatasetId) {
            message.warning('请先完成数据上传与项目选择');
            return;
        }
        try {
            const payload = {
                dataset_id: currentDatasetId,
                dep: values.dep,
                exclude_cols: Array.from(new Set(['id', 'uuid', 'user_id', 'date', ...excludeCols])),
                feature_list: selectedFeatures.length > 0 ? selectedFeatures : [],
                model_type: values.model_type,
                strategy_type: values.strategy_type,
                strategy_threshold: values.strategy_threshold,
                n_trials: values.n_trials,
                max_depth: values.max_depth,
                // 透传划分参数
                oot_col: values.oot_col,
                oot_start_time: values.oot_start_time,
                // 评分配置参数
                score_config: {
                    mode: values.score_config.mode,
                    base_score: values.score_config.base_score,
                    pdo: values.score_config.pdo,
                    base_odds: values.score_config.base_odds,
                    min_score: values.score_config.min_score,
                    max_score: values.score_config.max_score
                }
            };

            const res: any = await api.post(`/projects/${currentProjectId}/modeling/train`, payload);
            setTaskId(res.task_id);
            setTaskStatus('pending');
            setProgress(0);
            message.success('训练任务已提交');
            startPolling(res.task_id);
        } catch (err) {
            message.error('启动训练失败');
        }
    };

    const startPolling = (tid: number) => {
        if (timerRef.current) clearInterval(timerRef.current);
        timerRef.current = setInterval(async () => {
            try {
                const res: any = await api.get(`/tasks/${tid}`);
                setTaskStatus(res.status);
                setProgress(Math.floor(res.progress));

                // 发送心跳信号，告知后端我还在线
                try {
                    await api.post(`/tasks/${tid}/heartbeat`);
                } catch (hErr) {
                    console.warn('Heartbeat failed', hErr);
                }

                if (res.status === 'completed') {
                    clearInterval(timerRef.current);
                    setTaskResult(res.result);
                    message.success('训练完毕!');
                } else if (res.status === 'failed') {
                    clearInterval(timerRef.current);
                    setErrorMsg(res.error_msg);
                    message.error('训练失败');
                }
            } catch (err) {
                console.error('轮询状态错误', err);
            }
        }, 2000);
    };

    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, []);

    return (
        <div>
            <Title level={4}>AutoModeling 模型训练</Title>

            <Card title="配置调优参数" style={{ marginBottom: 24 }}>
                {selectedFeatures.length > 0 && (
                    <div style={{ marginBottom: 16 }}>
                        <Text type="secondary">当前将基于变量分析筛选出的 </Text>
                        <Text strong style={{ color: '#1677ff', fontSize: 16 }}>{selectedFeatures.length}</Text>
                        <Text type="secondary"> 个特征进行训练</Text>
                    </div>
                )}
                <Form
                    form={form}
                    layout="vertical"
                    onFinish={startTraining}
                    initialValues={{
                        dep: 'label',
                        model_type: 'xgb',
                        strategy_type: 1,
                        strategy_threshold: 0.03,
                        n_trials: 50,
                        max_depth: 6,
                        score_config: {
                            mode: 'manual',
                            base_score: 600,
                            pdo: 20,
                            base_odds: 50,
                            min_score: 300,
                            max_score: 900
                        }
                    }}
                    disabled={taskStatus === 'pending' || taskStatus === 'running'}
                >
                    <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                        <Form.Item name="dep" label="目标变量名称">
                            <Input />
                        </Form.Item>
                        <Form.Item name="model_type" label="算法模型">
                            <Select style={{ width: 160 }}>
                                <Option value="xgb">XGBoost</Option>
                                <Option value="lgb">LightGBM</Option>
                                <Option value="lr">Logistic Regression</Option>
                            </Select>
                        </Form.Item>
                        <Form.Item name="strategy_type" label="评估策略 (Optuna Strategy)" style={{ minWidth: 280 }}>
                            <Select dropdownMatchSelectWidth={false}>
                                {STRATEGIES.map(group => (
                                    <Select.OptGroup key={group.group} label={group.group}>
                                        {group.options.map(opt => (
                                            <Option key={opt.value} value={opt.value}>
                                                {opt.label}
                                            </Option>
                                        ))}
                                    </Select.OptGroup>
                                ))}
                            </Select>
                        </Form.Item>

                        <Form.Item label="策略容忍度 (阈值)" name="strategy_threshold" style={{ width: 220 }} tooltip="此阈值控制 Train/Valid/OOT 之间差异的允许范围。越小则越要求模型三集表现严丝合缝，但搜索空间会变窄。">
                            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                                <InputNumber min={0.005} max={0.2} step={0.005} precision={3} style={{ width: 100 }} />
                                <Text type="secondary" style={{ fontSize: '12px' }}>极小意味着强制对齐</Text>
                            </div>
                        </Form.Item>

                        <Form.Item name="n_trials" label="搜索次数 (Trials)">
                            <InputNumber min={5} max={1000} />
                        </Form.Item>
                        <Form.Item name="max_depth" label="最大树深度 (Max Depth)">
                            <InputNumber min={1} max={15} />
                        </Form.Item>
                    </div>

                    <Divider>评分卡参数配置 (Scorecard Params)</Divider>
                    <Form.Item name={['score_config', 'mode']} label="参数模式">
                        <Radio.Group buttonStyle="solid">
                            <Radio.Button value="manual">手动配置 (Manual)</Radio.Button>
                            <Radio.Button value="auto">自动寻参 (Auto Mode)</Radio.Button>
                        </Radio.Group>
                    </Form.Item>

                    {scoreMode === 'manual' ? (
                        <div style={{ display: 'flex', gap: 24, marginBottom: 12 }}>
                            <Form.Item name={['score_config', 'base_score']} label="基准分 (Base Score)" tooltip="对应基准胜率的目标分数">
                                <InputNumber min={1} max={3000} style={{ width: 140 }} />
                            </Form.Item>
                            <Form.Item name={['score_config', 'base_odds']} label="基准胜率 (Base Odds)" tooltip="对应基准分的 Odds (Good/Bad)">
                                <InputNumber min={0.1} max={1000} style={{ width: 140 }} />
                            </Form.Item>
                            <Form.Item name={['score_config', 'pdo']} label="PDO (Points to Double Odds)" tooltip="Odds 翻倍时分数增加的值">
                                <InputNumber min={1} max={200} style={{ width: 140 }} />
                            </Form.Item>
                        </div>
                    ) : (
                        <div style={{ display: 'flex', gap: 24, marginBottom: 12 }}>
                            <Form.Item name={['score_config', 'min_score']} label="最小分数限制">
                                <InputNumber min={0} max={1000} style={{ width: 140 }} />
                            </Form.Item>
                            <Form.Item name={['score_config', 'max_score']} label="最大分数限制">
                                <InputNumber min={500} max={3000} style={{ width: 140 }} />
                            </Form.Item>
                            <div style={{ alignSelf: 'center', color: '#888', fontStyle: 'italic' }}>
                                * 系统将自动寻找最优 A/B 参数使分数尽量分布在以上区间内
                            </div>
                        </div>
                    )}

                    {/* 策略详细说明区 */}
                    <Alert
                        message="策略定义详解"
                        description={
                            <div style={{ maxHeight: 60, overflowY: 'auto' }}>
                                <Text code>
                                    {STRATEGIES.flatMap(g => g.options).find(o => o.value === strategyType)?.desc || "尚未选择策略"}
                                </Text>
                            </div>
                        }
                        type="info"
                        showIcon
                        style={{ marginBottom: 20 }}
                    />
                    <Form.Item>
                        <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />} size="large">
                            开始训练
                        </Button>
                    </Form.Item>
                </Form>
            </Card>

            {taskId && (
                <Card title="训练实时进度">
                    <Steps
                        current={
                            taskStatus === 'pending' ? 0 :
                                taskStatus === 'running' ? 1 :
                                    taskStatus === 'failed' ? 1 : 2
                        }
                        status={taskStatus === 'failed' ? 'error' : 'process'}
                        items={[
                            { title: '排队等待', description: '任务已进入队列' },
                            { title: '模型探索中', description: 'Optuna 超参数随机搜索' },
                            { title: '训练完成', description: '保存最终模型' },
                        ]}
                        style={{ marginBottom: 32 }}
                    />

                    {taskStatus === 'running' && (
                        <div style={{ textAlign: 'center', margin: '40px 0' }}>
                            <Progress type="circle" percent={progress} />
                            <p style={{ marginTop: 16 }}>优化正在进行，请勿关闭页面...</p>
                        </div>
                    )}

                    {taskStatus === 'completed' && taskResult && (
                        <Result
                            status="success"
                            title="模型训练完成！"
                            subTitle={`已执行 ${taskResult.total_trials} 次调参探索。最佳得分：${taskResult.best_value?.toFixed(4)} `}
                            extra={[
                                <Button type="primary" key="console" onClick={() => navigate('/results')}>
                                    查看详细模型报告
                                </Button>,
                            ]}
                        />
                    )}

                    {taskStatus === 'failed' && (
                        <Result
                            status="error"
                            title="任务执行失败"
                            subTitle={errorMsg}
                        />
                    )}
                </Card>
            )}
        </div>
    );
};

export default ModelingPage;
