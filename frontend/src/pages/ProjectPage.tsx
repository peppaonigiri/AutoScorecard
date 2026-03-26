import React, { useEffect, useState } from 'react';
import { Table, Button, Space, message, Modal, Form, Input, Card, Typography, Tag } from 'antd';
import { PlusOutlined, DeleteOutlined, EnterOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import { useAppStore } from '../stores';
import { useUserStore } from '../stores/userStore';
import type { Project } from '../types';

const { Title } = Typography;

const ProjectPage: React.FC = () => {
    const [projects, setProjects] = useState<Project[]>([]);
    const [loading, setLoading] = useState(false);
    const { user } = useUserStore();
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [form] = Form.useForm();

    const navigate = useNavigate();

    const fetchProjects = async () => {
        setLoading(true);
        try {
            const res: any = await api.get('/projects');
            setProjects(res.items || []);
        } catch (error) {
            message.error('获取项目列表失败');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchProjects();
    }, []);

    const handleCreate = async (values: any) => {
        try {
            await api.post('/projects', values);
            message.success('创建成功');
            setIsModalVisible(false);
            form.resetFields();
            fetchProjects();
        } catch (error) {
            message.error('创建失败');
        }
    };

    const handleDelete = async (id: number) => {
        Modal.confirm({
            title: '确认删除',
            content: '删除项目将丢失所有相关数据和模型结果，无法恢复！',
            onOk: async () => {
                try {
                    await api.delete(`/projects/${id}`);
                    message.success('删除成功');
                    fetchProjects();
                    if (useAppStore.getState().currentProjectId === id) {
                        useAppStore.setState({ currentProjectId: null });
                    }
                } catch (error) {
                    message.error('删除失败');
                }
            }
        });
    };

    const handleToggleVisibility = async (record: Project) => {
        const newStatus = record.is_public === 1 ? 0 : 1;
        try {
            await api.put(`/projects/${record.id}/visibility`, { is_public: newStatus });
            message.success(newStatus === 1 ? '项目已公开' : '项目已设为私有');
            fetchProjects();
        } catch (error) {
            message.error('修改可见性失败');
        }
    };

    const handleEnterProject = (id: number) => {
        // 进入新项目前，清空旧项目的上下文
        useAppStore.setState({
            currentProjectId: id,
            currentDatasetId: null,
            datasetInfo: null,
            datasetStats: null,
            previewData: null,
            ivReport: [],
            filterResult: null,
            selectedFeatures: []
        });
        navigate('/dataset');
        message.success(`已进入项目 ${id}`);
    };

    const columns = [
        { title: '项目名称', dataIndex: 'name', key: 'name', width: 140 },
        { title: '所有者', dataIndex: 'owner_name', key: 'owner_name', width: 100, render: (name: string) => name || '系统' },
        {
            title: '可见性',
            dataIndex: 'is_public',
            key: 'is_public',
            width: 80,
            render: (v: number) => v === 1 ? <Tag color="green">公开</Tag> : <Tag color="default">私有</Tag>
        },
        { title: '描述', dataIndex: 'description', key: 'description', ellipsis: true },
        { title: '状态', dataIndex: 'status', key: 'status', width: 90 },
        { title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 150, render: (t: string) => new Date(t).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) },
        {
            title: '操作',
            key: 'action',
            fixed: 'right' as const,
            width: 260,
            render: (_: any, record: Project) => {
                const isOwner = record.owner_id === user?.id;
                const isAdmin = user?.is_admin === 1;
                return (
                    <Space size="small">
                        <Button type="primary" size="small" icon={<EnterOutlined />} onClick={() => handleEnterProject(record.id)}>
                            进入
                        </Button>
                        {(isOwner || isAdmin) && (
                            <Button
                                size="small"
                                onClick={() => handleToggleVisibility(record)}
                            >
                                {record.is_public === 1 ? '设为私有' : '公开'}
                            </Button>
                        )}
                        {(isOwner || isAdmin) && (
                            <Button danger size="small" icon={<DeleteOutlined />} onClick={() => handleDelete(record.id)}>
                                删除
                            </Button>
                        )}
                    </Space>
                );
            },
        },
    ];

    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
                <Title level={4}>项目管理</Title>
                <Button type="primary" icon={<PlusOutlined />} onClick={() => setIsModalVisible(true)}>
                    新建项目
                </Button>
            </div>

            <Card variant="outlined">
                <Table
                    rowKey="id"
                    columns={columns}
                    dataSource={projects}
                    loading={loading}
                    pagination={{ defaultPageSize: 10 }}
                    scroll={{ x: 1000 }}
                />
            </Card>

            <Modal
                title="新建项目"
                open={isModalVisible}
                onCancel={() => setIsModalVisible(false)}
                onOk={() => form.submit()}
            >
                <Form form={form} layout="vertical" onFinish={handleCreate}>
                    <Form.Item name="name" label="项目名称" rules={[{ required: true, message: '请输入项目名称' }]}>
                        <Input placeholder="例如: A卡模型_v1" />
                    </Form.Item>
                    <Form.Item name="description" label="项目描述">
                        <Input.TextArea rows={3} placeholder="选填..." />
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

export default ProjectPage;
