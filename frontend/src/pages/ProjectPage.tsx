import React, { useEffect, useState } from 'react';
import { Table, Button, Space, message, Modal, Form, Input, Card, Typography } from 'antd';
import { PlusOutlined, DeleteOutlined, EnterOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import { useAppStore } from '../stores';
import type { Project } from '../types';

const { Title } = Typography;

const ProjectPage: React.FC = () => {
    const [projects, setProjects] = useState<Project[]>([]);
    const [loading, setLoading] = useState(false);
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
                    useAppStore.setState({ currentProjectId: null });
                } catch (error) {
                    message.error('删除失败');
                }
            }
        });
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
        { title: 'ID', dataIndex: 'id', key: 'id', width: 80 },
        { title: '项目名称', dataIndex: 'name', key: 'name' },
        { title: '描述', dataIndex: 'description', key: 'description' },
        { title: '状态', dataIndex: 'status', key: 'status' },
        { title: '创建时间', dataIndex: 'created_at', key: 'created_at', render: (t: string) => new Date(t).toLocaleString() },
        {
            title: '操作',
            key: 'action',
            render: (_: any, record: Project) => (
                <Space size="middle">
                    <Button type="primary" icon={<EnterOutlined />} onClick={() => handleEnterProject(record.id)}>
                        进入应用
                    </Button>
                    <Button danger icon={<DeleteOutlined />} onClick={() => handleDelete(record.id)}>
                        删除
                    </Button>
                </Space>
            ),
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

            <Card bordered={false}>
                <Table
                    rowKey="id"
                    columns={columns}
                    dataSource={projects}
                    loading={loading}
                    pagination={{ defaultPageSize: 10 }}
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
