import React, { useEffect, useState } from 'react';
import { Table, Card, Typography, Switch, message, Tag, Button, Modal, Input, Space } from 'antd';
import api from '../services/api';
import { useUserStore } from '../stores/userStore';
import { useNavigate } from 'react-router-dom';

const { Title } = Typography;

const AdminUserPage: React.FC = () => {
    const [users, setUsers] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const { user } = useUserStore();
    const navigate = useNavigate();

    useEffect(() => {
        if (user && !user.is_admin) {
            message.error('无权限访问');
            navigate('/');
            return;
        }
        fetchUsers();
    }, [user, navigate]);

    const fetchUsers = async () => {
        setLoading(true);
        try {
            const data: any = await api.get('/v1/users/all');
            setUsers(data);
        } catch (error: any) {
            message.error('获取用户列表失败');
        } finally {
            setLoading(false);
        }
    };

    const handleRoleChange = async (userId: number, isAdmin: boolean) => {
        try {
            await api.put(`/v1/users/${userId}/role`, { is_admin: isAdmin ? 1 : 0 });
            message.success('权限修改成功');
            fetchUsers();
        } catch (error: any) {
            message.error(error.response?.data?.detail || '权限修改失败');
        }
    };

    const columns = [
        {
            title: 'ID',
            dataIndex: 'id',
            key: 'id',
            width: 80,
        },
        {
            title: '用户名',
            dataIndex: 'username',
            key: 'username',
        },
        {
            title: '注册时间',
            dataIndex: 'created_at',
            key: 'created_at',
            render: (text: string) => text ? new Date(text).toLocaleString() : '-',
        },
        {
            title: '角色',
            key: 'role',
            render: (_: any, record: any) => (
                <Tag color={record.is_admin ? 'gold' : 'blue'}>
                    {record.is_admin ? '管理员' : '普通用户'}
                </Tag>
            )
        },
        {
            title: '操作',
            key: 'action',
            render: (_: any, record: any) => (
                <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span>设为管理员:</span>
                        <Switch
                            checked={record.is_admin === 1}
                            onChange={(checked) => handleRoleChange(record.id, checked)}
                            disabled={record.username === 'root'} // Root用户禁止取消管理员
                        />
                    </div>
                    <Button
                        type="link"
                        onClick={() => openProjectModal(record)}
                        style={{ padding: 0 }}
                    >
                        项目管理
                    </Button>
                    <Button
                        type="link"
                        danger
                        onClick={() => handleDeleteUser(record)}
                        disabled={record.username === 'root'}
                        style={{ padding: 0 }}
                    >
                        删除用户
                    </Button>
                    <Button
                        type="link"
                        danger
                        onClick={() => openResetModal(record)}
                        style={{ padding: 0 }}
                    >
                        重置密码
                    </Button>
                </div>
            )
        }
    ];

    // 重置密码 Modal 状态
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [resetTargetUser, setResetTargetUser] = useState<any>(null);
    const [newPassword, setNewPassword] = useState('');
    const [resetting, setResetting] = useState(false);

    // 用户项目管理状态
    const [isProjectModalOpen, setIsProjectModalOpen] = useState(false);
    const [targetUserProjects, setTargetUserProjects] = useState<any[]>([]);
    const [projectLoading, setProjectLoading] = useState(false);

    const openResetModal = (targetUser: any) => {
        setResetTargetUser(targetUser);
        setNewPassword('');
        setIsModalOpen(true);
    };

    const openProjectModal = async (targetUser: any) => {
        setResetTargetUser(targetUser);
        setIsProjectModalOpen(true);
        setProjectLoading(true);
        try {
            const data: any = await api.get(`/v1/users/${targetUser.id}/projects`);
            setTargetUserProjects(data);
        } catch (error) {
            message.error('获取用户项目失败');
        } finally {
            setProjectLoading(false);
        }
    };

    const handleToggleProjectVisibility = async (record: any) => {
        const newStatus = record.is_public === 1 ? 0 : 1;
        try {
            await api.put(`/projects/${record.id}/visibility`, { is_public: newStatus });
            message.success('可见性更新成功');
            // 刷新当前弹窗的项目列表
            const data: any = await api.get(`/v1/users/${resetTargetUser.id}/projects`);
            setTargetUserProjects(data);
        } catch (error) {
            message.error('操作失败');
        }
    };

    const handleDeleteUser = (userToDelete: any) => {
        Modal.confirm({
            title: '确认删除用户？',
            content: `删除用户 "${userToDelete.username}" 将永久移除其下的所有项目数据和物理文件，操作不可撤销！`,
            okText: '确认删除',
            okButtonProps: { danger: true },
            onOk: async () => {
                try {
                    await api.delete(`/v1/users/${userToDelete.id}`);
                    message.success(`用户 ${userToDelete.username} 已彻底删除`);
                    fetchUsers();
                } catch (error: any) {
                    message.error(error.response?.data?.detail || '删除失败');
                }
            }
        });
    };

    const handleResetPassword = async () => {
        if (!newPassword) {
            message.warning('请输入新密码');
            return;
        }
        setResetting(true);
        try {
            await api.put(`/v1/users/${resetTargetUser.id}/reset-password`, { new_password: newPassword });
            message.success(`用户 ${resetTargetUser.username} 密码重置成功`);
            setIsModalOpen(false);
        } catch (error: any) {
            message.error(error.response?.data?.detail || '密码重置失败');
        } finally {
            setResetting(false);
        }
    };

    return (
        <div style={{ padding: 24 }}>
            <Card title={<Title level={4} style={{ margin: 0 }}>用户管理</Title>}>
                <Table
                    columns={columns}
                    dataSource={users}
                    rowKey="id"
                    loading={loading}
                    pagination={{ defaultPageSize: 10 }}
                />
            </Card>

            <Modal
                title={`重置密码: ${resetTargetUser?.username}`}
                open={isModalOpen}
                onOk={handleResetPassword}
                onCancel={() => setIsModalOpen(false)}
                confirmLoading={resetting}
                destroyOnClose
            >
                <div>
                    <p>请输入为该用户设置的新密码：</p>
                    <Input.Password
                        value={newPassword}
                        onChange={(e: any) => setNewPassword(e.target.value)}
                        placeholder="新密码"
                    />
                </div>
            </Modal>

            <Modal
                title={`项目管理: ${resetTargetUser?.username}`}
                open={isProjectModalOpen}
                onCancel={() => setIsProjectModalOpen(false)}
                footer={null}
                width={800}
            >
                <Table
                    size="small"
                    rowKey="id"
                    dataSource={targetUserProjects}
                    loading={projectLoading}
                    columns={[
                        { title: 'ID', dataIndex: 'id', width: 60 },
                        { title: '名称', dataIndex: 'name' },
                        {
                            title: '可见性',
                            dataIndex: 'is_public',
                            render: (v) => v === 1 ? <Tag color="green">公开</Tag> : <Tag color="default">私有</Tag>
                        },
                        { title: '状态', dataIndex: 'status' },
                        {
                            title: '操作',
                            render: (_, record) => (
                                <Space>
                                    <Button size="small" onClick={() => handleToggleProjectVisibility(record)}>
                                        {record.is_public === 1 ? '设为私有' : '设为公开'}
                                    </Button>
                                    <Button size="small" danger onClick={async () => {
                                        Modal.confirm({
                                            title: '确认删除',
                                            onOk: async () => {
                                                await api.delete(`/projects/${record.id}`);
                                                message.success('删除成功');
                                                openProjectModal(resetTargetUser);
                                            }
                                        });
                                    }}>
                                        删除
                                    </Button>
                                </Space>
                            )
                        }
                    ]}
                />
            </Modal>
        </div>
    );
};

export default AdminUserPage;
