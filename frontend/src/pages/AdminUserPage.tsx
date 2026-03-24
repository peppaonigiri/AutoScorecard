import React, { useEffect, useState } from 'react';
import { Table, Card, Typography, Switch, message, Tag, Button, Modal, Input } from 'antd';
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

    const openResetModal = (targetUser: any) => {
        setResetTargetUser(targetUser);
        setNewPassword('');
        setIsModalOpen(true);
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
        </div>
    );
};

export default AdminUserPage;
