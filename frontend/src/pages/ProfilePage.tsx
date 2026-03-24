import React, { useState } from 'react';
import { Card, Form, Input, Button, message, Typography, Divider } from 'antd';
import { LockOutlined } from '@ant-design/icons';
import api from '../services/api';
import { useUserStore } from '../stores/userStore';

const { Title, Text } = Typography;

const ProfilePage: React.FC = () => {
    const { user, logout } = useUserStore();
    const [loading, setLoading] = useState(false);
    const [form] = Form.useForm();

    const onFinish = async (values: any) => {
        if (values.new_password !== values.confirm_password) {
            message.error('两次输入的新密码不一致');
            return;
        }
        setLoading(true);
        try {
            await api.put('/v1/users/me/password', {
                old_password: values.old_password,
                new_password: values.new_password
            });
            message.success('密码修改成功，请重新登录');
            form.resetFields();
            setTimeout(() => {
                logout();
            }, 1000);
        } catch (error: any) {
            message.error(error.response?.data?.detail || '密码修改失败');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ padding: 24, maxWidth: 600, margin: '0 auto' }}>
            <Card title={<Title level={4} style={{ margin: 0 }}>个人中心</Title>}>
                <div style={{ marginBottom: 24 }}>
                    <Text type="secondary">用户名：</Text>
                    <Text strong>{user?.username}</Text>
                    <Divider type="vertical" />
                    <Text type="secondary">角色：</Text>
                    <Text strong>{user?.is_admin ? '管理员' : '普通用户'}</Text>
                </div>
                <Divider />
                <Title level={5}>修改密码</Title>
                <Form form={form} layout="vertical" onFinish={onFinish} style={{ marginTop: 24 }}>
                    <Form.Item name="old_password" label="原密码" rules={[{ required: true, message: '请输入原密码' }]}>
                        <Input.Password prefix={<LockOutlined />} placeholder="原密码" />
                    </Form.Item>
                    <Form.Item name="new_password" label="新密码" rules={[{ required: true, message: '请输入新密码' }]}>
                        <Input.Password prefix={<LockOutlined />} placeholder="新密码" />
                    </Form.Item>
                    <Form.Item name="confirm_password" label="确认新密码" rules={[{ required: true, message: '请再次输入新密码' }]}>
                        <Input.Password prefix={<LockOutlined />} placeholder="确认新密码" />
                    </Form.Item>
                    <Form.Item>
                        <Button type="primary" htmlType="submit" loading={loading}>
                            提交修改
                        </Button>
                    </Form.Item>
                </Form>
            </Card>
        </div>
    );
};

export default ProfilePage;
