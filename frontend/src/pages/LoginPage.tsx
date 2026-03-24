import React, { useState } from 'react';
import { Card, Form, Input, Button, Tabs, message, Typography } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import { useUserStore } from '../stores/userStore';

const { Title } = Typography;

const LoginPage: React.FC = () => {
    const [activeTab, setActiveTab] = useState('login');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();
    const { setToken, setUser } = useUserStore();

    const onLogin = async (values: any) => {
        setLoading(true);
        try {
            const formData = new URLSearchParams();
            formData.append('username', values.username);
            formData.append('password', values.password);

            const res: any = await api.post('/v1/auth/login', formData, {
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
            });

            setToken(res.access_token);
            setUser(res.user);
            message.success('登录成功');
            navigate('/');
        } catch (error: any) {
            message.error(error.response?.data?.detail || '登录失败，请检查用户名和密码');
        } finally {
            setLoading(false);
        }
    };

    const onRegister = async (values: any) => {
        if (values.password !== values.confirmPassword) {
            message.error('两次输入的密码不一致');
            return;
        }
        setLoading(true);
        try {
            await api.post('/v1/users/register', {
                username: values.username,
                password: values.password
            });
            message.success('注册成功，请登录');
            setActiveTab('login');
        } catch (error: any) {
            message.error(error.response?.data?.detail || '注册失败');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', background: '#f0f2f5' }}>
            <Card style={{ width: 400, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
                <div style={{ textAlign: 'center', marginBottom: 24 }}>
                    <Title level={3}>AutoModeling</Title>
                    <div style={{ color: '#8c8c8c' }}>自动化风控建模与策略平台</div>
                </div>

                <Tabs activeKey={activeTab} onChange={setActiveTab} centered items={[
                    { key: 'login', label: '登录' },
                    { key: 'register', label: '注册' }
                ]} />

                {activeTab === 'login' ? (
                    <Form name="login" onFinish={onLogin} layout="vertical" size="large">
                        <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
                            <Input prefix={<UserOutlined />} placeholder="用户名" />
                        </Form.Item>
                        <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
                            <Input.Password prefix={<LockOutlined />} placeholder="密码" />
                        </Form.Item>
                        <Form.Item>
                            <Button type="primary" htmlType="submit" block loading={loading}>
                                登录
                            </Button>
                        </Form.Item>
                    </Form>
                ) : (
                    <Form name="register" onFinish={onRegister} layout="vertical" size="large">
                        <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
                            <Input prefix={<UserOutlined />} placeholder="用户名" />
                        </Form.Item>
                        <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
                            <Input.Password prefix={<LockOutlined />} placeholder="密码" />
                        </Form.Item>
                        <Form.Item name="confirmPassword" rules={[{ required: true, message: '请确认密码' }]}>
                            <Input.Password prefix={<LockOutlined />} placeholder="确认密码" />
                        </Form.Item>
                        <Form.Item>
                            <Button type="primary" htmlType="submit" block loading={loading}>
                                注册
                            </Button>
                        </Form.Item>
                    </Form>
                )}
            </Card>
        </div>
    );
};

export default LoginPage;
